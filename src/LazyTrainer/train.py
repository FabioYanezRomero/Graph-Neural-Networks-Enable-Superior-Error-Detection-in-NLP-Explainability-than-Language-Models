"""LazyTrainer train.py

Entry script to train GNNs using LazyGraphDataset.  Mirrors the CLI of the
original Clean_Code/GNN_Training/train.py but avoids pre-loading all graphs.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Local imports – models and utils come from original package, dataset from ours
# ---------------------------------------------------------------------------
from src.gnn_training.training import GNN_Classifier
from src.gnn_training.training import (
    create_optimizer,
    create_scheduler,
    get_device,
    save_metrics,
    set_seed,
)
from src.gnn_training.training import load_graph_data

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


# ------------------------
# Default arguments for debugging
# ------------------------
default_args = [
    '--dataset_name', 'stanfordnlp/sst2',
    '--data_dir', 'outputs/pyg_graphs/stanfordnlp/sst2',
    '--module', 'GATConv',
    '--batch_size', '32',
    '--num_epochs', '5',
    '--cuda',
    '--graph_type', 'constituency',
    '--dropout', '0.5',
    '--seed', '42'
    # Add or remove arguments as needed for your debugging
]

def parse_args() -> argparse.Namespace:  # noqa: D401
    parser = argparse.ArgumentParser(description="Lazy GNN trainer (memory-efficient)")

    # Data
    parser.add_argument("--dataset_name", type=str, default="stanfordnlp/sst2")
    parser.add_argument("--data_dir", type=str, required=True, help="Folder with PyG graph files")
    parser.add_argument("--label_source", type=str, default="llm", choices=["original", "llm"])
    parser.add_argument("--graph_type", type=str, required=True, choices=["syntactic", "constituency"], help="Type of graph to train on (syntactic or constituency)")

    # Model
    parser.add_argument("--module", type=str, default="GCNConv",
                        choices=["GCNConv", "GATConv", "GraphConv", "SAGEConv", "RGCNConv", "RGATConv"])
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--layer_norm", action="store_true")
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--pooling", type=str, default="max", choices=["max", "mean", "add"])
    parser.add_argument("--num_relations", type=int, default=3)

    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr_scheduler", type=str, default="linear", choices=["linear", "cosine", "constant"])
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--output_dir", type=str, default="outputs/output_lazy")

    import sys
    if len(sys.argv) > 1:
        return parser.parse_args()
    else:
        print("[INFO] No CLI arguments detected, using default_args for debugging.")
        return parser.parse_args(default_args)


# ---------------------------------------------------------------------------
# Training / evaluation helpers (adapted from original)
# ---------------------------------------------------------------------------

def train_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    device: torch.device,
    criterion: torch.nn.Module,
    fp16: bool = False,
    scaler: torch.cuda.amp.GradScaler | None = None,
):
    model.train()
    total_loss = 0.0
    all_true, all_pred = [], []

    bar = tqdm(dataloader, desc="Training")
    for step, batch in enumerate(bar):
        batch = batch.to(device)
        optimizer.zero_grad()

        if fp16 and scaler is not None:
            with torch.cuda.amp.autocast():
                logits = _forward(model, batch)
                loss = criterion(logits, batch.y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = _forward(model, batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        all_true.extend(batch.y.detach().cpu().tolist())
        all_pred.extend(torch.argmax(logits, 1).detach().cpu().tolist())
        bar.set_postfix(loss=total_loss / (step + 1))

    return total_loss / len(dataloader), all_true, all_pred


def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
    fp16: bool = False,
):
    model.eval()
    total_loss = 0.0
    all_true, all_pred = [], []
    with torch.no_grad():
        bar = tqdm(dataloader, desc="Evaluating")
        for step, batch in enumerate(bar):
            batch = batch.to(device)
            if fp16:
                with torch.cuda.amp.autocast():
                    logits = _forward(model, batch)
                    loss = criterion(logits, batch.y)
            else:
                logits = _forward(model, batch)
                loss = criterion(logits, batch.y)
            total_loss += loss.item()
            all_true.extend(batch.y.detach().cpu().tolist())
            all_pred.extend(torch.argmax(logits, 1).detach().cpu().tolist())
            bar.set_postfix(loss=total_loss / (step + 1))
    return total_loss / len(dataloader), all_true, all_pred


def _forward(model, batch):  # shared between train/eval
    if hasattr(batch, "edge_type") and "R" in model.__class__.__name__:
        return model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
    return model(batch.x, batch.edge_index, batch.batch)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    # Print all arguments for debugging
    print("Arguments:")
    print(json.dumps(vars(args), indent=2))
    set_seed(args.seed)

    # output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use consolidated output path
    lazy_base = os.path.join("outputs/output_lazy", args.graph_type)
    os.makedirs(lazy_base, exist_ok=True)
    run_dir = os.path.join(lazy_base, f"{args.dataset_name.replace('/', '_')}_{args.module}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Device
    print("torch.cuda.is_available():", torch.cuda.is_available())
    device = get_device(args.cuda)
    print("Using device:", device)

    # Data (simple split assumption: train / validation / test subdirs)
    split_paths: Dict[str, str] = {}
    for split in ("train", "validation", "test"):
        p = os.path.join(args.data_dir, args.graph_type, split)
        if os.path.isdir(p):
            split_paths[split] = p
    # No fallback: require explicit structure
    if "train" not in split_paths:
        raise FileNotFoundError(f"Could not find training data in {os.path.join(args.data_dir, args.graph_type, 'train')}")

    train_ds, train_loader = load_graph_data(split_paths["train"], batch_size=args.batch_size, shuffle=True)
    val_loader = None
    test_loader = None
    if "validation" in split_paths:
        _, val_loader = load_graph_data(split_paths["validation"], args.batch_size, shuffle=False)
    if "test" in split_paths:
        _, test_loader = load_graph_data(split_paths["test"], args.batch_size, shuffle=False)

    # num classes (force by dataset name, warn if mismatch)
    dataset_to_num_classes = {
        "stanfordnlp/sst2": 2,
        "setfit/ag_news": 4,
        # Add more dataset mappings as needed
    }
    forced_num_classes = dataset_to_num_classes.get(args.dataset_name.lower())
    inferred_num_classes = train_ds.num_classes
    if forced_num_classes is not None:
        num_classes = forced_num_classes
        if inferred_num_classes != forced_num_classes:
            print(f"WARNING: Inferred {inferred_num_classes} classes from data, but using {forced_num_classes} for dataset '{args.dataset_name}'")
    else:
        num_classes = inferred_num_classes
        print(f"No hardcoded class count for dataset '{args.dataset_name}', using inferred value: {num_classes}")
    num_node_features = train_ds.num_node_features
    print(f"Number of classes: {num_classes}, Node features: {num_node_features}")

    # Model
    model = GNN_Classifier(
            input_dim=num_node_features,
            hidden_dim=args.hidden_dim,
            output_dim=num_classes,
            num_layers=args.num_layers,
            dropout=args.dropout,
            module=args.module,
            layer_norm=args.layer_norm,
            residual=args.residual,
            pooling=args.pooling,
        )
    model.to(device)

    print("Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = create_optimizer(model, args.learning_rate, args.weight_decay)
    scheduler = create_scheduler(
        optimizer,
        args.lr_scheduler,
        args.num_epochs * len(train_loader),
        args.warmup_steps,
    )

    criterion = CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None

    # ------------------------- loop ------------------------------------
    best_val_f1 = 0.0
    metrics = {"train_loss": [], "val_loss": [], "val_f1": []}
    for epoch in range(1, args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        train_loss, y_true, y_pred = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            criterion,
            args.fp16,
            scaler,
        )
        train_acc = accuracy_score(y_true, y_pred)
        train_f1 = f1_score(y_true, y_pred, average="weighted")
        print(f"Train loss {train_loss:.4f} acc {train_acc:.4f} f1 {train_f1:.4f}")

        # Validation / test
        if val_loader is not None:
            val_loss, vt, vp = evaluate(model, val_loader, device, criterion, args.fp16)
            val_acc = accuracy_score(vt, vp)
            val_f1 = f1_score(vt, vp, average="weighted")
            print(f"Val   loss {val_loss:.4f} acc {val_acc:.4f} f1 {val_f1:.4f}")
        else:
            val_loss = 0.0
            val_f1 = 0.0

        # Save checkpoint
        ckpt_path = os.path.join(run_dir, f"model_epoch{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print("Saved checkpoint", ckpt_path)

        # track
        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["val_f1"].append(val_f1)

        # Save classification report for val/test if available
        if val_loader is not None:
            with open(os.path.join(run_dir, f"val_report_epoch{epoch}.json"), "w") as f:
                json.dump(classification_report(vt, vp, output_dict=True), f, indent=2)

        # best
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))

    # final save
    torch.save(model.state_dict(), os.path.join(run_dir, "final_model.pt"))
    save_metrics(metrics, run_dir)
    print("Training finished. Artifacts saved to", run_dir)


if __name__ == "__main__":
    main()
