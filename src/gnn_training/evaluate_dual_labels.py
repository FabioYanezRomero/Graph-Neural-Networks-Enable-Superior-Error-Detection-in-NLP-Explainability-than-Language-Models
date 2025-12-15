#!/usr/bin/env python3
"""
Evaluate a trained GNN checkpoint against both the teacher (LLM) labels and
the original dataset labels stored alongside each graph (when available).

Example:
    python -m src.gnn_training.evaluate_dual_labels \
        --run-dirs outputs/gnn_models/SetFit/ag_news/skipgrams \
        --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from .eval_dataset import load_graph_stream
from .gnn_eval_model import GNNClassifier


def _read_json(path: Path) -> Dict:
    with path.open("r") as handle:
        return json.load(handle)


def _safe_divide(num: float, den: float) -> float:
    return num / den if den != 0 else 0.0


def _compute_metrics(y_true: List[int], y_pred: List[int]) -> Optional[Dict[str, float]]:
    if not y_true:
        return None

    true_arr = np.asarray(y_true, dtype=np.int64)
    pred_arr = np.asarray(y_pred, dtype=np.int64)
    total = len(true_arr)

    accuracy = float(np.mean(true_arr == pred_arr))
    labels, counts = np.unique(true_arr, return_counts=True)

    precision_weighted = 0.0
    recall_weighted = 0.0
    f1_weighted = 0.0

    for label, support in zip(labels, counts):
        true_mask = true_arr == label
        pred_mask = pred_arr == label

        tp = np.logical_and(true_mask, pred_mask).sum()
        fp = np.logical_and(~true_mask, pred_mask).sum()
        fn = np.logical_and(true_mask, ~pred_mask).sum()

        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2 * precision * recall, precision + recall) if (precision + recall) else 0.0

        weight = support / total
        precision_weighted += precision * weight
        recall_weighted += recall * weight
        f1_weighted += f1 * weight

    return {
        "accuracy": accuracy,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "support": total,
    }


def _resolve_data_dir(config: Dict, split: str, override: Optional[str]) -> str:
    if override:
        return override

    split_lower = split.lower()
    key_map = {
        "train": "train_data_dir",
        "training": "train_data_dir",
        "val": "val_data_dir",
        "validation": "val_data_dir",
        "dev": "val_data_dir",
        "test": "test_data_dir",
    }
    key = key_map.get(split_lower, f"{split_lower}_data_dir")
    data_dir = config.get(key)
    if not data_dir:
        raise ValueError(f"No data_dir configured for split '{split}' (looked up '{key}')")
    return data_dir


def _build_model(config: Dict, input_dim: int, num_classes: int, device: torch.device) -> GNNClassifier:
    model = GNNClassifier(
        input_dim=input_dim,
        hidden_dim=config["hidden_dim"],
        output_dim=num_classes,
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        module=config.get("module", "GCNConv"),
        layer_norm=config.get("layer_norm", False),
        residual=config.get("residual", False),
        pooling=config.get("pooling", "mean"),
        heads=config.get("heads", 1),
    )
    return model.to(device)


def _load_state_dict(checkpoint_path: Path, device: torch.device) -> Dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def _infer_output_dim(state_dict: Dict) -> int:
    classifier_weights = [
        value
        for key, value in state_dict.items()
        if key.startswith("classifier") and key.endswith(".weight") and value.ndim == 2
    ]
    if not classifier_weights:
        raise ValueError("Could not infer output dimension from checkpoint.")
    final_layer = classifier_weights[-1]
    return final_layer.size(0)


def evaluate_run(
    run_dir: Path,
    *,
    split: str,
    checkpoint_name: str,
    data_dir_override: Optional[str],
    batch_size_override: Optional[int],
    num_workers_override: Optional[int],
    device: torch.device,
    output_name: Optional[str],
) -> Path:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {run_dir}")
    config = _read_json(config_path)

    data_dir = _resolve_data_dir(config, split, data_dir_override)
    batch_size = batch_size_override or config.get("batch_size", 1)
    num_workers = (
        num_workers_override if num_workers_override is not None else config.get("num_workers", 0)
    )

    if num_workers not in (0, None):
        print("Warning: num_workers ignored for streaming evaluation (always 0).")

    print(f"[load] streaming data from {data_dir} (batch_size={batch_size})")
    summary, loader = load_graph_stream(
        data_dir=data_dir,
        batch_size=batch_size,
    )

    checkpoint_path = run_dir / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint '{checkpoint_name}' not found in {run_dir}")
    state_dict = _load_state_dict(checkpoint_path, device)
    output_dim = _infer_output_dim(state_dict)

    print(f"[model] building model with input_dim={summary.num_node_features}, output_dim={output_dim}")
    model = _build_model(config, summary.num_node_features, output_dim, device)
    model.load_state_dict(state_dict)
    print(f"[model] loaded checkpoint {checkpoint_path.name} onto {device}")

    teacher_preds: List[int] = []
    teacher_labels: List[int] = []
    gold_preds: List[int] = []
    gold_labels: List[int] = []

    model.eval()
    total_batches = len(loader) if hasattr(loader, "__len__") and len(loader) > 0 else None
    progress_bar = tqdm(
        loader,
        total=total_batches,
        desc=f"Evaluating {run_dir.name}",
        unit="batch",
        leave=False,
    )
    with torch.no_grad():
        for batch in progress_bar:
            batch = batch.to(device)
            logits = model(batch)
            preds = torch.argmax(logits, dim=1).cpu()

            labels = batch.y.view(-1).cpu()
            teacher_preds.extend(preds.tolist())
            teacher_labels.extend(labels.tolist())

            if hasattr(batch, "true_label"):
                true_labels = batch.true_label.view(-1).cpu()
                valid_mask = true_labels >= 0
                if valid_mask.any():
                    gold_preds.extend(preds[valid_mask].tolist())
                    gold_labels.extend(true_labels[valid_mask].tolist())
    progress_bar.close()

    teacher_metrics = _compute_metrics(teacher_labels, teacher_preds)
    gold_metrics = _compute_metrics(gold_labels, gold_preds)

    num_examples = summary.num_examples if summary.num_examples else len(teacher_labels)
    num_classes = max(summary.num_classes, output_dim)

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": checkpoint_name,
        "split": split,
        "data_dir": data_dir,
        "num_examples": num_examples,
        "teacher_metrics": teacher_metrics,
        "gold_metrics": gold_metrics,
        "num_classes": num_classes,
    }

    output_filename = output_name or f"dual_eval_{split.lower()}.json"
    output_path = run_dir / output_filename
    with output_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[ok] saved {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate GNN checkpoints against teacher and gold labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="One or more directories containing config.json and checkpoints.",
    )
    parser.add_argument("--split", default="test", help="Which dataset split to evaluate.")
    parser.add_argument(
        "--checkpoint",
        default="best_model.pth",
        help="Checkpoint filename relative to each run directory.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override data directory (otherwise inferred from config).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override evaluation batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override number of worker processes used by the PyG DataLoader.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device to run on (defaults to cuda if available).",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Optional filename for the saved summary JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    for run_dir_str in args.run_dirs:
        run_dir = Path(run_dir_str).resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        evaluate_run(
            run_dir,
            split=args.split,
            checkpoint_name=args.checkpoint,
            data_dir_override=args.data_dir,
            batch_size_override=args.batch_size,
            num_workers_override=args.num_workers,
            device=device,
            output_name=args.output_name,
        )


if __name__ == "__main__":
    main()
