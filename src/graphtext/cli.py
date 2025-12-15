from __future__ import annotations
import argparse
import json
from pathlib import Path
import os
import subprocess
import sys
from typing import Dict

from .graphs import BaseGraphBuilder
from .graphs.base import BuildArgs
from .embeddings import FineTuner, FineTuneConfig, GraphEmbedder, EmbedGraphsConfig
from .convert import GraphBatchConverter, BatchConvertConfig
from .training import TrainerConfig, train_via_legacy
from .explain import ExplainConfig, run_explainer
from .registry import GRAPH_BUILDERS
from .metadata import log_step
from src.explain.gnn.config import DEFAULT_GNN_ROOT, DEFAULT_GRAPH_DATA_ROOT


def cmd_finetune(args: argparse.Namespace):
    cfg = FineTuneConfig(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
        fp16=args.fp16,
        lr_scheduler=args.lr_scheduler,
        warmup_steps=args.warmup_steps,
        warmup_proportion=args.warmup_proportion,
        output_dir=args.output_dir,
        seed=args.seed,
        no_cuda=args.no_cuda,
    )
    out = FineTuner().run(cfg)
    print(f"Fine-tuning completed. Output: {out}")
    log_step(
        step="finetune",
        params=vars(args),
        outputs={"results_dir": out},
    )


def cmd_build(args: argparse.Namespace):
    builder_cls = GRAPH_BUILDERS.get(args.graph_type)
    builder: BaseGraphBuilder = builder_cls()
    bargs = BuildArgs(
        graph_type=args.graph_type,
        dataset=args.dataset,
        subsets=args.subsets,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
        window_size=getattr(args, 'window_size', 5),
        model_name=getattr(args, 'model_name', None),
    )
    builder.process_dataset(bargs)
    log_step(
        step="build-graphs",
        params=vars(args),
        outputs={"output_dir": args.output_dir},
    )


def cmd_embed(args: argparse.Namespace):
    cfg = EmbedGraphsConfig(
        graph_type=args.graph_type,
        dataset_name=args.dataset_name,
        split=args.split,
        tree_dir=args.tree_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        weights_path=getattr(args, 'weights_path', None),
        device=args.device,
        batch_size=args.batch_size,
    )
    GraphEmbedder().run(cfg)
    log_step(
        step="embed",
        params=vars(args),
        outputs={"output_dir": args.output_dir},
    )


def cmd_convert(args: argparse.Namespace):
    cfg = BatchConvertConfig(
        label_source=args.label_source,
        use_pred=args.use_pred,
        hf_dataset_name=args.hf_dataset_name,
        graph_type=args.graph_type,
    )
    GraphBatchConverter().run(cfg)
    log_step(
        step="to-pyg",
        params=vars(args),
        outputs={},
    )


def cmd_train(args: argparse.Namespace):
    cfg = TrainerConfig(
        train_data_dir=args.train_data_dir,
        val_data_dir=args.val_data_dir,
        module=args.module,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
        pooling=args.pooling,
        layer_norm=args.layer_norm,
        residual=args.residual,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        patience=args.patience,
        num_workers=args.num_workers,
        cache_size=args.cache_size,
    )
    train_via_legacy(cfg)
    log_step(
        step="train",
        params=vars(args),
        outputs={},
    )


def cmd_explain(args: argparse.Namespace):
    if args.num_jobs > 1 and args.job_index == -1:
        base_tokens = sys.argv[1:]
        filtered: list[str] = []
        i = 0
        while i < len(base_tokens):
            token = base_tokens[i]
            if token in {"--num_jobs", "--job_index"}:
                i += 2
                continue
            filtered.append(token)
            i += 1

        processes = []
        for job_idx in range(args.num_jobs):
            cmd = [
                sys.executable,
                "-m",
                "src.graphtext.cli",
            ] + filtered + ["--num_jobs", str(args.num_jobs), "--job_index", str(job_idx)]
            print(f"Launching explain job {job_idx + 1}/{args.num_jobs}...")
            processes.append((job_idx, subprocess.Popen(cmd)))

        failures = False
        for job_idx, proc in processes:
            rc = proc.wait()
            if rc != 0:
                print(f"Explain job {job_idx + 1} failed with exit code {rc}.")
                failures = True

        if failures:
            raise SystemExit(1)

        print("All explain jobs finished successfully.")
        return

    method = None if args.method == "auto" else args.method
    overrides: Dict[str, float] = {}
    if args.hyperparams:
        overrides = json.loads(args.hyperparams)

    profile = args.performance_profile
    if profile == "balanced":
        profile = None

    shard_index = args.job_index if args.job_index >= 0 else 0

    cfg = ExplainConfig(
        dataset=args.dataset,
        graph_type=args.graph_type,
        backbone=args.backbone,
        split=args.split,
        method=method,
        device=args.device,
        checkpoint_name=args.checkpoint_name,
        gnn_root=Path(args.gnn_root),
        graph_data_root=Path(args.graph_data_root),
        hyperparams=overrides,
        profile=profile,
        num_shards=max(1, args.num_jobs),
        shard_index=shard_index,
    )

    output = run_explainer(cfg, progress=not args.no_progress)
    print(
        "Explainability via "
        f"{output.method} finished for {len(output.results)} graphs. "
        f"Summary saved to {output.summary_path}"
    )
    if output.raw_path:
        print(f"Raw artefacts stored at {output.raw_path}")

    log_step(
        step="explain",
        params=vars(args),
        outputs={
            "method": output.method,
            "num_graphs": len(output.results),
            "artifact_dir": str(output.artifact_dir),
            "summary_path": str(output.summary_path),
            "raw_path": str(output.raw_path) if output.raw_path else None,
            "profile": cfg.profile or "balanced",
            "num_shards": cfg.num_shards,
            "shard_index": cfg.shard_index,
        },
    )


def cmd_run(args: argparse.Namespace):
    # Simple JSON config runner for the end-to-end pipeline
    config_path = Path(args.config)
    with open(config_path, "r") as f:
        cfg = json.load(f)

    # 1) Fine-tune
    if "finetune" in cfg:
        cmd_finetune(argparse.Namespace(**cfg["finetune"]))

    # 2) Build graphs
    if "build" in cfg:
        cmd_build(argparse.Namespace(**cfg["build"]))

    # 3) Embed graphs
    if "embed" in cfg:
        cmd_embed(argparse.Namespace(**cfg["embed"]))

    # 4) Convert to PyG
    if "convert" in cfg:
        cmd_convert(argparse.Namespace(**cfg["convert"]))

    # 5) Train GNN
    if "train" in cfg:
        cmd_train(argparse.Namespace(**cfg["train"]))

    # 6) Explain
    if "explain" in cfg:
        cmd_explain(argparse.Namespace(**cfg["explain"]))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="graphtext", description="GraphText CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # finetune
    s = sub.add_parser("finetune", help="Fine-tune a HF model on a dataset")
    s.add_argument("--dataset_name", required=True)
    s.add_argument("--model_name", default="google-bert/bert-base-uncased")
    s.add_argument("--num_epochs", type=int, default=5)
    s.add_argument("--batch_size", type=int, default=16)
    s.add_argument("--learning_rate", type=float, default=1e-6)
    s.add_argument("--weight_decay", type=float, default=1e-4)
    s.add_argument("--max_length", type=int, default=128)
    s.add_argument("--fp16", action="store_true", default=True)
    s.add_argument("--lr_scheduler", choices=["linear", "constant"], default="linear")
    s.add_argument("--warmup_steps", type=int, default=0)
    s.add_argument("--warmup_proportion", type=float, default=0.1)
    s.add_argument("--output_dir", default="./outputs/llm")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--no_cuda", action="store_true")
    s.set_defaults(func=cmd_finetune)

    # build graphs
    s = sub.add_parser("build-graphs", help="Generate graphs from text dataset")
    s.add_argument("--graph_type", choices=GRAPH_BUILDERS.names(), required=True)
    s.add_argument("--dataset", required=True)
    s.add_argument("--subsets", nargs="+", default=["train", "validation", "test"])
    s.add_argument("--batch_size", type=int, default=256)
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--output_dir", default="./outputs/graphs")
    s.add_argument("--window_size", type=int, default=5, help="Window size (only for graph_type=window)")
    s.add_argument("--model_name", required=True, help="HF model or checkpoint directory for builders needing tokenization/embeddings")
    s.set_defaults(func=cmd_build)

    # embed
    s = sub.add_parser("embed", help="Attach LLM embeddings to nodes in graphs")
    s.add_argument("--graph_type", default="syntactic")
    s.add_argument("--dataset_name", default="stanfordnlp/sst2")
    s.add_argument("--split", default="validation")
    s.add_argument("--tree_dir", required=True)
    s.add_argument("--output_dir", required=True)
    s.add_argument("--model_name", default="bert-base-uncased")
    s.add_argument("--weights_path", default=os.environ.get('GRAPHTEXT_WEIGHTS_PATH', ''), help="Optional path to finetuned .pt state_dict to load into base model")
    s.add_argument("--device", default="cuda")
    s.add_argument("--batch_size", type=int, default=128)
    s.set_defaults(func=cmd_embed)

    # convert
    s = sub.add_parser("to-pyg", help="Convert NetworkX batches to PyG with labels")
    s.add_argument("--label_source", choices=["llm", "original"], default="llm")
    s.add_argument("--use_pred", action="store_true")
    s.add_argument("--hf_dataset_name", default="stanfordnlp/sst2")
    s.add_argument("--graph_type", default="syntactic")
    s.set_defaults(func=cmd_convert)

    # train
    s = sub.add_parser("train", help="Train a GNN on PyG graphs")
    s.add_argument("--train_data_dir", required=True)
    s.add_argument("--val_data_dir", required=True)
    s.add_argument("--module", default="TransformerConv")
    s.add_argument("--hidden_dim", type=int, default=128)
    s.add_argument("--num_layers", type=int, default=2)
    s.add_argument("--heads", type=int, default=4)
    s.add_argument("--dropout", type=float, default=0.5)
    s.add_argument("--pooling", choices=["mean", "max", "sum"], default="mean")
    s.add_argument("--layer_norm", action="store_true")
    s.add_argument("--residual", action="store_true")
    s.add_argument("--epochs", type=int, default=5)
    s.add_argument("--batch_size", type=int, default=16)
    s.add_argument("--learning_rate", type=float, default=1e-3)
    s.add_argument("--weight_decay", type=float, default=1e-4)
    s.add_argument("--optimizer", choices=["Adam", "AdamW"], default="Adam")
    s.add_argument("--scheduler", choices=["ReduceLROnPlateau", "StepLR", "None"], default="ReduceLROnPlateau")
    s.add_argument("--patience", type=int, default=5)
    s.add_argument("--num_workers", type=int, default=4)
    s.add_argument("--cache_size", type=int, default=0)
    s.set_defaults(func=cmd_train)

    # explain
    s = sub.add_parser("explain", help="Run explainability on a trained GNN")
    s.add_argument("--dataset", required=True)
    s.add_argument("--graph_type", required=True)
    s.add_argument("--backbone", default="SetFit")
    s.add_argument("--split", default="validation")
    s.add_argument("--method", choices=["auto", "subgraphx", "graphsvx"], default="auto")
    s.add_argument("--device")
    s.add_argument("--checkpoint_name", default="best_model.pt")
    s.add_argument("--gnn_root", default=str(DEFAULT_GNN_ROOT))
    s.add_argument("--graph_data_root", default=str(DEFAULT_GRAPH_DATA_ROOT))
    s.add_argument(
        "--hyperparams",
        help="JSON string with hyperparameter overrides, e.g. '{\"rollout\": 25}'",
    )
    s.add_argument(
        "--performance_profile",
        choices=["balanced", "fast", "quality"],
        default="balanced",
        help="Performance preset applied to explainer-specific hyperparameters.",
    )
    s.add_argument(
        "--num_jobs",
        type=int,
        default=1,
        help="Number of parallel explain processes to spawn."
    )
    s.add_argument(
        "--job_index",
        type=int,
        default=-1,
        help=argparse.SUPPRESS,
    )
    s.add_argument("--no_progress", action="store_true")
    s.set_defaults(func=cmd_explain)

    # pipeline
    s = sub.add_parser("run", help="Run full pipeline from a JSON config")
    s.add_argument("--config", required=True)
    s.set_defaults(func=cmd_run)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
