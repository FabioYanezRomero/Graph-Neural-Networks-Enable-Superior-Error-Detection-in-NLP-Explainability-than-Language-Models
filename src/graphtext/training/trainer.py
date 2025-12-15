from __future__ import annotations
from dataclasses import dataclass
import sys
import subprocess


@dataclass
class TrainerConfig:
    # Data paths should point to folders with .pt/.pkl graphs
    train_data_dir: str
    val_data_dir: str
    # Model
    module: str = "TransformerConv"
    hidden_dim: int = 128
    num_layers: int = 2
    heads: int = 4
    dropout: float = 0.5
    pooling: str = "mean"
    layer_norm: bool = False
    residual: bool = False
    # Training
    epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "Adam"
    scheduler: str = "ReduceLROnPlateau"
    patience: int = 5
    num_workers: int = 4
    cache_size: int = 0


def train_via_legacy(cfg: TrainerConfig) -> None:
    """Invoke the optimized legacy training script with the given config."""
    args = [
        sys.executable, "-m", "src.gnn_training.training",
        "--train_data_dir", cfg.train_data_dir,
        "--val_data_dir", cfg.val_data_dir,
        "--module", cfg.module,
        "--hidden_dim", str(cfg.hidden_dim),
        "--num_layers", str(cfg.num_layers),
        "--heads", str(cfg.heads),
        "--dropout", str(cfg.dropout),
        "--pooling", cfg.pooling,
        "--epochs", str(cfg.epochs),
        "--batch_size", str(cfg.batch_size),
        "--learning_rate", str(cfg.learning_rate),
        "--weight_decay", str(cfg.weight_decay),
        "--optimizer", cfg.optimizer,
        "--scheduler", cfg.scheduler,
        "--patience", str(cfg.patience),
        "--num_workers", str(cfg.num_workers),
        "--cache_size", str(cfg.cache_size),
    ]
    if cfg.layer_norm:
        args.append("--layer_norm")
    if cfg.residual:
        args.append("--residual")
    subprocess.run(args, check=True)
