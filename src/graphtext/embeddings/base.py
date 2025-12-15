from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, List
import os
import sys
import subprocess


@dataclass
class FineTuneConfig:
    dataset_name: str
    model_name: str = "google-bert/bert-base-uncased"
    num_epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 1e-6
    weight_decay: float = 1e-4
    max_length: int = 128
    fp16: bool = True
    lr_scheduler: str = "linear"
    warmup_steps: int = 0
    warmup_proportion: float = 0.1
    output_dir: str = "./outputs/llm"
    seed: int = 42
    no_cuda: bool = False


class FineTuner:
    """Wrapper for the existing fine-tuner under Clean_Code."""

    def run(self, cfg: FineTuneConfig) -> str:
        from src.finetuning import fine_tune
        config_dict = {
            'num_epochs': cfg.num_epochs,
            'learning_rate': cfg.learning_rate,
            'weight_decay': cfg.weight_decay,
            'adam_epsilon': 1e-8,
            'batch_size': cfg.batch_size,
            'model_name': cfg.model_name,
            'dataset_name': cfg.dataset_name,
            'max_length': cfg.max_length,
            'fp16': cfg.fp16,
            'lr_scheduler': cfg.lr_scheduler,
            'warmup_steps': cfg.warmup_steps,
            'warmup_proportion': cfg.warmup_proportion,
            'output_dir': cfg.output_dir,
            'cuda': (not cfg.no_cuda),
            'seed': cfg.seed,
        }
        return fine_tune(config_dict)


@dataclass
class EmbedGraphsConfig:
    graph_type: str
    dataset_name: str
    split: str
    tree_dir: str
    output_dir: str
    model_name: str = "bert-base-uncased"
    weights_path: str | None = None
    device: str = "cuda"
    batch_size: int = 128


class GraphEmbedder:
    """
    Wrapper that invokes the existing graph-embedding script. To preserve
    compatibility with its relative imports, we execute it as a module.
    """

    def run(self, cfg: EmbedGraphsConfig) -> None:
        # Execute the legacy script as a module to avoid import breakage
        # Equivalent to:
        # python -m src.Clean_Code.Graph_Embeddings.generate_graphs_with_embeddings --args
        args = [
            sys.executable,
            "-m",
            "src.embeddings.generate",
            "--graph_type", cfg.graph_type,
            "--dataset_name", cfg.dataset_name,
            "--split", cfg.split,
            "--tree_dir", cfg.tree_dir,
            "--output_dir", cfg.output_dir,
            "--model_name", cfg.model_name,
            "--weights_path", cfg.weights_path or "",
            "--device", cfg.device,
            "--batch_size", str(cfg.batch_size),
        ]
        subprocess.run(args, check=True)
