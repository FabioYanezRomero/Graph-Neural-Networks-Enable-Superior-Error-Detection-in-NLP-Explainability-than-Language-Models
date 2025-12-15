"""Inference utilities for using large language models as GNN teachers."""

from .model import QwenInferenceConfig, QwenTeacher, load_dataset_records

__all__ = [
    "QwenInferenceConfig",
    "QwenTeacher",
    "load_dataset_records",
]
