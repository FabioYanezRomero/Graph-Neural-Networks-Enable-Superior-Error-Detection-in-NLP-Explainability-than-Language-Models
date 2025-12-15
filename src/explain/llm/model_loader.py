from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import DatasetProfile

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelBundle:
    """Container bundling the finetuned transformer model and its tokenizer."""

    model: AutoModelForSequenceClassification
    tokenizer: AutoTokenizer
    device: torch.device


def _resolve_checkpoint_path(profile: DatasetProfile) -> Path:
    """Return the checkpoint file to load for a dataset profile.

    The function first honours ``profile.checkpoint_file`` when provided. When the
    specified file is missing it falls back to discovering ``*.pt`` artefacts
    inside ``checkpoint_dir`` preferring ``model_final.pt`` whenever available.
    """

    if profile.checkpoint_file:
        candidate = profile.checkpoint_dir / profile.checkpoint_file
        if candidate.exists():
            return candidate
        LOGGER.warning("Checkpoint '%s' not found; attempting discovery.", candidate)

    final_candidate = profile.checkpoint_dir / "model_final.pt"
    if final_candidate.exists():
        return final_candidate

    pt_files = sorted(profile.checkpoint_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(
            f"No checkpoint files found under {profile.checkpoint_dir}."
        )
    return pt_files[-1]


def load_finetuned_model(
    profile: DatasetProfile,
    *,
    device: Optional[str] = None,
) -> ModelBundle:
    """Load the finetuned transformer model and tokenizer for ``profile``."""

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    tokenizer = AutoTokenizer.from_pretrained(profile.base_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        profile.base_model_name,
        num_labels=profile.num_labels,
    )

    checkpoint_path = _resolve_checkpoint_path(profile)
    state = torch.load(checkpoint_path, map_location=resolved_device)
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "weights"):
            if key in state:
                state = state[key]
                break

    model.load_state_dict(state, strict=False)
    model.to(resolved_device)
    model.eval()

    LOGGER.info(
        "Loaded finetuned model '%s' from %s", profile.base_model_name, checkpoint_path
    )
    return ModelBundle(model=model, tokenizer=tokenizer, device=resolved_device)


def load_dataset_split(profile: DatasetProfile) -> Dataset:
    """Return the HuggingFace dataset split configured in ``profile``."""

    dataset = load_dataset(profile.dataset_name, split=profile.split)
    return dataset


