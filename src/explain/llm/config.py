from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

# Default hyperparameters for TokenSHAP
TOKEN_SHAP_DEFAULTS: Dict[str, float] = {
    "sampling_ratio": 0.1,
    "num_samples_override": None,
    "top_k_nodes": 5,
}


@dataclass(frozen=True)
class SamplingProfile:
    """Adaptive sampling configuration for TokenSHAP.

    The ``schedule`` attribute stores pairs ``(max_tokens, ratio)`` sorted in
    ascending order. The first entry whose ``max_tokens`` is greater than or
    equal to the requested token count determines the sampling ratio. The
    ``fallback`` ratio is used when no schedule entry matches.
    """

    schedule: Sequence[tuple[int, float]] = ((8, 0.5), (12, 0.1), (16, 0.01))
    fallback: float = 0.001

    def ratio_for(self, token_count: int) -> float:
        for limit, value in self.schedule:
            if token_count <= limit:
                return value
        return self.fallback


@dataclass(frozen=True)
class DatasetProfile:
    """Descriptor gathering dataset-specific explainability settings."""

    key: str
    dataset_name: str
    dataset_label: Optional[str]
    split: str
    text_field: str
    label_field: str
    num_labels: int
    checkpoint_dir: Path
    checkpoint_file: Optional[str] = None
    base_model_name: str = "google-bert/bert-base-uncased"
    backbone: Optional[str] = None
    max_tokens: int = 512  # Allow full sequences (increased from 21)
    max_length: int = 256
    max_samples: Optional[int] = None
    sampling: SamplingProfile = field(default_factory=SamplingProfile)
    output_subdir: Optional[Path] = None
    method_name: str = "token_shap_llm"
    run_id_factory: Optional[Callable[["DatasetProfile"], str]] = None
    graph_type: str = "tokens"
    insight_basename: str = "token_shap"

    def derive_backbone(self) -> str:
        if self.backbone:
            return self.backbone
        return self.key.split("/", 1)[0]

    def derive_output_dir(self, root: Path) -> Path:
        if self.output_subdir is not None:
            return root / self.output_subdir
        backbone = self.derive_backbone()
        dataset_slug = self.key.replace("/", "_")
        return root / backbone / dataset_slug

    def resolve_run_id(self) -> Optional[str]:
        if self.run_id_factory is None:
            checkpoint_parent = self.checkpoint_dir
            if self.checkpoint_file:
                checkpoint_parent = self.checkpoint_dir / self.checkpoint_file
            return checkpoint_parent.name
        return self.run_id_factory(self)

    def insight_dataset(self) -> str:
        if self.dataset_label:
            return self.dataset_label
        backbone = self.derive_backbone()
        return f"{backbone}/{self.dataset_name}"


DEFAULT_FINETUNED_ROOT = Path("outputs/finetuned_llms")
DEFAULT_INSIGHTS_ROOT = Path("outputs/insights/LLM")

# Default hyperparameters for TokenSHAP
TOKEN_SHAP_DEFAULTS: Dict[str, float] = {
    "sampling_ratio": 0.1,
    "min_samples": 50,
    "max_samples": 2048,
    "target_forward_passes": 400,  # Default target for fair comparison mode (reduced to match SubgraphX tractability)
}


def build_default_profiles(
    *,
    finetuned_root: Path = DEFAULT_FINETUNED_ROOT,
) -> Dict[str, DatasetProfile]:
    """Return repository-specific dataset profiles used for LLM explainability."""

    return {
        "setfit/ag_news": DatasetProfile(
            key="setfit/ag_news",
            dataset_name="ag_news",
            dataset_label="SetFit/ag_news",
            split="test",
            text_field="text",
            label_field="label",
            num_labels=4,
            checkpoint_dir=finetuned_root / "setfit" / "ag_news",
            checkpoint_file="model_final.pt",
            backbone="SetFit",
            output_subdir=Path("SetFit") / "ag_news",
            max_length=128,
        ),
        "stanfordnlp/sst2": DatasetProfile(
            key="stanfordnlp/sst2",
            dataset_name="stanfordnlp/sst2",
            dataset_label="stanfordnlp/sst2",
            split="validation",
            text_field="sentence",
            label_field="label",
            num_labels=2,
            checkpoint_dir=finetuned_root
            / "stanfordnlp"
            / "sst2"
            / "sst2_2025-06-04_14-52-49",
            checkpoint_file="model_final.pt",
            backbone="stanfordnlp",
            output_subdir=Path("stanfordnlp") / "sst2",
            max_length=128,
        ),
    }


@dataclass
class LLMExplainerRequest:
    """Runtime configuration required to run TokenSHAP on finetuned LLMs."""

    profile: DatasetProfile
    device: Optional[str] = None
    sampling_override: Optional[float] = None
    max_samples: Optional[int] = None
    sufficiency_threshold: float = 0.9
    top_k_nodes: int = 5
    insights_root: Path = DEFAULT_INSIGHTS_ROOT
    output_basename: Optional[str] = None
    store_raw: bool = True
    num_shards: int = 1
    shard_index: int = 0
    fair_comparison: bool = False
    target_forward_passes: int = 400  # Reduced to match SubgraphX tractability

    def resolve_device(self) -> str:
        if self.device:
            return self.device
        return "cuda" if torch_cuda_available() else "cpu"

    def max_tokens(self) -> int:
        return self.profile.max_tokens

    def max_length(self) -> int:
        return self.profile.max_length

    def effective_max_samples(self) -> Optional[int]:
        if self.max_samples is not None:
            return self.max_samples
        return self.profile.max_samples

    def sampling_ratio(self, token_count: int) -> float:
        if self.sampling_override is not None:
            return self.sampling_override
        return self.profile.sampling.ratio_for(token_count)

    def output_dir(self) -> Path:
        return self.profile.derive_output_dir(self.insights_root)

    def insight_dataset(self) -> str:
        return self.profile.insight_dataset()

    def graph_type(self) -> str:
        return self.profile.graph_type

    def output_basename_or_default(self) -> str:
        if self.output_basename:
            return self.output_basename
        return self.profile.insight_basename


def torch_cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False

