from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Dict, Optional


def _as_path(value: Path | str) -> Path:
    return value if isinstance(value, Path) else Path(value)

# Root folders for serialized artefacts. The defaults match the repository layout.
DEFAULT_GNN_ROOT = Path("/app/outputs/gnn_models")
DEFAULT_GRAPH_DATA_ROOT = Path("/app/outputs/pyg_graphs")

# Graph builders that produce hierarchical structures (tree-like) where SubgraphX
# tends to provide the most faithful explanations.
HIERARCHICAL_GRAPH_TYPES = {"syntactic", "constituency"}


@dataclass
class ExplainerRequest:
    """Configuration payload used to build the desired explainer."""

    dataset: str
    graph_type: str
    backbone: str = "SetFit"
    split: str = "validation"
    method: Optional[str] = None
    device: Optional[str] = None
    checkpoint_name: str = "best_model.pt"
    gnn_root: Path = DEFAULT_GNN_ROOT
    graph_data_root: Path = DEFAULT_GRAPH_DATA_ROOT
    hyperparams: Dict[str, float] = field(default_factory=dict)
    profile: Optional[str] = None
    num_shards: int = 1
    shard_index: int = 0
    fair_comparison: bool = False

    def resolved_method(self) -> str:
        """Infer the explainer to use when not explicitly provided."""
        if self.method:
            return self.method.lower()
        if self.graph_type.lower() in HIERARCHICAL_GRAPH_TYPES:
            return "subgraphx"
        return "graphsvx"

    @property
    def dataset_subpath(self) -> Path:
        parts = list(PurePosixPath(self.dataset).parts)
        if parts and parts[0].lower() == self.backbone.lower():
            parts = parts[1:]
        return Path(*parts) if parts else Path()

    @property
    def dataset_root(self) -> Path:
        return _as_path(self.graph_data_root) / self.backbone / self.dataset_subpath

    @property
    def graph_split_root(self) -> Path:
        return self.dataset_root / self.split / self.graph_type


# Default hyperparameters sourced from the official repositories.
# SubgraphX (Dive Into Graph)
SUBGRAPHX_DEFAULTS: Dict[str, float] = {
    "num_hops": 1,
    "rollout": 50,
    "min_atoms": 2,
    "c_puct": 10.0,
    "expand_atoms": 2,
    "local_radius": 1,
    "sample_num": 5,
    "max_nodes": 15,
}

# GraphSVX (original repository)
GRAPH_SVX_DEFAULTS: Dict[str, float] = {
    "sampling_ratio": 0.1,
    "num_samples_override": 512,
    "keep_special_tokens": 1,
    "top_k_nodes": 10,
}


def clone_request_with_method(request: ExplainerRequest, method: str) -> ExplainerRequest:
    return ExplainerRequest(
        dataset=request.dataset,
        graph_type=request.graph_type,
        backbone=request.backbone,
        split=request.split,
        method=method,
        device=request.device,
        checkpoint_name=request.checkpoint_name,
        gnn_root=_as_path(request.gnn_root),
        graph_data_root=_as_path(request.graph_data_root),
        hyperparams=dict(request.hyperparams),
        profile=request.profile,
        num_shards=request.num_shards,
        shard_index=request.shard_index,
        fair_comparison=request.fair_comparison,
    )


SUBGRAPHX_PROFILES: Dict[str, Dict[str, float]] = {
    "fast": {
        "num_hops": 1,
        "max_nodes": 5,
        "rollout": 20,
        "sample_num": 2,
        "expand_atoms": 1,
        "local_radius": 0,
    },
    "quality": {
        "num_hops": 2,
        "max_nodes": 15,
        "rollout": 50,
        "sample_num": 8,
        "expand_atoms": 3,
        "local_radius": 2,
    },
}


GRAPH_SVX_PROFILES: Dict[str, Dict[str, float]] = {
    "fast": {
        "sampling_ratio": 0.05,
        "num_samples_override": 256,
        "top_k_nodes": 5,
    },
    "quality": {
        "sampling_ratio": 0.2,
        "num_samples_override": 1024,
        "top_k_nodes": 15,
    },
}
