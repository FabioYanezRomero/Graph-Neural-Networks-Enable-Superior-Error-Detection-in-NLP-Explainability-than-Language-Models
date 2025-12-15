from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from src.explain.gnn import (
    GRAPH_SVX_DEFAULTS,
    SUBGRAPHX_DEFAULTS,
    ExplainerOutput,
    ExplainerRequest,
    run as run_gnn_explainer,
)
from src.explain.gnn.config import DEFAULT_GNN_ROOT, DEFAULT_GRAPH_DATA_ROOT, HIERARCHICAL_GRAPH_TYPES


@dataclass
class ExplainConfig:
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

    def to_request(self) -> ExplainerRequest:
        return ExplainerRequest(
            dataset=self.dataset,
            graph_type=self.graph_type,
            backbone=self.backbone,
            split=self.split,
            method=self.method,
            device=self.device,
            checkpoint_name=self.checkpoint_name,
            gnn_root=self.gnn_root,
            graph_data_root=self.graph_data_root,
            hyperparams=self.hyperparams,
            profile=self.profile,
            num_shards=self.num_shards,
            shard_index=self.shard_index,
            fair_comparison=self.fair_comparison,
        )


def default_hyperparams(method: str) -> Dict[str, float]:
    method = method.lower()
    if method == "subgraphx":
        return dict(SUBGRAPHX_DEFAULTS)
    if method == "graphsvx":
        return dict(GRAPH_SVX_DEFAULTS)
    raise ValueError(f"Unknown explainer method: {method}")


def run_explainer(cfg: ExplainConfig, *, progress: bool = True) -> ExplainerOutput:
    request = cfg.to_request()
    return run_gnn_explainer(request, progress=progress)


__all__ = [
    "ExplainConfig",
    "ExplainerOutput",
    "HIERARCHICAL_GRAPH_TYPES",
    "default_hyperparams",
    "run_explainer",
]
