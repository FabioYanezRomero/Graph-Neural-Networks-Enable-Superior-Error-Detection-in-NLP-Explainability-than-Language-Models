from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import math
import torch
from torch_geometric.data import Data

from src.explain.gnn.config import GRAPH_SVX_DEFAULTS

INT_PARAM_KEYS = {"num_samples_override", "top_k_nodes"}
FLOAT_PARAM_KEYS = {"sampling_ratio"}
BOOL_PARAM_KEYS = {"keep_special_tokens"}


@dataclass(frozen=True)
class ArchitectureSpec:
    """Minimal GNN architecture traits used for GraphSVX heuristics."""

    num_layers: int
    module: str
    heads: int


@dataclass(frozen=True)
class GraphContext:
    """Dataset-level metadata that influences GraphSVX defaults."""

    dataset: str
    graph_type: str
    backbone: str


@dataclass(frozen=True)
class GraphStats:
    """Lightweight summary of an individual graph."""

    num_nodes: int
    num_edges: int
    avg_degree: float
    density: float

    @classmethod
    def from_data(cls, data: Data) -> "GraphStats":
        num_nodes = int(getattr(data, "num_nodes", 0) or (data.x.size(0) if hasattr(data, "x") else 0))
        edge_index = getattr(data, "edge_index", None)
        if edge_index is not None:
            if edge_index.is_cuda:
                edge_index = edge_index.detach().cpu()
            else:
                edge_index = edge_index.detach()
            num_edges = int(edge_index.size(1))
        else:
            num_edges = int(getattr(data, "num_edges", 0))
        safe_nodes = max(num_nodes, 1)
        avg_degree = float(num_edges) / safe_nodes
        density_denominator = max(safe_nodes * max(safe_nodes - 1, 1), 1)
        density = float(num_edges) / density_denominator
        return cls(
            num_nodes=num_nodes,
            num_edges=num_edges,
            avg_degree=avg_degree,
            density=density,
        )

    @property
    def is_sparse(self) -> bool:
        return self.avg_degree < 2.5

    @property
    def is_dense(self) -> bool:
        return self.density > 0.2


class GraphSVXHyperparameterAdvisor:
    """Suggest per-graph GraphSVX hyperparameters based on structure and model traits."""

    def __init__(
        self,
        architecture: ArchitectureSpec,
        context: GraphContext,
        *,
        locked_params: Optional[Dict[str, float]] = None,
        base_defaults: Optional[Dict[str, float]] = None,
    ) -> None:
        self.architecture = architecture
        self.context = context
        self.base_defaults = self._cast_params(base_defaults or GRAPH_SVX_DEFAULTS)
        self.locked_params = self._cast_params(locked_params or {})
        for key, value in self.locked_params.items():
            self.base_defaults[key] = value

    def suggest(self, data: Data) -> Dict[str, float]:
        stats = GraphStats.from_data(data)
        params = dict(self.base_defaults)

        if "sampling_ratio" not in self.locked_params:
            params["sampling_ratio"] = self._suggest_sampling_ratio(stats)

        if "num_samples_override" not in self.locked_params:
            params["num_samples_override"] = self._suggest_num_samples(stats)

        if "keep_special_tokens" not in self.locked_params:
            params["keep_special_tokens"] = self._suggest_keep_special_tokens(stats)

        if "top_k_nodes" not in self.locked_params:
            params["top_k_nodes"] = self._suggest_top_k(stats)

        return self._sanitise(params, stats)

    def sanitise_for_graph(self, params: Dict[str, float], data: Data) -> Dict[str, float]:
        stats = GraphStats.from_data(data)
        return self._sanitise(dict(params), stats)

    def _cast_params(self, params: Dict[str, float]) -> Dict[str, float]:
        casted: Dict[str, float] = {}
        for key, value in params.items():
            if value is None:
                continue
            if key in INT_PARAM_KEYS:
                casted[key] = int(value)
            elif key in FLOAT_PARAM_KEYS:
                casted[key] = float(value)
            elif key in BOOL_PARAM_KEYS:
                casted[key] = bool(int(value)) if isinstance(value, (int, float)) else bool(value)
            else:
                casted[key] = value
        return casted

    def _suggest_sampling_ratio(self, stats: GraphStats) -> float:
        nodes = stats.num_nodes
        if nodes <= 12:
            ratio = 0.6
        elif nodes <= 24:
            ratio = 0.45
        elif nodes <= 48:
            ratio = 0.3
        elif nodes <= 96:
            ratio = 0.2
        else:
            ratio = 0.12

        if stats.is_sparse:
            ratio = min(ratio * 1.2, 0.85)
        if stats.is_dense:
            ratio = max(ratio * 0.85, 0.05)

        if "skip" in self.context.graph_type.lower():
            ratio = min(ratio * 1.05, 0.9)
        if "window" in self.context.graph_type.lower():
            ratio = min(ratio * 1.1, 0.95)

        return float(min(max(ratio, 0.05), 0.95))

    def _suggest_num_samples(self, stats: GraphStats) -> Optional[int]:
        content_nodes = max(stats.num_nodes - 2, 1)
        theoretical_max = 2 ** min(content_nodes, 12)
        baseline = 64 + stats.num_nodes * (4 + max(self.architecture.num_layers - 2, 0))
        if stats.is_dense:
            baseline *= 1.15
        if stats.is_sparse:
            baseline *= 0.85
        if "skip" in self.context.graph_type.lower():
            baseline *= 1.1
        if "window" in self.context.graph_type.lower():
            baseline *= 1.05
        estimate = int(min(theoretical_max, baseline, 1024))
        if estimate < 32:
            estimate = 32
        return estimate

    def _suggest_keep_special_tokens(self, stats: GraphStats) -> bool:
        graph_type = self.context.graph_type.lower()
        if "skip" in graph_type or "window" in graph_type or "syntactic" in graph_type:
            return True
        if stats.num_nodes <= 4:
            return False
        return True

    def _suggest_top_k(self, stats: GraphStats) -> int:
        if stats.num_nodes <= 10:
            return max(3, stats.num_nodes // 2)
        if stats.num_nodes <= 30:
            return max(5, int(math.ceil(stats.num_nodes * 0.3)))
        if stats.num_nodes <= 80:
            return min(25, int(math.ceil(stats.num_nodes * 0.25)))
        return min(40, int(math.ceil(stats.num_nodes * 0.2)))

    def _sanitise(self, params: Dict[str, float], stats: GraphStats) -> Dict[str, float]:
        cleaned: Dict[str, float] = {}
        for key in GRAPH_SVX_DEFAULTS.keys():
            value = params.get(key, GRAPH_SVX_DEFAULTS[key])
            if key in INT_PARAM_KEYS:
                cleaned[key] = int(value) if value is not None else None
            elif key in FLOAT_PARAM_KEYS:
                cleaned[key] = float(value)
            elif key in BOOL_PARAM_KEYS:
                cleaned[key] = bool(value)
            else:
                cleaned[key] = value
        cleaned["sampling_ratio"] = float(min(max(cleaned["sampling_ratio"], 0.01), 0.99))
        override = cleaned.get("num_samples_override")
        if override is not None:
            cleaned["num_samples_override"] = max(8, min(int(override), 2048))
        cleaned["top_k_nodes"] = max(1, min(int(cleaned["top_k_nodes"]), stats.num_nodes or int(cleaned["top_k_nodes"])))
        return cleaned
