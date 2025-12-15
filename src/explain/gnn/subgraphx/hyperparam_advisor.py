from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import math

import torch
from torch_geometric.data import Data

from src.explain.gnn.config import SUBGRAPHX_DEFAULTS

INT_PARAM_KEYS = {
    "num_hops",
    "rollout",
    "min_atoms",
    "expand_atoms",
    "local_radius",
    "sample_num",
    "max_nodes",
}
FLOAT_PARAM_KEYS = {"c_puct"}


@dataclass(frozen=True)
class ArchitectureSpec:
    """Minimal description of the trained GNN relevant for SubgraphX tuning."""

    num_layers: int
    module: str
    heads: int


@dataclass(frozen=True)
class GraphStats:
    """Lightweight summary of a graph needed for heuristic hyperparameter tuning."""

    num_nodes: int
    num_edges: int
    avg_degree: float
    density: float
    max_degree: float
    median_degree: float

    @classmethod
    def from_data(cls, data: Data) -> "GraphStats":
        num_nodes = int(getattr(data, "num_nodes", 0))
        edge_index = getattr(data, "edge_index", None)
        if edge_index is not None:
            edge_index = edge_index.detach()
            if edge_index.is_cuda:
                edge_index = edge_index.cpu()
            num_edges = int(edge_index.size(1))
            if num_nodes > 0 and num_edges > 0:
                degrees = torch.bincount(edge_index[0], minlength=num_nodes)
                degrees = degrees.to(torch.float32)
                max_degree = float(degrees.max().item())
                median_degree = float(degrees.median().item())
            else:
                max_degree = 0.0
                median_degree = 0.0
        else:
            num_edges = int(getattr(data, "num_edges", 0))
            max_degree = 0.0
            median_degree = 0.0
        safe_nodes = max(num_nodes, 1)
        avg_degree = num_edges / safe_nodes
        directed_possible = max(num_nodes * max(num_nodes - 1, 1), 1)
        density = float(num_edges) / directed_possible
        return cls(
            num_nodes=num_nodes,
            num_edges=num_edges,
            avg_degree=avg_degree,
            density=density,
            max_degree=max_degree,
            median_degree=median_degree,
        )

    @property
    def is_sparse(self) -> bool:
        return self.avg_degree < 2.5

    @property
    def is_dense(self) -> bool:
        return self.density > 0.2


class SubgraphXHyperparameterAdvisor:
    """Suggests SubgraphX hyperparameters conditioned on graph/model characteristics."""

    def __init__(
        self,
        architecture: ArchitectureSpec,
        locked_params: Optional[Dict[str, float]] = None,
        base_defaults: Optional[Dict[str, float]] = None,
    ) -> None:
        self.architecture = architecture
        self.base_defaults = self._cast_params(base_defaults or SUBGRAPHX_DEFAULTS)
        self.locked_params = self._cast_params(locked_params or {})
        # Ensure locked values override defaults immediately
        for key, value in self.locked_params.items():
            self.base_defaults[key] = value

    def suggest(self, data: Data) -> Dict[str, float]:
        stats = GraphStats.from_data(data)
        params = dict(self.base_defaults)

        if "num_hops" not in self.locked_params:
            params["num_hops"] = self._suggest_num_hops(stats)

        if "max_nodes" not in self.locked_params:
            params["max_nodes"] = self._suggest_max_nodes(stats, params["num_hops"])

        if "rollout" not in self.locked_params:
            params["rollout"] = self._suggest_rollout(stats, params["max_nodes"])

        if "sample_num" not in self.locked_params:
            params["sample_num"] = self._suggest_sample_num(stats, params["max_nodes"])

        if "expand_atoms" not in self.locked_params:
            params["expand_atoms"] = self._suggest_expand_atoms(stats)

        if "min_atoms" not in self.locked_params:
            params["min_atoms"] = self._suggest_min_atoms(stats, params["max_nodes"])

        if "local_radius" not in self.locked_params:
            params["local_radius"] = self._suggest_local_radius(params["num_hops"], stats)

        if "c_puct" not in self.locked_params:
            params["c_puct"] = self._suggest_c_puct(stats, params["rollout"], params["max_nodes"])

        return self._sanitise(params, stats)

    def _cast_params(self, params: Dict[str, float]) -> Dict[str, float]:
        casted: Dict[str, float] = {}
        for key, value in params.items():
            if value is None:
                continue
            if key in INT_PARAM_KEYS:
                casted[key] = int(value)
            elif key in FLOAT_PARAM_KEYS:
                casted[key] = float(value)
            else:
                casted[key] = value
        return casted

    def _suggest_num_hops(self, stats: GraphStats) -> int:
        layers = max(self.architecture.num_layers, 1)
        module = self.architecture.module.lower()
        hop_multiplier = 1
        if "tag" in module or "cheb" in module:
            hop_multiplier = 2
        elif "unet" in module:
            hop_multiplier = 2
        elif module in {"sgconv", "agnnconv"}:
            hop_multiplier = 2
        hops = layers * hop_multiplier
        if module in {"transformerconv", "gatconv", "gatv2conv"} and self.architecture.heads > 1:
            hops += 1
        if stats.num_nodes <= 8:
            hops = min(hops, 2)
        return max(1, min(int(hops), 6))

    def _suggest_max_nodes(self, stats: GraphStats, num_hops: int) -> int:
        if stats.num_nodes <= 4:
            return max(stats.num_nodes, 2)
        branching = min(max(stats.avg_degree, 1.3), 6.0)
        if stats.median_degree > 0:
            skew = stats.avg_degree / max(stats.median_degree, 1e-6)
            if skew > 1.5:
                # Heavy-tailed degrees: prioritise the typical neighbourhood over hubs.
                branching = max(branching * 0.9, stats.median_degree + 0.8)
        if stats.max_degree > 0 and stats.avg_degree > 0:
            hub_ratio = stats.max_degree / max(stats.avg_degree, 1e-6)
            if hub_ratio > 2.5:
                branching = min(
                    branching + min(hub_ratio * 0.25, 2.0),
                    max(4.0, math.sqrt(stats.max_degree) + 1.0),
                )
        coverage = 1.0
        frontier = 1.0
        for _ in range(num_hops):
            frontier *= branching
            coverage += frontier
            branching = max(branching * 0.85, 1.2)
        estimate = int(coverage)
        absolute_cap = 18 + num_hops * 4
        if stats.num_nodes > 60:
            absolute_cap = 25 + num_hops * 5
        if stats.max_degree > 0:
            hub_adjusted_cap = int(max(10, stats.max_degree * (1 + 0.15 * num_hops)))
            absolute_cap = min(absolute_cap, hub_adjusted_cap)
        estimate = min(estimate, stats.num_nodes, absolute_cap)
        if stats.is_dense:
            dense_cap = max(6, int(stats.num_nodes * 0.35))
            estimate = min(estimate, dense_cap)
        if stats.is_sparse:
            estimate = max(estimate, min(stats.num_nodes, 6 + num_hops * 3))
        if stats.median_degree > 0 and stats.median_degree < 1.2:
            estimate = max(estimate, min(stats.num_nodes, 4 + num_hops * 2))
        estimate = max(3, estimate)
        return max(2, min(estimate, stats.num_nodes))

    def _suggest_rollout(self, stats: GraphStats, max_nodes: int) -> int:
        base = 22 + max_nodes * (1.5 if stats.is_dense else 1.2)
        if stats.num_nodes > 80:
            base *= 1.2
        if stats.is_sparse:
            base *= 0.9
        return max(20, min(int(base), 200))

    def _suggest_sample_num(self, stats: GraphStats, max_nodes: int) -> int:
        base = 3 if max_nodes <= 10 else 4
        if max_nodes > 18:
            base += 1
        if max_nodes > 30:
            base += 1
        if stats.is_dense:
            base += 1
        if stats.is_sparse and base > 3:
            base -= 1
        return max(2, min(int(base), 8))

    def _suggest_expand_atoms(self, stats: GraphStats) -> int:
        if stats.avg_degree < 2:
            return 1
        if stats.avg_degree < 4:
            return 2
        return 3

    def _suggest_min_atoms(self, stats: GraphStats, max_nodes: int) -> int:
        if max_nodes <= 3:
            return 2
        ratio = 0.15 if stats.num_nodes > 50 else 0.25
        candidate = int(max(2, min(max_nodes - 1, max_nodes * ratio)))
        if stats.is_sparse and candidate > 3:
            candidate = 3
        return max(2, min(candidate, max_nodes))

    def _suggest_local_radius(self, num_hops: int, stats: GraphStats) -> int:
        radius = max(0, num_hops - 1)
        if stats.is_dense:
            radius = min(radius + 1, num_hops)
        if stats.is_sparse:
            radius = max(0, radius - 1)
        return min(radius, 3)

    def _suggest_c_puct(self, stats: GraphStats, rollout: int, max_nodes: int) -> float:
        base = 8.0
        if stats.is_dense:
            base += 2.0
        if stats.avg_degree > 5:
            base += 1.5
        if stats.avg_degree < 2:
            base -= 1.5
        if max_nodes > 25:
            base += 1.0
        if rollout > 120:
            base += 1.0
        if stats.num_nodes < 12:
            base -= 0.5
        return float(max(3.0, min(base, 15.0)))

    def _sanitise(self, params: Dict[str, float], stats: GraphStats) -> Dict[str, float]:
        cleaned: Dict[str, float] = {}
        for key in SUBGRAPHX_DEFAULTS.keys():
            value = params.get(key, SUBGRAPHX_DEFAULTS[key])
            if key in INT_PARAM_KEYS:
                cleaned[key] = int(value)
            elif key in FLOAT_PARAM_KEYS:
                cleaned[key] = float(value)
            else:
                cleaned[key] = value
        cleaned["num_hops"] = max(1, cleaned["num_hops"])
        cleaned["max_nodes"] = max(2, min(cleaned["max_nodes"], stats.num_nodes or cleaned["max_nodes"]))
        cleaned["min_atoms"] = max(2, min(cleaned["min_atoms"], cleaned["max_nodes"]))
        cleaned["expand_atoms"] = max(1, min(cleaned["expand_atoms"], cleaned["max_nodes"]))
        cleaned["local_radius"] = max(0, min(cleaned["local_radius"], cleaned["num_hops"]))
        cleaned["sample_num"] = max(1, cleaned["sample_num"])
        cleaned["rollout"] = max(10, cleaned["rollout"])
        return cleaned

    def sanitise_for_graph(self, params: Dict[str, float], data: Data) -> Dict[str, float]:
        """Project arbitrary hyperparameters onto valid ranges for the supplied graph."""

        stats = GraphStats.from_data(data)
        return self._sanitise(dict(params), stats)
