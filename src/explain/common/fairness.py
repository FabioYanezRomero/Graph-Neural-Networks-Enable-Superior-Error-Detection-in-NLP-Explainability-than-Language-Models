"""Shared hyperparameter advisor for fair multimodal explainability comparisons."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class FairnessConfig:
    """Configuration parameters for the fair multimodal advisor."""

    compute_budget: int = 400  # Reduced from 2000 to make SubgraphX tractable
    sparsity_ratio: float = 0.2
    graph_sampling_ratio: float = 0.25
    expand_atoms: int = 1
    c_puct: float = 10.0
    min_top_k: int = 1


class FairMultimodalHyperparameterAdvisor:
    """Provide aligned hyperparameters across explainers for fair benchmarking."""

    def __init__(self, config: FairnessConfig | None = None) -> None:
        self.config = config or FairnessConfig()

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _top_k(self, total_elements: int) -> int:
        if total_elements <= 0:
            return self.config.min_top_k
        return max(self.config.min_top_k, int(math.ceil(total_elements * self.config.sparsity_ratio)))

    def _factorise_budget(self, rollout_guess: int) -> Tuple[int, int]:
        """Return rollout/sample values whose product matches the budget."""

        budget = max(1, self.config.compute_budget)
        rollout = max(1, rollout_guess)
        sample_num = max(1, budget // rollout)
        if rollout * sample_num != budget:
            divisor = math.gcd(budget, rollout)
            rollout = divisor if divisor > 0 else 1
            sample_num = budget // rollout
        return rollout, max(1, sample_num)

    def _subgraphx_rollout(self, num_nodes: int) -> int:
        """Select a rollout value (dividing the budget) based on graph size."""

        if num_nodes <= 30:
            return 10  # 10 * 40 = 400
        if num_nodes <= 100:
            return 10  # 10 * 40 = 400
        return 8  # 8 * 50 = 400

    # ------------------------------------------------------------------
    # Public API used by explainability modules
    # ------------------------------------------------------------------
    def describe(self) -> Dict[str, float]:
        return {
            "compute_budget": float(self.config.compute_budget),
            "sparsity_ratio": float(self.config.sparsity_ratio),
            "graph_sampling_ratio": float(self.config.graph_sampling_ratio),
            "expand_atoms": float(self.config.expand_atoms),
            "c_puct": float(self.config.c_puct),
        }

    def graphsvx(self, *, num_nodes: int, keep_special_tokens: bool = True) -> Dict[str, float]:
        top_k = self._top_k(num_nodes)
        return {
            "sampling_ratio": float(self.config.graph_sampling_ratio),
            "num_samples_override": int(self.config.compute_budget),
            "keep_special_tokens": bool(keep_special_tokens),
            "top_k_nodes": max(1, min(top_k, num_nodes if num_nodes > 0 else top_k)),
        }

    def subgraphx(self, *, num_nodes: int, num_layers: int) -> Dict[str, float]:
        max_nodes = max(2, min(self._top_k(num_nodes), num_nodes if num_nodes > 0 else self._top_k(num_nodes)))
        rollout_guess = self._subgraphx_rollout(num_nodes)
        rollout, sample_num = self._factorise_budget(rollout_guess)
        num_hops = max(1, num_layers)
        return {
            "num_hops": num_hops,
            "local_radius": num_hops,
            "expand_atoms": int(self.config.expand_atoms),
            "rollout": int(rollout),
            "sample_num": int(sample_num),
            "c_puct": float(self.config.c_puct),
            "max_nodes": max_nodes,
            "min_atoms": max(1, min(max_nodes, 2)),
        }

    def tokenshap(self, *, num_tokens: int) -> Dict[str, float]:
        top_k = self._top_k(num_tokens)
        if num_tokens <= 0:
            ratio = 1.0
        else:
            essential = num_tokens
            total_subsets = max(1, (1 << num_tokens) - 1)
            effective_budget = max(0, self.config.compute_budget - essential)
            ratio = effective_budget / float(total_subsets)
            ratio = max(0.0, min(1.0, ratio))
        return {
            "sampling_ratio": ratio,
            "num_samples_override": None,
            "top_k_tokens": max(1, min(top_k, num_tokens if num_tokens > 0 else top_k)),
        }
