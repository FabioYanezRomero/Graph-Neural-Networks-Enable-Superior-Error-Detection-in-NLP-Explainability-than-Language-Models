"""Hyperparameter advisor for TokenSHAP based on token count and dataset characteristics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass(frozen=True)
class ModelSpec:
    """Specification of the transformer model."""

    base_model_name: str
    num_labels: int
    max_length: int


@dataclass(frozen=True)
class DatasetContext:
    """Context about the dataset being explained."""

    dataset: str
    task_type: str
    backbone: str


@dataclass(frozen=True)
class SentenceStats:
    """Statistics computed from a tokenized sentence."""

    num_tokens: int
    num_chars: int
    avg_token_length: float
    max_token_length: int

    @classmethod
    def from_tokens(cls, tokens: Sequence[str]) -> "SentenceStats":
        """Compute statistics from a list of tokens."""
        num_tokens = len(tokens)
        if num_tokens == 0:
            return cls(0, 0, 0.0, 0)

        lengths = [len(tok) for tok in tokens]
        num_chars = sum(lengths)
        avg_token_length = num_chars / num_tokens
        max_token_length = max(lengths) if lengths else 0

        return cls(
            num_tokens=num_tokens,
            num_chars=num_chars,
            avg_token_length=avg_token_length,
            max_token_length=max_token_length,
        )


class TokenSHAPHyperparameterAdvisor:
    """
    Suggests adaptive hyperparameters for TokenSHAP based on token count.

    Similar to GraphSVXHyperparameterAdvisor, this class provides per-sample
    hyperparameter suggestions to balance quality and computational cost.
    """

    def __init__(
        self,
        model_spec: ModelSpec,
        context: DatasetContext,
        locked_params: Dict[str, float] | None = None,
    ):
        self.model_spec = model_spec
        self.context = context
        self.locked_params = locked_params or {}

        # Base defaults for TokenSHAP
        self.base_defaults: Dict[str, float] = {
            "sampling_ratio": 0.1,
            "min_samples": 50,
            "max_samples": 2048,
        }

    def suggest(self, tokens: Sequence[str]) -> Dict[str, float]:
        """
        Suggest hyperparameters for a given tokenized sentence.

        Args:
            tokens: List of tokens (excluding special tokens like [CLS], [SEP])

        Returns:
            Dictionary of suggested hyperparameters
        """
        stats = SentenceStats.from_tokens(tokens)
        num_tokens = stats.num_tokens

        # Start with base defaults
        params = dict(self.base_defaults)

        # Adaptive sampling ratio based on token count
        # Shorter sequences can afford higher sampling ratios
        if num_tokens <= 5:
            params["sampling_ratio"] = 0.8
        elif num_tokens <= 8:
            params["sampling_ratio"] = 0.5
        elif num_tokens <= 12:
            params["sampling_ratio"] = 0.2
        elif num_tokens <= 16:
            params["sampling_ratio"] = 0.05
        elif num_tokens <= 20:
            params["sampling_ratio"] = 0.01
        else:
            params["sampling_ratio"] = 0.005

        # Compute max combinations to inform min/max samples
        max_combinations = 2**num_tokens if num_tokens < 30 else float("inf")

        # Ensure min_samples doesn't exceed max_combinations
        if max_combinations != float("inf"):
            params["min_samples"] = min(params["min_samples"], int(max_combinations * 0.1))
            params["max_samples"] = min(params["max_samples"], int(max_combinations * 0.5))

        # Apply locked parameters (user overrides)
        params.update(self.locked_params)

        return params

    def sanitise_for_sample(
        self,
        candidate_params: Dict[str, float],
        tokens: Sequence[str],
    ) -> Dict[str, float]:
        """
        Sanitise candidate parameters for a specific sample.

        Ensures that suggested parameters are valid for the given token count.
        """
        stats = SentenceStats.from_tokens(tokens)
        params = dict(candidate_params)

        # Cap sampling_ratio to reasonable bounds
        if "sampling_ratio" in params:
            params["sampling_ratio"] = max(0.001, min(1.0, params["sampling_ratio"]))

        return params
