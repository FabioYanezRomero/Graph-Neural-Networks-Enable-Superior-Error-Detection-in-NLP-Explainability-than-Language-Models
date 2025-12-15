"""Fair comparison sampling strategy for TokenSHAP explainability."""

from __future__ import annotations

import math
from typing import Dict


def compute_fair_sampling_ratio(
    num_tokens: int,
    *,
    target_samples: int = 400,
    min_ratio: float = 0.001,
    max_ratio: float = 1.0,
) -> float:
    """
    Compute a fair sampling ratio based on token count.

    The formula ensures that sequences with different token counts are sampled
    with comparable computational budgets. For a sequence with `n` tokens,
    there are 2^n possible coalitions. We target a fixed number of samples
    (default: 2000 forward passes) regardless of sequence length.

    Args:
        num_tokens: Number of tokens in the sequence
        target_samples: Target number of samples to generate (default: 2000)
        min_ratio: Minimum sampling ratio
        max_ratio: Maximum sampling ratio

    Returns:
        Sampling ratio clamped between min_ratio and max_ratio
    """
    if num_tokens <= 0:
        return max_ratio

    max_combinations = 2**num_tokens
    if max_combinations <= target_samples:
        # If total combinations are less than target, sample everything
        return max_ratio

    # Compute ratio to get approximately target_samples
    ratio = target_samples / max_combinations

    # Clamp to reasonable bounds
    return max(min_ratio, min(ratio, max_ratio))


def compute_fair_hyperparams(
    num_tokens: int,
    *,
    target_samples: int = 400,
) -> Dict[str, float]:
    """
    Compute fair comparison hyperparameters for a given token count.

    Args:
        num_tokens: Number of tokens in the sequence
        target_samples: Target number of samples (default: 2000 forward passes)

    Returns:
        Dictionary of hyperparameters including sampling_ratio
    """
    sampling_ratio = compute_fair_sampling_ratio(
        num_tokens,
        target_samples=target_samples,
    )

    return {
        "sampling_ratio": sampling_ratio,
        "target_samples": target_samples,
        "num_tokens": num_tokens,
        "max_combinations": 2**num_tokens if num_tokens < 30 else float("inf"),
    }

