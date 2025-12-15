"""
Core explainability insights package.

This package hosts ingestion, metrics, and reporting utilities that unify
outputs from multiple explainability methods (e.g., GraphSVX, SubgraphX for GNNs,
TokenSHAP for LLMs).

Modules:
    records: canonical dataclasses representing explanations and coalitions.
    readers: parsers for raw explanation artefacts (JSON/CSV bundles).
    metrics: reusable computations for faithfulness, sparsity, stability, etc.
    reporting: helpers to aggregate and persist analysis results.
    providers: graph artifact providers for GNN explanations.
    llm_providers: token/text providers for LLM explanations.
    cli: simple entry point to drive the workflow from the command line.
"""

from .records import Coalition, ExplanationRecord, RelatedPrediction
from .metrics import (
    fidelity_plus,
    fidelity_minus,
    faithfulness,
    insertion_auc,
    deletion_auc,
    stability_average,
)

try:
    from .llm_providers import LLMExplanationProvider, TokenInfo
    __all__ = [
        "Coalition",
        "ExplanationRecord",
        "RelatedPrediction",
        "fidelity_plus",
        "fidelity_minus",
        "faithfulness",
        "insertion_auc",
        "deletion_auc",
        "stability_average",
        "TokenInfo",
        "LLMExplanationProvider",
    ]
except ImportError:
    __all__ = [
        "Coalition",
        "ExplanationRecord",
        "RelatedPrediction",
        "fidelity_plus",
        "fidelity_minus",
        "faithfulness",
        "insertion_auc",
        "deletion_auc",
        "stability_average",
    ]
