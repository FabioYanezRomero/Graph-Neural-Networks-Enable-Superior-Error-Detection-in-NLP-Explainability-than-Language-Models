from __future__ import annotations

from typing import List, Sequence, Set, Tuple

PROGRESSION_TOP_K: Tuple[int, ...] = (1, 3, 5, 10)

ALLOWED_AUC_FEATURES: Set[str] = {"auc_deletion_auc", "auc_insertion_auc"}

FIDELITY_FEATURES: Set[str] = {
    "fidelity_plus",
    "fidelity_minus",
    "fidelity_asymmetry",
}

CONSISTENCY_FEATURES: Set[str] = {
    "consistency_baseline_margin",
    "consistency_preservation_sufficiency",
    "consistency_preservation_necessity",
}

PROGRESSION_MASKOUT_DROP_FEATURES: Set[str] = {
    f"progression_maskout_drop_k{k}" for k in PROGRESSION_TOP_K
}
PROGRESSION_SUFFICIENCY_DROP_FEATURES: Set[str] = {
    f"progression_sufficiency_drop_k{k}" for k in PROGRESSION_TOP_K
}
PROGRESSION_CONCENTRATION_FEATURES: Set[str] = {
    f"progression_concentration_top{k}" for k in PROGRESSION_TOP_K
}

PROGRESSION_FEATURES: Set[str] = (
    PROGRESSION_MASKOUT_DROP_FEATURES
    | PROGRESSION_SUFFICIENCY_DROP_FEATURES
    | PROGRESSION_CONCENTRATION_FEATURES
)

ALLOWED_FEATURES: Set[str] = (
    ALLOWED_AUC_FEATURES | FIDELITY_FEATURES | CONSISTENCY_FEATURES | PROGRESSION_FEATURES
)


def filter_allowed_features(columns: Sequence[str]) -> List[str]:
    return [col for col in columns if col in ALLOWED_FEATURES]


def dimension_for_feature(feature: str) -> str:
    normalized = feature.lower()
    if normalized in ALLOWED_AUC_FEATURES:
        return "AUC"
    if normalized in FIDELITY_FEATURES:
        return "Fidelity"
    if normalized in CONSISTENCY_FEATURES:
        return "Consistency"
    if normalized in PROGRESSION_FEATURES:
        return "Progression"
    raise ValueError(
        f"Unrecognised analytic feature '{feature}' – expected members of the configured AUC/Fidelity/Consistency/Progression sets."
    )
