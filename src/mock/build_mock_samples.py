#!/usr/bin/env python3
"""Generate easy-to-read mock samples (10 correct + 10 incorrect) per module."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from use_case.feature_config import filter_allowed_features  # type: ignore


MODULE_DATASET_DIR = Path("outputs/use_case/module_datasets")
OUTPUT_DIR = Path("src/mock")
DATASETS: List[str] = ["setfit_ag_news", "stanfordnlp_sst2"]
MODULES: Dict[str, Iterable[str]] = {
    "graphsvx": ("skipgrams", "window"),
    "subgraphx": ("constituency", "syntactic"),
    "token_shap_llm": ("tokens",),
}

DROP_FEATURES = {
    "global_graph_index",
    "graph_index",
    "label",
    "label_id",
    "prediction_class",
    "prediction_confidence",
    "is_correct",
}

DISPLAY_COLUMNS = [
    "dataset_slug",
    "dataset_label",
    "method",
    "graph_type",
    "module_id",
    "global_graph_index",
    "label",
    "prediction_class",
    "prediction_confidence",
    "is_correct",
    "logistic_probability",
    "logistic_prediction",
    "logistic_accuracy",
    "top_features",
    "auc__origin_confidence",
    "auc__deletion_auc",
    "auc__insertion_auc",
    "fidelity_plus",
    "fidelity_minus",
    "consistency__baseline_margin",
    "consistency__preservation_sufficiency",
    "consistency__preservation_necessity",
    "progression__maskout_progression_drop_len",
    "progression__sufficiency_progression_drop_len",
    "progression_maskout_drop_k1",
    "progression_maskout_drop_k3",
    "progression_sufficiency_drop_k1",
    "progression_sufficiency_drop_k3",
    "progression_concentration_top1",
    "progression_concentration_top3",
    "text",
]


def load_module_dataset(dataset: str, method: str, graph: str) -> pd.DataFrame:
    csv_path = MODULE_DATASET_DIR / dataset / f"module_dataset_{dataset}_{method}_{graph}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Module dataset missing: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Module dataset empty: {csv_path}")
    return df


def load_token_rankings(dataset: str, method: str, graph: str) -> pd.DataFrame | None:
    token_path = Path("outputs/analytics/tokens") / method / dataset / f"{graph}.csv"
    if not token_path.exists():
        return None
    df = pd.read_csv(token_path)
    needed = {"global_graph_index", "ranked_tokens", "ranked_nodes", "ranked_scores"}
    available = [col for col in df.columns if col in needed]
    if not available:
        return None
    return df.loc[:, available].copy()


def parse_ranked_list(value: str, top_k: int = 10) -> str:
    if not isinstance(value, str):
        return ""
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if not parts:
        return ""
    return ", ".join(parts[:top_k])


def attach_top_features(df: pd.DataFrame, dataset: str, method: str, graph: str) -> pd.DataFrame:
    token_df = load_token_rankings(dataset, method, graph)
    if token_df is None:
        df["top_features"] = ""
        return df
    token_df = token_df.copy()
    if "ranked_tokens" in token_df:
        token_df["top_features"] = token_df["ranked_tokens"].apply(parse_ranked_list)
    elif "ranked_nodes" in token_df:
        token_df["top_features"] = token_df["ranked_nodes"].apply(parse_ranked_list)
    else:
        token_df["top_features"] = ""
    merged = df.merge(token_df[["global_graph_index", "top_features"]], on="global_graph_index", how="left")
    merged["top_features"] = merged["top_features"].fillna("")
    return merged


def _filter_feature_columns(columns: List[str]) -> List[str]:
    return filter_allowed_features(columns)


def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, List[str], np.ndarray]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in DROP_FEATURES]
    feature_cols = _filter_feature_columns(feature_cols)
    if not feature_cols:
        raise ValueError("No numeric analytics available for logistic regression.")
    X = df[feature_cols].to_numpy(copy=True)
    y = df["is_correct"].astype(int).to_numpy(copy=True)
    if len(np.unique(y)) < 2:
        raise ValueError("Only one class present; cannot fit logistic model.")
    return X, feature_cols, y


def fit_logistic_detector(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))
    pipeline.fit(X, y)
    return pipeline


def annotate_with_predictions(df: pd.DataFrame, pipeline, feature_cols: List[str]) -> pd.DataFrame:
    X = df[feature_cols].to_numpy(copy=True)
    probs = pipeline.predict_proba(X)[:, 1]
    preds = pipeline.predict(X)
    df = df.copy()
    df["logistic_probability"] = probs
    df["logistic_prediction"] = preds
    accuracy = float((preds == df["is_correct"].to_numpy(copy=False)).mean())
    df["logistic_accuracy"] = accuracy
    return df


def sample_examples(df: pd.DataFrame, correct: bool, sample_size: int = 10, seed: int = 42) -> pd.DataFrame:
    subset = df[df["is_correct"] == int(correct)]
    if subset.empty:
        return subset
    n = min(sample_size, len(subset))
    return subset.sample(n=n, random_state=seed if correct else seed + 99)


def projection(df: pd.DataFrame) -> pd.DataFrame:
    available = [col for col in DISPLAY_COLUMNS if col in df.columns]
    return df.loc[:, available]


def build_samples(dataset: str, method: str, graph: str) -> Path:
    df = load_module_dataset(dataset, method, graph)
    df = attach_top_features(df, dataset, method, graph)

    enriched_parts: List[pd.DataFrame] = []
    for label_value in sorted(df["label"].dropna().unique()):
        subset = df[df["label"] == label_value]
        if subset.empty:
            continue
        try:
            X, feature_cols, y = prepare_features(subset)
        except ValueError:
            continue
        pipeline = fit_logistic_detector(X, y)
        annotated = annotate_with_predictions(subset, pipeline, feature_cols)
        enriched_parts.append(annotated)

    if enriched_parts:
        enriched = pd.concat(enriched_parts, ignore_index=True)
    else:
        # fallback to global fit if label-specific failed
        X, feature_cols, y = prepare_features(df)
        pipeline = fit_logistic_detector(X, y)
        enriched = annotate_with_predictions(df, pipeline, feature_cols)

    correct_samples = sample_examples(enriched, True)
    incorrect_samples = sample_examples(enriched, False)
    combined = pd.concat([correct_samples, incorrect_samples], ignore_index=True)
    combined = combined.sort_values(["is_correct", "logistic_probability"], ascending=[False, False])
    projected = projection(combined)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"mock_samples_{dataset}_{method}_{graph}.csv"
    projected.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create mock explainability samples per dataset/module.")
    parser.add_argument(
        "--datasets",
        action="append",
        choices=DATASETS,
        help="Dataset slugs to include (default: both).",
    )
    args = parser.parse_args()
    datasets = args.datasets or DATASETS

    for dataset in datasets:
        for method, graphs in MODULES.items():
            for graph in graphs:
                try:
                    path = build_samples(dataset, method, graph)
                    print(f"✓ {dataset}:{method}:{graph} -> {path}")
                except (FileNotFoundError, ValueError) as exc:
                    print(f"! Skipping {dataset}:{method}:{graph} – {exc}")


if __name__ == "__main__":  # pragma: no cover
    main()
