#!/usr/bin/env python3
"""Aggregate AUC analytics into stratified summary tables."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

BASE_INPUT_ROOT = Path("outputs/analytics/auc")
DEFAULT_OUTPUT_ROOT = Path("outputs/analytics/auc")
SUMMARY_FILENAME = "auc_summary.csv"

REQUIRED_COLUMNS = {
    "method",
    "graph_type",
    "dataset",
    "label",
    "is_correct",
    "prediction_confidence",
    "deletion_auc",
    "insertion_auc",
    "normalised_deletion_auc",
}

METRIC_COLUMNS: List[str] = [
    "deletion_auc",
    "insertion_auc",
    "normalised_deletion_auc",
]


def normalize_label(value: object) -> Optional[str]:
    """Convert arbitrary label values into stable string identifiers."""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return str(int(float(text)))
    except Exception:
        return text


def normalize_correctness(value: object) -> Optional[bool]:
    """Coerce correctness flags to booleans."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def dataset_slug(df: pd.DataFrame) -> str:
    """Return a stable dataset identifier without path separators."""
    if "dataset_backbone" in df.columns and pd.notna(df["dataset_backbone"].iloc[0]):
        return str(df["dataset_backbone"].iloc[0]).replace("/", "_")
    if "dataset_raw" in df.columns and pd.notna(df["dataset_raw"].iloc[0]):
        return str(df["dataset_raw"].iloc[0]).replace("/", "_")
    if "dataset" in df.columns and pd.notna(df["dataset"].iloc[0]):
        return str(df["dataset"].iloc[0]).replace("/", "_")
    return "unknown_dataset"


def discover_auc_csvs(root: Path) -> List[Path]:
    """Locate all per-graph AUC CSVs below the given root."""
    candidates: List[Path] = []
    for path in root.glob("**/*.csv"):
        try:
            parts = path.relative_to(root).parts
        except ValueError:
            continue
        if len(parts) == 3:  # method/dataset/graph.csv
            candidates.append(path)
    return sorted(candidates)


def metric_stats(series: pd.Series) -> Dict[str, float]:
    """Return summary statistics for a metric series."""
    clean = series.dropna()
    if clean.empty:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "q25": np.nan,
            "q75": np.nan,
        }
    return {
        "mean": round(float(clean.mean()), 4),
        "std": round(float(clean.std(ddof=0)), 4),
        "min": round(float(clean.min()), 4),
        "max": round(float(clean.max()), 4),
        "q25": round(float(clean.quantile(0.25)), 4),
        "q75": round(float(clean.quantile(0.75)), 4),
    }


def summarise_group(
    group: pd.DataFrame,
    *,
    method: str,
    dataset: str,
    graph: str,
    backbone: Optional[str],
    group_name: str,
) -> Dict[str, object]:
    """Summarise one cohort of rows."""
    if group.empty:
        return {}

    summary: Dict[str, object] = {
        "method": method,
        "dataset": dataset,
        "graph": graph,
        "group": group_name,
        "sample_size": int(len(group)),
    }
    if backbone is not None:
        summary["backbone"] = backbone

    if "prediction_confidence" in group.columns:
        summary["mean_prediction_confidence"] = round(
            float(group["prediction_confidence"].mean()), 4
        )

    if "origin_confidence" in group.columns:
        summary["mean_origin_confidence"] = round(
            float(group["origin_confidence"].mean()), 4
        )

    for column in METRIC_COLUMNS:
        if column not in group.columns:
            continue
        stats = metric_stats(group[column])
        for key, value in stats.items():
            summary[f"{column}_{key}"] = value

    return summary


def summarise_csv(csv_path: Path, output_root: Path) -> pd.DataFrame:
    """Build the summary table for a single per-graph CSV."""
    df = pd.read_csv(csv_path)
    if df.empty:
        return pd.DataFrame()

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")

    df = df.copy()
    df["is_correct"] = df["is_correct"].apply(normalize_correctness)
    df["label_norm"] = df["label"].apply(normalize_label)

    method = str(df["method"].iloc[0])
    dataset = dataset_slug(df)
    graph = str(df["graph_type"].iloc[0])
    backbone = str(df["backbone"].iloc[0]) if "backbone" in df.columns else None

    records: List[Dict[str, object]] = []
    records.append(
        summarise_group(
            df,
            method=method,
            dataset=dataset,
            graph=graph,
            backbone=backbone,
            group_name="overall",
        )
    )

    for is_correct, subset in df.groupby("is_correct", dropna=False):
        label = "correct_unknown"
        if isinstance(is_correct, (bool, np.bool_)):
            label = "correct_true" if bool(is_correct) else "correct_false"
        records.append(
            summarise_group(
                subset,
                method=method,
                dataset=dataset,
                graph=graph,
                backbone=backbone,
                group_name=label,
            )
        )

    for label_value, subset in df.groupby("label_norm", dropna=False):
        label_str = label_value if label_value is not None else "unknown"
        records.append(
            summarise_group(
                subset,
                method=method,
                dataset=dataset,
                graph=graph,
                backbone=backbone,
                group_name=f"class_{label_str}",
            )
        )

    for (label_value, is_correct), subset in df.groupby(
        ["label_norm", "is_correct"], dropna=False
    ):
        label_str = label_value if label_value is not None else "unknown"
        corr_part = "correct_unknown"
        if isinstance(is_correct, (bool, np.bool_)):
            corr_part = "correct_true" if bool(is_correct) else "correct_false"
        records.append(
            summarise_group(
                subset,
                method=method,
                dataset=dataset,
                graph=graph,
                backbone=backbone,
                group_name=f"class_{label_str}_{corr_part}",
            )
        )

    summary_df = pd.DataFrame([r for r in records if r])
    summary_df.sort_values(
        ["method", "dataset", "graph", "group"], inplace=True, ignore_index=True
    )

    graph_dir = output_root / method / dataset / graph
    graph_dir.mkdir(parents=True, exist_ok=True)
    summary_path = graph_dir / SUMMARY_FILENAME
    summary_df.to_csv(summary_path, index=False)

    return summary_df


def aggregate_auc_metrics(input_root: Path, output_root: Path) -> pd.DataFrame:
    """Process every per-graph CSV and return the combined summary."""
    csv_files = discover_auc_csvs(input_root)
    if not csv_files:
        raise FileNotFoundError(f"No AUC CSVs found under {input_root}")

    combined: List[pd.DataFrame] = []
    for csv_path in csv_files:
        print(f"Processing {csv_path.relative_to(input_root)}")
        summary_df = summarise_csv(csv_path, output_root)
        if not summary_df.empty:
            combined.append(summary_df)

    if not combined:
        raise RuntimeError("No summaries generated; check input data.")

    all_summaries = pd.concat(combined, ignore_index=True)
    global_summary_path = output_root / SUMMARY_FILENAME
    all_summaries.to_csv(global_summary_path, index=False)
    print(f"\n✓ Global summary saved to {global_summary_path}")
    return all_summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate per-graph AUC analytics into stratified summaries."
    )
    parser.add_argument(
        "--input",
        default=str(BASE_INPUT_ROOT),
        help="Root directory containing per-method AUC CSVs (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory to store summary CSVs (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input)
    output_root = Path(args.output)

    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    aggregate_auc_metrics(input_root, output_root)


if __name__ == "__main__":
    main()
