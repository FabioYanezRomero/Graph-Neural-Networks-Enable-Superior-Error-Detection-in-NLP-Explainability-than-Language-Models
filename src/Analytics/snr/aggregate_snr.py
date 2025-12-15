#!/usr/bin/env python3
"""
Aggregate signal-to-noise analytics across methods/datasets/graph types.

Input layout (from the SNR extractors):

    outputs/analytics/snr/<method>/<dataset>/<graph>.csv

Each row stores per-instance statistics (signal/noise means & stds, SNR, etc.).
This script summarises those CSVs using a hierarchy similar to other analytics:

  • overall
  • correctness (correct / incorrect)
  • per-class
  • per-class × correctness

It also derives discriminability metrics following the paper's definition:
|μ_incorrect - μ_correct| / ((σ_correct + σ_incorrect) / 2).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import math
import numpy as np
import pandas as pd

BASE_INPUT_ROOT = Path("outputs/analytics/snr")
DEFAULT_OUTPUT_ROOT = Path("outputs/analytics/snr")
SUMMARY_FILENAME = "snr_summary.csv"
EPS = 1e-9
HIGH_SNR_THRESHOLD = 1.0

REQUIRED_COLUMNS = {
    "method",
    "graph_type",
    "label",
    "is_correct",
    "prediction_class",
    "signal_mean",
    "signal_std",
    "noise_mean",
    "noise_std",
    "snr_linear",
    "snr_db",
    "signal_count",
    "noise_count",
    "top_k_source",
    "aggregated_from_coalitions",
}


def normalize_label(value: object) -> Optional[str]:
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


def discover_snr_csvs(root: Path) -> List[Path]:
    candidates: List[Path] = []
    for path in root.glob("**/*.csv"):
        try:
            parts = path.relative_to(root).parts
        except ValueError:
            continue
        if len(parts) == 3:  # method/dataset/graph.csv
            candidates.append(path)
    return sorted(candidates)


def dataset_slug(df: pd.DataFrame) -> str:
    if "dataset_backbone" in df.columns and pd.notna(df["dataset_backbone"].iloc[0]):
        return str(df["dataset_backbone"].iloc[0]).replace("/", "_")
    if "dataset" in df.columns and pd.notna(df["dataset"].iloc[0]):
        return str(df["dataset"].iloc[0]).replace("/", "_")
    return "unknown_dataset"


def discriminability_metric(group: pd.DataFrame, *, column: str) -> float:
    correct = group[group["is_correct"] == True]  # noqa: E712
    incorrect = group[group["is_correct"] == False]  # noqa: E712
    if correct.empty or incorrect.empty:
        return float("nan")
    mu_correct = float(correct[column].mean())
    mu_incorrect = float(incorrect[column].mean())
    sigma_correct = float(correct[column].std(ddof=0))
    sigma_incorrect = float(incorrect[column].std(ddof=0))
    denom = (sigma_correct + sigma_incorrect) / 2.0
    if denom <= 0.0:
        denom = max(sigma_correct, sigma_incorrect, EPS)
    return abs(mu_incorrect - mu_correct) / denom


def summarise_group(group: pd.DataFrame, *, method: str, dataset: str, graph: str, group_name: str) -> Dict[str, object]:
    if group.empty:
        return {}

    summary: Dict[str, object] = {
        "method": method,
        "dataset": dataset,
        "graph": graph,
        "group": group_name,
        "sample_size": int(len(group)),
        "signal_mean": round(float(group["signal_mean"].mean()), 6),
        "signal_std": round(float(group["signal_mean"].std(ddof=0)), 6),
        "signal_min": round(float(group["signal_mean"].min()), 6),
        "signal_max": round(float(group["signal_mean"].max()), 6),
        "noise_mean": round(float(group["noise_mean"].mean()), 6),
        "noise_std": round(float(group["noise_mean"].std(ddof=0)), 6),
        "noise_min": round(float(group["noise_mean"].min()), 6),
        "noise_max": round(float(group["noise_mean"].max()), 6),
        "snr_linear_mean": round(float(group["snr_linear"].mean()), 6),
        "snr_linear_std": round(float(group["snr_linear"].std(ddof=0)), 6),
        "snr_linear_min": round(float(group["snr_linear"].min()), 6),
        "snr_linear_max": round(float(group["snr_linear"].max()), 6),
        "snr_db_mean": round(float(group["snr_db"].replace(-np.inf, np.nan).mean()), 6),
        "snr_db_std": round(float(group["snr_db"].replace(-np.inf, np.nan).std(ddof=0)), 6),
        "high_snr_count": int((group["snr_linear"] >= HIGH_SNR_THRESHOLD).sum()),
        "high_snr_pct": round(float((group["snr_linear"] >= HIGH_SNR_THRESHOLD).mean() * 100.0), 2),
    }

    if "signal_count" in group.columns:
        summary["signal_k_mean"] = round(float(group["signal_count"].mean()), 4)
    if "noise_count" in group.columns:
        summary["noise_k_mean"] = round(float(group["noise_count"].mean()), 4)
    if "aggregated_from_coalitions" in group.columns:
        summary["aggregated_fraction"] = round(
            float(group["aggregated_from_coalitions"].astype(float).mean()), 4
        )
    if "top_k_source" in group.columns:
        mode_series = group["top_k_source"].dropna()
        summary["top_k_source_mode"] = (
            mode_series.mode().iloc[0] if not mode_series.empty else None
        )

    try:
        summary["snr_discriminability_signal"] = round(discriminability_metric(group, column="signal_mean"), 6)
    except Exception:
        summary["snr_discriminability_signal"] = float("nan")

    try:
        summary["snr_discriminability_snr"] = round(discriminability_metric(group, column="snr_linear"), 6)
    except Exception:
        summary["snr_discriminability_snr"] = float("nan")

    if group_name == "overall":
        correct = group[group["is_correct"] == True]  # noqa: E712
        incorrect = group[group["is_correct"] == False]  # noqa: E712
        if not correct.empty:
            summary["snr_correct_mean"] = round(float(correct["snr_linear"].mean()), 6)
            summary["snr_correct_std"] = round(float(correct["snr_linear"].std(ddof=0)), 6)
            summary["snr_correct_cv"] = round(
                float(
                    correct["snr_linear"].std(ddof=0)
                    / max(abs(correct["snr_linear"].mean()), EPS)
                ),
                6,
            )
        else:
            summary["snr_correct_mean"] = float("nan")
            summary["snr_correct_std"] = float("nan")
            summary["snr_correct_cv"] = float("nan")
        if not incorrect.empty:
            summary["snr_incorrect_mean"] = round(float(incorrect["snr_linear"].mean()), 6)
            summary["snr_incorrect_std"] = round(float(incorrect["snr_linear"].std(ddof=0)), 6)
            summary["snr_incorrect_cv"] = round(
                float(
                    incorrect["snr_linear"].std(ddof=0)
                    / max(abs(incorrect["snr_linear"].mean()), EPS)
                ),
                6,
            )
        else:
            summary["snr_incorrect_mean"] = float("nan")
            summary["snr_incorrect_std"] = float("nan")
            summary["snr_incorrect_cv"] = float("nan")

        if not correct.empty and not incorrect.empty:
            gap = float(correct["snr_linear"].mean() - incorrect["snr_linear"].mean())
            std_correct = float(correct["snr_linear"].std(ddof=0))
            std_incorrect = float(incorrect["snr_linear"].std(ddof=0))
            n1 = len(correct)
            n2 = len(incorrect)
            pooled = np.sqrt(
                (
                    ((n1 - 1) * (std_correct ** 2) + (n2 - 1) * (std_incorrect ** 2))
                    / max(n1 + n2 - 2, 1)
                )
            )
            cohens_d = gap / max(pooled, EPS)
            summary["snr_gap"] = round(gap, 6)
            summary["snr_cohens_d"] = round(cohens_d, 6)
            # Non-overlap percentage (assuming normal): erf(|d| / sqrt(2))
            summary["snr_non_overlap_pct"] = round(
                float(math.erf(abs(cohens_d) / math.sqrt(2.0))) * 100.0,
                4,
            )
            # AUC approximation using d
            summary["snr_auc"] = round(float(0.5 * (1 + math.erf(cohens_d / math.sqrt(2.0)))), 6)
        else:
            summary["snr_gap"] = float("nan")
            summary["snr_cohens_d"] = float("nan")
            summary["snr_non_overlap_pct"] = float("nan")
            summary["snr_auc"] = float("nan")

    return summary


def process_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        return pd.DataFrame()

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    method = str(df["method"].iloc[0])
    dataset = dataset_slug(df)
    graph = str(df["graph_type"].iloc[0])

    records: List[Dict[str, object]] = []

    records.append(
        summarise_group(df, method=method, dataset=dataset, graph=graph, group_name="overall")
    )

    for is_correct, subset in df.groupby("is_correct", dropna=False):
        label = f"correct_{str(is_correct).lower()}"
        records.append(
            summarise_group(subset, method=method, dataset=dataset, graph=graph, group_name=label)
        )

    for label_value, subset in df.groupby("prediction_class", dropna=False):
        label_str = normalize_label(label_value)
        group_label = f"class_{label_str}" if label_str is not None else "class_unknown"
        records.append(
            summarise_group(subset, method=method, dataset=dataset, graph=graph, group_name=group_label)
        )

    for (label_value, is_correct), subset in df.groupby(["prediction_class", "is_correct"], dropna=False):
        label_str = normalize_label(label_value)
        class_part = f"class_{label_str}" if label_str is not None else "class_unknown"
        corr_part = f"correct_{str(is_correct).lower()}"
        records.append(
            summarise_group(
                subset,
                method=method,
                dataset=dataset,
                graph=graph,
                group_name=f"{class_part}_{corr_part}",
            )
        )

    cleaned = [rec for rec in records if rec]
    if not cleaned:
        return pd.DataFrame()
    return pd.DataFrame(cleaned)


def aggregate_snr(csv_paths: List[Path], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    all_records: List[pd.DataFrame] = []

    for csv_path in csv_paths:
        summary_df = process_csv(csv_path)
        if summary_df.empty:
            continue

        method = summary_df["method"].iloc[0]
        dataset = summary_df["dataset"].iloc[0]
        graph = summary_df["graph"].iloc[0]

        graph_dir = output_root / method / dataset / graph
        graph_dir.mkdir(parents=True, exist_ok=True)

        graph_df = summary_df.sort_values(
            ["method", "dataset", "graph", "group"],
            kind="mergesort",
        )
        graph_path = graph_dir / SUMMARY_FILENAME
        graph_df.to_csv(graph_path, index=False)
        print(f"  [snr] {graph_path} ({len(graph_df)} rows)")
        all_records.append(graph_df)

    if not all_records:
        print("No SNR summaries produced.")
        return

    global_df = pd.concat(all_records, ignore_index=True)
    global_df.sort_values(
        ["method", "dataset", "graph", "group"],
        inplace=True,
        kind="mergesort",
    )
    global_path = output_root / SUMMARY_FILENAME
    global_df.to_csv(global_path, index=False)
    print(f"  [snr] global summary -> {global_path} ({len(global_df)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate SNR metrics across methods/datasets/graphs.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=BASE_INPUT_ROOT,
        help="Root directory containing SNR CSV files (method/dataset/graph.csv).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where aggregated summaries will be written.",
    )
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    if not base_dir.exists():
        raise SystemExit(f"Base directory not found: {base_dir}")

    csv_paths = discover_snr_csvs(base_dir)
    if not csv_paths:
        print("No SNR CSV files found.")
        return

    print("=" * 120)
    print(f"[snr] base_dir={base_dir}")
    print(f"[snr] output_root={args.output_root.resolve()}")
    print("=" * 120)
    aggregate_snr(csv_paths, args.output_root.resolve())
    print("\nAggregation complete.")


if __name__ == "__main__":
    main()
