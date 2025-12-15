#!/usr/bin/env python3
"""
Aggregate progression analytics for multiple fields (maskout/sufficiency drops & confidences).

This script scans the progression outputs (method/dataset/graph CSVs), computes the same set of
concentration metrics for each requested progression field, and stores the summaries under a
field-specific directory layout so downstream plots can live beside their data.

Output structure (default):

    outputs/
      analytics/
        progression/
          maskout_progression_drop/
            method/
              dataset/
                graph/
                  concentration_summary.csv
            concentration_summary.csv      # global summary for the field
            plots/
          maskout_progression_confidence/
            ...
          sufficiency_progression_drop/
            ...
          sufficiency_progression_confidence/
            ...

By default we process the four progression signals listed above, but the --fields CLI flag can be
used to restrict the run.  The legacy single-field output
`outputs/analytics/progression/concentration_summary.csv` is still produced when
`maskout_progression_drop` is requested to maintain compatibility with existing tooling.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_INPUT_ROOT = Path("outputs/analytics/progression")
DEFAULT_OUTPUT_ROOT = BASE_INPUT_ROOT

DEFAULT_FIELDS = [
    "maskout_progression_drop",
    "maskout_progression_confidence",
    "sufficiency_progression_drop",
    "sufficiency_progression_confidence",
]

SUMMARY_FILENAME = "concentration_summary.csv"
TOP_K_VALUES = (1, 3, 5, 10)
EPS = 1e-10


# ---------------------------------------------------------------------------
# Normalised progression representation
# ---------------------------------------------------------------------------

@dataclass
class NormalizedProgression:
    per_step: np.ndarray
    cumulative: np.ndarray


def normalize_progression(values_list: List[float]) -> Optional[NormalizedProgression]:
    """Normalise a progression series into a non-negative per-step distribution that sums to one."""
    if not values_list:
        return None

    values = np.asarray(values_list, dtype=float)
    if values.size == 0:
        return None

    finite_mask = np.isfinite(values)
    if not np.all(finite_mask):
        values = values[finite_mask]
    if values.size == 0:
        return None

    final_value = float(values[-1])
    deltas = np.diff(np.concatenate(([0.0], values)))
    if deltas.size == 0:
        deltas = np.array([final_value], dtype=float)
    if not np.any(np.isfinite(deltas)):
        return None

    direction = 1.0
    if abs(final_value) > EPS and np.isfinite(final_value):
        direction = np.sign(final_value) or 1.0
    else:
        fallback_total = float(np.nansum(deltas))
        if abs(fallback_total) > EPS and np.isfinite(fallback_total):
            direction = np.sign(fallback_total) or 1.0

    aligned = np.abs(deltas * direction)

    denom = abs(final_value)
    if denom <= EPS or not np.isfinite(denom):
        denom = float(np.nansum(aligned))
    denom = abs(denom)
    if denom <= EPS or not np.isfinite(denom):
        return None

    per_step = aligned / denom
    total = np.sum(per_step)
    if total <= EPS or not np.isfinite(total):
        return None
    per_step = per_step / total

    cumulative = np.cumsum(per_step)
    if cumulative.size == 0:
        return None

    final_cumulative = cumulative[-1]
    if final_cumulative <= EPS or not np.isfinite(final_cumulative):
        return None
    if abs(final_cumulative - 1.0) > 1e-6:
        per_step = per_step / final_cumulative
        cumulative = np.cumsum(per_step)

    return NormalizedProgression(per_step=per_step, cumulative=cumulative)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_sequence(raw: object) -> List[float]:
    """Convert the raw CSV cell into a list of floats."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [float(item) for item in raw]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return [float(item) for item in parsed]
        except Exception:
            try:
                return [float(text)]
            except Exception:
                return []
    try:
        return [float(raw)]
    except Exception:
        return []


def normalize_label(value: object) -> Optional[str]:
    """Return a normalised class label string or None."""
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


def add_group_value(groups: Dict[str, List[float]], group: str, value: float) -> None:
    groups.setdefault(group, []).append(value)


# ---------------------------------------------------------------------------
# Metric computations
# ---------------------------------------------------------------------------

def compute_topk_concentration(per_step: np.ndarray, k: int) -> float:
    if per_step.size == 0:
        return 0.0
    top_k = min(k, per_step.size)
    return float(np.sum(per_step[:top_k]))


def compute_half_threshold_fraction(progress: NormalizedProgression, threshold: float = 0.5) -> Optional[float]:
    cumulative = progress.cumulative
    if cumulative.size == 0:
        return None
    target = threshold * cumulative[-1]
    indices = np.where(cumulative >= target)[0]
    if indices.size == 0:
        return None
    idx = int(indices[0])
    return float((idx + 1) / cumulative.size)


def compute_normalized_auc(progress: NormalizedProgression) -> Optional[float]:
    if progress.cumulative.size == 0:
        return None
    steps = progress.cumulative.size
    x = np.linspace(1.0 / steps, 1.0, steps)
    clipped = np.clip(progress.cumulative, 0.0, 1.0)
    return float(np.trapezoid(clipped, x=x))


def compute_negative_step_mass(per_step: np.ndarray) -> float:
    negatives = per_step[per_step < 0.0]
    if negatives.size == 0:
        return 0.0
    return float(np.sum(np.abs(negatives)))


def compute_negative_step_count(per_step: np.ndarray) -> float:
    return float(np.count_nonzero(per_step < 0.0))


def summarize(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"count": 0, "mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


# ---------------------------------------------------------------------------
# File discovery & processing
# ---------------------------------------------------------------------------

def discover_progression_csvs(root: Path) -> List[Path]:
    candidates: List[Path] = []
    for path in root.glob("**/*.csv"):
        try:
            parts = path.relative_to(root).parts
        except ValueError:
            continue
        if len(parts) == 3:  # method/dataset/graph.csv
            candidates.append(path)
    return sorted(candidates)


def process_csv(csv_path: Path, progression_field: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty or progression_field not in df.columns:
        return pd.DataFrame()

    method = csv_path.parts[-3]
    dataset = csv_path.parts[-2]
    graph = csv_path.stem

    records: List[Dict[str, object]] = []
    metrics_accumulator: Dict[str, Dict[str, List[float]]] = {}

    for _, row in df.iterrows():
        progression_values = parse_sequence(row.get(progression_field))
        if not progression_values:
            continue

        normalized = normalize_progression(progression_values)
        if normalized is None:
            continue

        label = normalize_label(row.get("label"))

        correctness: Optional[bool] = None
        is_correct_raw = row.get("is_correct")
        if isinstance(is_correct_raw, (bool, np.bool_)):
            correctness = bool(is_correct_raw)
        elif isinstance(is_correct_raw, str):
            text = is_correct_raw.strip().lower()
            if text in {"true", "1", "yes", "y"}:
                correctness = True
            elif text in {"false", "0", "no", "n"}:
                correctness = False

        metric_values: Dict[str, Optional[float]] = {}
        for k in TOP_K_VALUES:
            metric_values[f"top{k}_concentration"] = compute_topk_concentration(normalized.per_step, k)

        metric_values["half_threshold_step_fraction"] = compute_half_threshold_fraction(normalized, threshold=0.5)
        metric_values["normalized_area_under_curve"] = compute_normalized_auc(normalized)
        metric_values["negative_step_mass"] = compute_negative_step_mass(normalized.per_step)
        metric_values["negative_step_count"] = compute_negative_step_count(normalized.per_step)

        for metric_name, value in metric_values.items():
            if value is None:
                continue
            bucket = metrics_accumulator.setdefault(metric_name, {"overall": []})
            add_group_value(bucket, "overall", float(value))

            if correctness is not None:
                add_group_value(bucket, f"correct_{correctness}", float(value))

            if label is not None:
                add_group_value(bucket, f"class_{label}", float(value))
                if correctness is not None:
                    add_group_value(bucket, f"class_{label}_correct_{correctness}", float(value))

    for metric_name, group_values in metrics_accumulator.items():
        for group, values in group_values.items():
            stats = summarize(values)
            if stats["count"] == 0:
                continue
            records.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "graph": graph,
                    "progression_field": progression_field,
                    "metric": metric_name,
                    "group": group,
                    **stats,
                }
            )

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main aggregation logic
# ---------------------------------------------------------------------------

def aggregate_field(csv_paths: List[Path], progression_field: str, output_root: Path) -> None:
    field_root = output_root / progression_field
    field_root.mkdir(parents=True, exist_ok=True)

    all_records: List[pd.DataFrame] = []

    for csv_path in csv_paths:
        field_df = process_csv(csv_path, progression_field)
        if field_df.empty:
            continue

        method = field_df["method"].iloc[0]
        dataset = field_df["dataset"].iloc[0]
        graph = field_df["graph"].iloc[0]

        graph_dir = field_root / method / dataset / graph
        graph_dir.mkdir(parents=True, exist_ok=True)

        graph_summary_path = graph_dir / SUMMARY_FILENAME
        field_df.sort_values(
            ["method", "dataset", "graph", "metric", "group"],
            inplace=True,
            kind="mergesort",
        )
        field_df.to_csv(graph_summary_path, index=False)

        print(f"  [field:{progression_field}] {graph_summary_path} ({len(field_df)} rows)")
        all_records.append(field_df)

    if not all_records:
        print(f"  ⚠️  No data produced for field '{progression_field}'.")
        return

    global_df = pd.concat(all_records, ignore_index=True)
    global_summary_path = field_root / SUMMARY_FILENAME
    global_df.sort_values(
        ["method", "dataset", "graph", "metric", "group"],
        inplace=True,
        kind="mergesort",
    )
    global_df.to_csv(global_summary_path, index=False)
    print(f"  [field:{progression_field}] global summary -> {global_summary_path} ({len(global_df)} rows)")

    # Maintain legacy path for maskout_progression_drop consumers
    if progression_field == "maskout_progression_drop":
        legacy_path = BASE_INPUT_ROOT / SUMMARY_FILENAME
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        global_df.to_csv(legacy_path, index=False)
        print(f"  [legacy] maskout summary -> {legacy_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate concentration metrics for multiple progression fields.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=BASE_INPUT_ROOT,
        help="Root directory containing per-graph progression CSV files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where field-specific summaries will be written.",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=DEFAULT_FIELDS,
        help="Progression fields to aggregate (default: all supported fields).",
    )
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    if not base_dir.exists():
        raise SystemExit(f"Base directory not found: {base_dir}")

    fields = list(dict.fromkeys(args.fields))  # deduplicate while preserving order

    csv_paths = discover_progression_csvs(base_dir)
    if not csv_paths:
        print("No progression CSV files found.")
        return

    print("=" * 120)
    print(f"[aggregate] base_dir={base_dir}")
    print(f"[aggregate] output_root={args.output_root.resolve()}")
    print(f"[aggregate] fields={fields}")
    print("=" * 120)

    for field in fields:
        print("\n" + "-" * 120)
        print(f"[aggregate] processing field: {field}")
        print("-" * 120)
        aggregate_field(csv_paths, field, args.output_root.resolve())

    print("\n" + "=" * 120)
    print("Aggregation complete. Field-specific summaries ready for downstream plots.")
    print("=" * 120)


if __name__ == "__main__":
    main()
