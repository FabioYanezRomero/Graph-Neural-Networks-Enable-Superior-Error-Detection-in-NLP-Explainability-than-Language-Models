#!/usr/bin/env python3
"""
Export normalised progression curves for every explanation.

For each raw progression CSV (method/dataset/graph) we parse the selected progression
field, normalise its per-step contributions, and store both the per-step distribution
and cumulative curve alongside metadata such as dataset, class label, and correctness.
This enables downstream visualisations that look at the entire sequence rather than
just the scalar concentration metrics.
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_BASE_DIR = Path("outputs/analytics/progression")
DEFAULT_FIELDS = [
    "maskout_progression_drop",
    "maskout_progression_confidence",
    "sufficiency_progression_drop",
    "sufficiency_progression_confidence",
]
OUTPUT_FILENAME = "normalized_curves.csv"
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


def discover_progression_csvs(root: Path) -> List[Path]:
    """Locate the per-graph progression CSV files (method/dataset/graph.csv)."""
    candidates: List[Path] = []
    for path in root.glob("**/*.csv"):
        try:
            parts = path.relative_to(root).parts
        except ValueError:
            continue
        if len(parts) == 3:  # method/dataset/graph.csv
            candidates.append(path)
    return sorted(candidates)


def coerce_bool(value: object) -> Optional[bool]:
    """Normalise correctness values to bools when possible."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        if value in (0, 1):
            return bool(int(value))
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n"}:
            return False
    return None


def format_sequence(values: Iterable[float]) -> str:
    return json.dumps([float(v) for v in values])


def slugify(text: Optional[str]) -> str:
    if not text:
        return "unknown"
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(text))
    return safe.strip("_").lower() or "unknown"


# ---------------------------------------------------------------------------
# Core export logic
# ---------------------------------------------------------------------------

def process_csv(csv_path: Path, progression_field: str) -> List[Dict[str, object]]:
    df = pd.read_csv(csv_path)
    if df.empty or progression_field not in df.columns:
        return []

    method = csv_path.parts[-3]
    dataset = csv_path.parts[-2]
    graph = csv_path.stem

    records: List[Dict[str, object]] = []

    for _, row in df.iterrows():
        progression_values = parse_sequence(row.get(progression_field))
        if not progression_values:
            continue

        normalized = normalize_progression(progression_values)
        if normalized is None:
            continue

        label = normalize_label(row.get("label"))
        prediction_class = normalize_label(row.get("prediction_class"))
        correctness = coerce_bool(row.get("is_correct"))

        graph_type = row.get("graph_type") or csv_path.stem

        records.append(
            {
                "method": method,
                "dataset": dataset,
                "graph": graph,
                "progression_field": progression_field,
                "backbone": row.get("backbone"),
                "dataset_raw": row.get("dataset"),
                "dataset_backbone": row.get("dataset_backbone"),
                "graph_type": graph_type,
                "run_id": row.get("run_id"),
                "graph_index": row.get("graph_index"),
                "global_graph_index": row.get("global_graph_index"),
                "label": label,
                "prediction_class": prediction_class,
                "is_correct": correctness,
                "sequence_length": int(normalized.per_step.size),
                "per_step_distribution": format_sequence(normalized.per_step),
                "cumulative_distribution": format_sequence(normalized.cumulative),
            }
        )

    return records


def export_field(csv_paths: List[Path], field: str, output_root: Path) -> None:
    field_root = output_root / field
    field_root.mkdir(parents=True, exist_ok=True)

    grouped_records: Dict[Tuple[str, str, str], List[Dict[str, object]]] = {}

    for csv_path in csv_paths:
        field_records = process_csv(csv_path, field)
        if not field_records:
            continue
        for record in field_records:
            graph_type = slugify(record.get("graph_type"))
            key = (record["method"], record["dataset"], graph_type)
            grouped_records.setdefault(key, []).append(record)

    if not grouped_records:
        print(f"  ⚠️  No normalised curves generated for field '{field}'.")
        return

    total_records = 0

    for (method, dataset, graph_type), records in sorted(grouped_records.items()):
        target_dir = field_root / method / dataset / graph_type
        target_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(records)
        df.sort_values(
            ["graph", "graph_index"],
            inplace=True,
            kind="mergesort",
        )

        output_path = target_dir / OUTPUT_FILENAME
        df.to_csv(output_path, index=False)
        total_records += len(df)
        print(f"    • {method}/{dataset}/{graph_type}: {len(df)} rows -> {output_path}")

    print(
        f"  ✓ Stored {total_records} normalised curves across {len(grouped_records)} method/dataset/graph-type combos."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Export normalised progression curves per explanation.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Root directory containing per-graph progression CSV files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Directory where field-specific outputs will be written.",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=DEFAULT_FIELDS,
        help="Progression fields to process (default: all supported fields).",
    )
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    if not base_dir.exists():
        raise SystemExit(f"Base directory not found: {base_dir}")

    csv_paths = discover_progression_csvs(base_dir)
    if not csv_paths:
        print("No progression CSV files found.")
        return

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    fields = list(dict.fromkeys(args.fields))

    print("=" * 120)
    print(f"[normalized] base_dir={base_dir}")
    print(f"[normalized] output_root={output_root}")
    print(f"[normalized] fields={fields}")
    print("=" * 120)

    for field in fields:
        print("\n" + "-" * 120)
        print(f"[normalized] processing field: {field}")
        print("-" * 120)
        export_field(csv_paths, field, output_root)

    print("\n" + "=" * 120)
    print("Normalised progression export complete.")
    print("=" * 120)


if __name__ == "__main__":
    main()
