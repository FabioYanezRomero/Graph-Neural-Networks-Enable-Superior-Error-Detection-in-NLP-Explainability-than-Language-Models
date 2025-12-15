#!/usr/bin/env python3
"""Export dataset-level Fidelity quadrant counts for downstream tables."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd

import sys

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from Analytics.fidelity.final_plot import (  # noqa: E402
    DATASET_LABELS,
    FIDELITY_QUADRANT_ORDER,
    INSTANCE_ROOT,
    OUTPUT_ROOT,
    EXPERIMENT_ORDER,
    GRAPH_LABELS,
    METHOD_LABELS,
    annotate_fidelity_quadrants,
    load_dataset_instances,
)


SEGMENTS: Tuple[Tuple[str, object], ...] = (
    ("All", None),
    ("Correct", True),
    ("Incorrect", False),
)


def dataset_label(dataset: str) -> str:
    return DATASET_LABELS.get(dataset, dataset.replace("_", " ").title())


def build_quadrant_table(dataset: str, instance_root: Path, output_root: Path) -> Path:
    instances = load_dataset_instances(instance_root, dataset)
    if instances.empty:
        raise ValueError(f"No fidelity instances available for dataset '{dataset}'.")

    working = annotate_fidelity_quadrants(instances)
    working["method"] = working["method"].astype(str)
    working["graph_type"] = working["graph_type"].astype(str)

    available_experiments = {
        (method, graph)
        for method, graph in working[["method", "graph_type"]].drop_duplicates().itertuples(index=False, name=None)
    }
    experiment_sequence = [exp for exp in EXPERIMENT_ORDER if exp in available_experiments]
    for exp in sorted(available_experiments - set(experiment_sequence)):
        experiment_sequence.append(exp)

    rows = []
    for method, graph in experiment_sequence:
        exp_subset = working[(working["method"] == method) & (working["graph_type"] == graph)]
        if exp_subset.empty:
            continue
        method_label = METHOD_LABELS.get(method, method)
        graph_label = GRAPH_LABELS.get(graph, graph)
        for segment_name, correctness_flag in SEGMENTS:
            if correctness_flag is None:
                segment_subset = exp_subset
            else:
                segment_subset = exp_subset[exp_subset["is_correct"] == correctness_flag]
            segment_total = len(segment_subset)
            for quadrant in FIDELITY_QUADRANT_ORDER:
                count = int((segment_subset["quadrant_label"] == quadrant).sum())
                pct = 0.0 if segment_total == 0 else 100.0 * count / segment_total
                rows.append(
                    {
                        "dataset": dataset,
                        "dataset_label": dataset_label(dataset),
                        "method": method,
                        "method_label": method_label,
                        "graph_type": graph,
                        "graph_label": graph_label,
                        "segment": segment_name,
                        "segment_total": segment_total,
                        "quadrant": quadrant,
                        "count": count,
                        "percentage": pct,
                    }
                )

    summary = pd.DataFrame(rows)
    output_path = output_root / dataset / f"fidelity_quadrant_table_{dataset}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"✓ {dataset}: fidelity quadrant table -> {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Fidelity quadrant counts and percentages per dataset.")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASET_LABELS.keys()),
        help="Dataset slug(s) to process (default: all configured).",
    )
    parser.add_argument(
        "--instance-root",
        type=Path,
        default=INSTANCE_ROOT,
        help="Root directory containing per-instance Fidelity CSVs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory where summary CSVs will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets: Iterable[str] = args.dataset or sorted(DATASET_LABELS.keys())
    for dataset in datasets:
        build_quadrant_table(dataset, args.instance_root, args.output_root)


if __name__ == "__main__":  # pragma: no cover
    main()
