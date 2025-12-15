#!/usr/bin/env python3
"""
Visualise mean normalised progression curves per dataset/method.

Uses the exported `normalized_curves.csv` files to compare correct vs incorrect
explanations for each module (method + graph type), plotting one figure per
dataset for both maskout drop and sufficiency drop progressions.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import cycle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

GRID_POINTS = np.linspace(0.0, 1.0, 200)
NORMALIZED_FILENAME = "normalized_curves.csv"

DEFAULT_FIELDS = [
    "maskout_progression_drop",
    "sufficiency_progression_drop",
]

DEFAULT_DATASETS = [
    "setfit_ag_news",
    "stanfordnlp_sst2",
]

FIELD_LABELS = {
    "maskout_progression_drop": "Maskout Drop (Error Detection)",
    "sufficiency_progression_drop": "Sufficiency Drop (Recovery)",
}

DATASET_LABELS = {
    "setfit_ag_news": "AG News",
    "stanfordnlp_sst2": "SST-2",
}

METHOD_LABELS = {
    "token_shap_llm": "TokenSHAP",
    "graphsvx": "GraphSVX",
    "subgraphx": "SubgraphX",
}

COLOR_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def list_normalized_files(field_root: Path, dataset: str) -> List[Tuple[str, str, Path]]:
    """Return (method, graph_type, path) tuples for the given dataset."""
    files: List[Tuple[str, str, Path]] = []
    for method_dir in field_root.iterdir():
        if not method_dir.is_dir():
            continue
        method = method_dir.name
        dataset_dir = method_dir / dataset
        if not dataset_dir.exists():
            continue
        for graph_dir in dataset_dir.iterdir():
            if not graph_dir.is_dir():
                continue
            csv_path = graph_dir / NORMALIZED_FILENAME
            if csv_path.exists():
                files.append((method, graph_dir.name, csv_path))
    return sorted(files, key=lambda triple: (triple[0], triple[1]))


def interpolate_curve(sequence: Sequence[float]) -> Optional[np.ndarray]:
    arr = np.asarray(sequence, dtype=float)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    n = arr.size
    x = np.linspace(1.0 / n, 1.0, n)
    x = np.concatenate(([0.0], x))
    y = np.concatenate(([0.0], arr))
    return np.interp(GRID_POINTS, x, y)


def load_curves(csv_path: Path) -> List[Tuple[bool, np.ndarray]]:
    df = pd.read_csv(csv_path)
    if df.empty:
        return []

    curves: List[Tuple[bool, np.ndarray]] = []
    for _, row in df.iterrows():
        correctness = row.get("is_correct")
        if isinstance(correctness, float) and np.isnan(correctness):
            continue
        is_correct = bool(correctness)
        try:
            cumulative = json.loads(row["cumulative_distribution"])
        except Exception:
            continue
        grid_curve = interpolate_curve(cumulative)
        if grid_curve is None:
            continue
        curves.append((is_correct, grid_curve))
    return curves


def aggregate_curves(curves: List[Tuple[bool, np.ndarray]]) -> Dict[bool, Dict[str, np.ndarray]]:
    grouped: Dict[bool, List[np.ndarray]] = defaultdict(list)
    for correctness, curve in curves:
        grouped[correctness].append(curve)

    aggregates: Dict[bool, Dict[str, np.ndarray]] = {}
    for correctness, entries in grouped.items():
        stack = np.vstack(entries)
        count = len(entries)
        aggregates[correctness] = {
            "mean": stack.mean(axis=0),
            "std": stack.std(axis=0),
            "count": count,
        }
    return aggregates


def build_module_label(method: str, graph_type: str) -> str:
    base = METHOD_LABELS.get(method, method.replace("_", " ").title())
    graph_clean = graph_type.replace("_", " ").title()
    return f"{base} ({graph_clean})"


def ensure_plot_dir(field_root: Path) -> Path:
    plot_dir = field_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


def plot_dataset(
    field_root: Path,
    field_name: str,
    dataset: str,
) -> None:
    curve_files = list_normalized_files(field_root, dataset)
    if not curve_files:
        print(f"  ⚠️  No normalized curves for dataset '{dataset}' in field '{field_name}'.")
        return

    module_colors: Dict[str, str] = {}
    color_cycle = cycle(COLOR_PALETTE)

    fig = go.Figure()

    for method, graph_type, csv_path in curve_files:
        curves = load_curves(csv_path)
        if not curves:
            continue
        aggregates = aggregate_curves(curves)
        if not aggregates:
            continue

        module_name = build_module_label(method, graph_type)
        color = module_colors.setdefault(module_name, next(color_cycle))

        for correctness, style in ((False, "solid"), (True, "dash")):
            stats = aggregates.get(correctness)
            if stats is None:
                continue
            label_suffix = "Incorrect" if not correctness else "Correct"
            fig.add_trace(
                go.Scatter(
                    x=GRID_POINTS,
                    y=stats["mean"],
                    mode="lines",
                    name=f"{module_name} – {label_suffix}",
                    line=dict(color=color, dash=style, width=3),
                    hovertemplate="<b>%{text}</b><br>Step Fraction: %{x:.2f}<br>Cumulative Mass: %{y:.3f}<extra></extra>",
                    text=[f"{module_name} | {label_suffix} (n={stats['count']})"] * GRID_POINTS.size,
                    showlegend=True,
                )
            )

    if not fig.data:
        print(f"  ⚠️  Skipping {dataset} – no usable curves.")
        return

    fig.update_layout(
        title=(
            f"<b>Normalised Progression Curves</b><br>"
            f"<sub>{FIELD_LABELS.get(field_name, field_name)} | {DATASET_LABELS.get(dataset, dataset)}</sub>"
        ),
        xaxis_title="Fraction of Tokens Processed",
        yaxis_title="Cumulative Drop Mass",
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=1000,
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=80, r=80, t=120, b=120),
    )

    plot_dir = ensure_plot_dir(field_root)
    stem = plot_dir / f"normalized_curves_{dataset}"
    try:
        fig.write_image(str(stem.with_suffix(".pdf")), width=1000, height=600)
        print(f"    ✓ Saved {stem.with_suffix('.pdf')}")
    except Exception as exc:
        print(f"    ⚠️  PDF export failed for {stem}: {exc}")
    fig.write_html(str(stem.with_suffix(".html")))
    print(f"    ✓ Saved {stem.with_suffix('.html')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot mean normalized progression curves per dataset.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/analytics/progression"),
        help="Root directory containing progression field folders.",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=DEFAULT_FIELDS,
        help="Progression fields to plot (default: maskout & sufficiency drops).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets to plot (default: AG News & SST-2).",
    )
    args = parser.parse_args()

    root_dir = args.root.resolve()
    if not root_dir.exists():
        raise SystemExit(f"Root directory not found: {root_dir}")

    fields = list(dict.fromkeys(args.fields))
    datasets = list(dict.fromkeys(args.datasets))

    print("=" * 120)
    print(f"[plot-normalized] root={root_dir}")
    print(f"[plot-normalized] fields={fields}")
    print(f"[plot-normalized] datasets={datasets}")
    print("=" * 120)

    for field in fields:
        field_root = root_dir / field
        if not field_root.exists():
            print(f"⚠️  Field directory missing: {field_root}")
            continue

        print("\n" + "-" * 120)
        print(f"[plot-normalized] Processing field: {field}")
        print("-" * 120)

        for dataset in datasets:
            print(f"  Dataset: {dataset}")
            plot_dataset(field_root, field, dataset)

    print("\n" + "=" * 120)
    print("Normalised curve plotting complete.")
    print("=" * 120)


if __name__ == "__main__":
    main()
