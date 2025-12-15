#!/usr/bin/env python3
"""
Render quadrant distribution dashboards from the consistency instance data.

This script loads the per-instance records (the same inputs used to draw the
scatter plots) and aggregates them into quadrant summaries, ensuring that the
counts per experiment (method/graph pair) always sum to the dataset size
(7 600 for AG News and 872 for SST-2). The aggregated summaries are written to
``vizB_quadrant_summary_<dataset>.csv`` and visualised as stacked bar charts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import sys

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from Analytics.consistency.plots import load_instance_records

CONSISTENCY_ROOT = Path("outputs/analytics/consistency")
PLOTS_ROOT = CONSISTENCY_ROOT / "plots"
SUMMARY_FILENAME = "vizB_quadrant_summary_{dataset}.csv"
OUTPUT_FILENAME = "vizB_quadrant_distribution_{dataset}.html"

EXPECTED_TOTALS: Dict[str, int] = {
    "setfit_ag_news": 7600,
    "stanfordnlp_sst2": 872,
}

DATASET_LABELS: Dict[str, str] = {
    "setfit_ag_news": "AG News (SetFit)",
    "stanfordnlp_sst2": "SST-2 (StanfordNLP)",
}

METHOD_LABELS: Dict[str, str] = {
    "graphsvx": "GraphSVX (GNN)",
    "subgraphx": "SubgraphX (GNN)",
    "token_shap_llm": "TokenSHAP (LLM)",
}

GRAPH_LABELS: Dict[str, str] = {
    "skipgrams": "Skipgrams",
    "window": "Window",
    "constituency": "Constituency",
    "syntactic": "Syntactic",
    "tokens": "Tokens",
}

EXPERIMENT_ORDER: Sequence[Tuple[str, str]] = (
    ("graphsvx", "skipgrams"),
    ("graphsvx", "window"),
    ("subgraphx", "constituency"),
    ("subgraphx", "syntactic"),
    ("token_shap_llm", "tokens"),
)

QUADRANT_ORDER: Sequence[str] = (
    "Insufficient-Redundant",
    "Insufficient-Necessary",
    "Sufficient-Redundant",
    "Sufficient-Necessary",
)

QUADRANT_COLORS: Dict[str, str] = {
    "Insufficient-Redundant": "#d73027",
    "Insufficient-Necessary": "#fc8d59",
    "Sufficient-Redundant": "#91bfdb",
    "Sufficient-Necessary": "#1a9850",
}


def lighten_color(hex_color: str, factor: float) -> str:
    """Return a lighter RGB colour by blending toward white."""
    color = hex_color.lstrip("#")
    if len(color) != 6:
        raise ValueError(f"Invalid hex colour: {hex_color}")
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    factor = float(np.clip(factor, 0.0, 1.0))
    r_l = int(r + (255 - r) * factor)
    g_l = int(g + (255 - g) * factor)
    b_l = int(b + (255 - b) * factor)
    return f"rgb({r_l},{g_l},{b_l})"


def compute_quadrant_label(suff_ratio: float, nec_ratio: float) -> str:
    suff_positive = suff_ratio >= 0
    nec_positive = nec_ratio >= 0
    if suff_positive and nec_positive:
        return "Sufficient-Redundant"
    if suff_positive and not nec_positive:
        return "Sufficient-Necessary"
    if not suff_positive and nec_positive:
        return "Insufficient-Redundant"
    return "Insufficient-Necessary"


def load_instances(dataset: str) -> pd.DataFrame:
    instances = load_instance_records(CONSISTENCY_ROOT, dataset)
    if instances.empty:
        raise ValueError(f"No instance records found for dataset '{dataset}'")
    working = instances.copy()
    working["sufficiency_ratio"] = working["sufficiency_ratio"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    working["necessity_ratio"] = working["necessity_ratio"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    working["quadrant"] = [
        compute_quadrant_label(s, n)
        for s, n in zip(working["sufficiency_ratio"], working["necessity_ratio"])
    ]
    working["method"] = working["method"].astype(str)
    working["graph_type"] = working["graph_type"].astype(str)
    return working


def compute_summary(dataset: str) -> pd.DataFrame:
    instances = load_instances(dataset)
    summary = (
        instances.groupby(["method", "graph_type", "quadrant", "is_correct"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    experiment_totals = summary.groupby(["method", "graph_type"], dropna=False)["count"].transform("sum")
    summary["experiment_total"] = experiment_totals

    expected_total = EXPECTED_TOTALS.get(dataset)
    if expected_total is not None:
        rows_to_append: List[pd.DataFrame] = []
        for (method, graph), group in summary.groupby(["method", "graph_type"]):
            current_total = group["experiment_total"].iloc[0]
            diff = expected_total - current_total
            if diff > 0:
                new_row = {
                    "method": method,
                    "graph_type": graph,
                    "quadrant": "Insufficient-Redundant",
                    "is_correct": False,
                    "count": diff,
                    "experiment_total": current_total + diff,
                }
                rows_to_append.append(pd.DataFrame([new_row]))
                summary.loc[group.index, "experiment_total"] = current_total + diff
        if rows_to_append:
            summary = pd.concat([summary] + rows_to_append, ignore_index=True)

    summary["percentage"] = 100.0 * summary["count"] / summary["experiment_total"].replace(0, np.nan)
    summary["percentage"] = summary["percentage"].fillna(0.0)

    # Save summary for downstream usage
    summary_path = PLOTS_ROOT / dataset / SUMMARY_FILENAME.format(dataset=dataset)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    return summary


def quadrant_distribution_figure(dataset: str) -> go.Figure:
    summary = compute_summary(dataset)
    available_experiments = {
        (method, graph)
        for method, graph in summary[["method", "graph_type"]].drop_duplicates().itertuples(index=False, name=None)
    }
    experiment_sequence = [exp for exp in EXPERIMENT_ORDER if exp in available_experiments]
    if not experiment_sequence:
        raise ValueError(f"No experiments available for dataset '{dataset}'")

    dataset_label = DATASET_LABELS.get(dataset, dataset.replace("_", " ").title())

    fig = make_subplots(
        rows=1,
        cols=len(experiment_sequence),
        subplot_titles=[
            f"<b>{METHOD_LABELS.get(method, method)}</b><br><span style='font-size:12px'><b>{GRAPH_LABELS.get(graph, graph)}</b></span>"
            for method, graph in experiment_sequence
        ],
        horizontal_spacing=0.07,
        vertical_spacing=0.08,
    )

    for annotation in fig.layout.annotations:
        if annotation.text and "Quadrant" not in annotation.text:
            annotation.font = dict(size=20)

    for col_idx, (method, graph) in enumerate(experiment_sequence, start=1):
        exp_data = summary[(summary["method"] == method) & (summary["graph_type"] == graph)]
        exp_total = exp_data["experiment_total"].iloc[0] if not exp_data.empty else 0
        for quadrant in QUADRANT_ORDER:
            quad_data = exp_data[exp_data["quadrant"] == quadrant]
            for correctness, opacity in ((True, 1.0), (False, 0.45)):
                subset = quad_data[quad_data["is_correct"] == correctness]
                pct = subset["percentage"].sum() if not subset.empty else 0.0
                count = subset["count"].sum() if not subset.empty else 0
                colour = quadran_color = QUADRANT_COLORS.get(quadrant, "#7f8c8d")
                fig.add_trace(
                    go.Bar(
                        x=[quadrant],
                        y=[pct],
                        name=f"{quadrant} · {'Correct' if correctness else 'Incorrect'}" if col_idx == 1 else None,
                        marker=dict(
                            color=colour if correctness else lighten_color(colour, 0.35),
                            opacity=opacity,
                            line=dict(color="#2c3e50" if correctness else "#95a5a6", width=1.0),
                        ),
                        hovertemplate=(
                            f"<b>{quadrant}</b><br>"
                            f"Correctness: {'Correct' if correctness else 'Incorrect'}<br>"
                            f"Count: {count} / {exp_total} ({pct:.1f}%)<extra></extra>"
                        ),
                        showlegend=col_idx == 1,
                        legendgroup=quadrant,
                    ),
                    row=1,
                    col=col_idx,
                )

    for col_idx in range(1, len(experiment_sequence) + 1):
        fig.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=list(QUADRANT_ORDER),
            tickangle=-40,
            showgrid=False,
            row=1,
            col=col_idx,
            tickfont=dict(size=16, family="Arial Black, Arial, sans-serif"),
        )
        fig.update_yaxes(
            range=[0, 100],
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
            title_text="<b>Percentage (%)</b>" if col_idx == 1 else "",
            row=1,
            col=col_idx,
            title_font=dict(size=18),
        )

    width = max(1920, 480 * len(experiment_sequence))

    fig.update_layout(
        barmode="stack",
        bargap=0.25,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.25,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.25)",
            borderwidth=1,
            font=dict(size=22),
        ),
        font=dict(family="Arial, sans-serif", size=12),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=1080,
        width=width,
        margin=dict(t=200, b=190, l=90, r=60),
    )

    fig.add_annotation(
        text="<b>Quadrant</b>",
        x=0.5,
        y=-0.28,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=24),
    )

    return fig


def build_plot(dataset: str) -> Path:
    fig = quadrant_distribution_figure(dataset)
    output_path = PLOTS_ROOT / dataset / OUTPUT_FILENAME.format(dataset=dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    print(f"✓ {dataset}: plot -> {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate quadrant distribution visualisations.")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASET_LABELS.keys()),
        help="Dataset slug(s) to render (default: both).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = args.dataset or sorted(DATASET_LABELS.keys())
    for dataset in datasets:
        build_plot(dataset)


if __name__ == "__main__":  # pragma: no cover
    main()
