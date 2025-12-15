#!/usr/bin/env python3
"""
Fidelity Dimension Visual Analytics
===================================

This script renders interactive Plotly dashboards for the fidelity dimension.
It consumes the per-instance CSVs produced by the fidelity extractors alongside
the aggregated summaries and generates:

  • Asymmetry distribution violins (correct vs incorrect).
  • Fidelity⁺ vs Fidelity⁻ scatter matrices with quadrant guides.
  • Strong-asymmetry prevalence bars.
  • Correctness sensitivity comparisons.
  • Paradigm breakdown stacks (deletion / insertion / neutral).

Output HTML files are written beneath ``outputs/analytics/fidelity/plots/dimension4``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

FidelityPath = Path

INSTANCE_ROOT = Path("outputs/analytics/fidelity")
SUMMARY_PATH = Path("outputs/analytics/fidelity/fidelity_summary.csv")
OUTPUT_ROOT = Path("outputs/analytics/fidelity/plots/dimension4")

ASYMMETRY_THRESHOLD = 0.3
PLOT_TEMPLATE = "plotly_white"

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

DATASET_LABELS: Dict[str, str] = {
    "SetFit_ag_news": "AG News (SetFit)",
    "stanfordnlp_sst2": "SST-2 (StanfordNLP)",
}

METHOD_ORDER: Sequence[str] = ("graphsvx", "subgraphx", "token_shap_llm")
CORRECTNESS_ORDER: Sequence[str] = ("Correct", "Incorrect")
PARADIGM_ORDER: Sequence[str] = ("Deletion-dominant", "Insertion-dominant", "Neutral")

COLOR_CORRECTNESS: Dict[str, str] = {"Correct": "#2ecc71", "Incorrect": "#e74c3c"}
COLOR_PARADIGM: Dict[str, str] = {
    "Deletion-dominant": "#d73027",
    "Insertion-dominant": "#2b8cbe",
    "Neutral": "#f0ad4e",
}


# --------------------------------------------------------------------------- #
# Data loading                                                                #
# --------------------------------------------------------------------------- #

def _dataset_slug(text: str | float | None) -> str:
    if text is None:
        return "unknown_dataset"
    try:
        raw = str(text).strip()
    except Exception:
        return "unknown_dataset"
    if not raw:
        return "unknown_dataset"
    return raw.replace("/", "_").replace("-", "_")


def discover_instance_csvs(root: Path) -> List[Path]:
    paths: List[Path] = []
    for path in root.glob("*/*/*.csv"):
        if path.name == "fidelity_summary.csv":
            continue
        # Skip nested summary directories (method/dataset/graph/fidelity_summary.csv)
        if path.parent.name == "plots":
            continue
        paths.append(path)
    return sorted(paths)


def load_instances(root: Path) -> pd.DataFrame:
    csv_paths = discover_instance_csvs(root)
    if not csv_paths:
        raise FileNotFoundError(f"No fidelity instance CSVs found under {root}")

    frames: List[pd.DataFrame] = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"! Skipping {path} ({exc})")
            continue
        if df.empty:
            continue
        df = df.copy()
        df["method"] = df["method"].astype(str)
        df["graph_type"] = df["graph_type"].astype(str)
        df["dataset_slug"] = df.get("dataset_backbone", df.get("dataset", "unknown"))
        df["dataset_slug"] = df["dataset_slug"].apply(_dataset_slug)
        df["correctness_label"] = df["is_correct"].map({True: "Correct", False: "Incorrect"})
        df["experiment"] = df["method"] + " · " + df["graph_type"]
        frames.append(df)

    if not frames:
        raise ValueError("All fidelity instance CSVs were empty.")
    combined = pd.concat(frames, ignore_index=True)
    return combined


def load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {path}")
    summary = pd.read_csv(path)
    if summary.empty:
        raise ValueError(f"Summary CSV contains no data: {path}")
    summary = summary.copy()
    summary["dataset_slug"] = summary["dataset"].apply(_dataset_slug)
    summary["method"] = summary["method"].astype(str)
    summary["graph"] = summary["graph"].astype(str)
    return summary


# --------------------------------------------------------------------------- #
# Plot builders                                                               #
# --------------------------------------------------------------------------- #

def asymmetry_distribution_plot(instances: pd.DataFrame, dataset: str, output_dir: Path) -> None:
    subset = instances[(instances["dataset_slug"] == dataset) & (instances["correctness_label"].notna())]
    if subset.empty:
        print(f"  ! Skipping asymmetry distribution for {dataset}: no correctness-labelled samples.")
        return

    subset = subset.copy()
    subset["experiment_label"] = (
        subset["method"].map(METHOD_LABELS).fillna(subset["method"])
        + "<br>"
        + subset["graph_type"].map(GRAPH_LABELS).fillna(subset["graph_type"])
    )

    fig = px.violin(
        subset,
        x="experiment_label",
        y="fidelity_asymmetry",
        color="correctness_label",
        color_discrete_map=COLOR_CORRECTNESS,
        category_orders={"experiment_label": sorted(subset["experiment_label"].unique())},
        points=False,
        box=True,
        title=(
            "<b>Fidelity Asymmetry Distribution</b><br>"
            f"<sub>{DATASET_LABELS.get(dataset, dataset)} · Correctness stratified</sub>"
        ),
    )
    fig.update_traces(meanline_visible=True, spanmode="hard")
    fig.add_hline(y=0.0, line_dash="dot", line_color="#666", line_width=1)
    fig.update_layout(
        template=PLOT_TEMPLATE,
        xaxis_title="Method · Graph",
        yaxis_title="Fidelity Asymmetry (F⁻ − F⁺)",
        legend_title="Correctness",
        width=max(900, 220 * subset["experiment_label"].nunique()),
        height=520,
    )
    path = output_dir / f"asymmetry_violin_{dataset}.html"
    fig.write_html(str(path))
    print(f"  ✓ Asymmetry distribution → {path.relative_to(output_dir)}")


def tradeoff_scatter_plot(instances: pd.DataFrame, dataset: str, output_dir: Path) -> None:
    subset = instances[instances["dataset_slug"] == dataset]
    if subset.empty:
        print(f"  ! Skipping scatter for {dataset}: no samples.")
        return

    subset = subset.copy()
    subset["method_label"] = subset["method"].map(METHOD_LABELS).fillna(subset["method"])
    subset["graph_label"] = subset["graph_type"].map(GRAPH_LABELS).fillna(subset["graph_type"])
    subset["correctness_label"] = subset["correctness_label"].fillna("Unknown")

    fig = px.scatter(
        subset,
        x="fidelity_plus",
        y="fidelity_minus",
        color="correctness_label",
        color_discrete_map={**COLOR_CORRECTNESS, "Unknown": "#7f8c8d"},
        symbol="graph_label",
        facet_col="method_label",
        facet_col_wrap=3,
        hover_data={
            "graph_index": True,
            "label": True,
            "prediction_class": True,
            "fidelity_asymmetry": ":.4f",
        },
        title=(
            "<b>Fidelity Trade-off Scatter</b><br>"
            f"<sub>{DATASET_LABELS.get(dataset, dataset)} · F⁺ vs F⁻</sub>"
        ),
    )

    fig.update_layout(
        template=PLOT_TEMPLATE,
        legend_title="Correctness",
        width=1200,
        height=500,
    )
    fig.update_xaxes(title="Fidelity⁺ (mask keep)", zeroline=True, zerolinewidth=1, zerolinecolor="#999")
    fig.update_yaxes(title="Fidelity⁻ (mask remove)", zeroline=True, zerolinewidth=1, zerolinecolor="#999")

    for annotation in fig.layout.annotations or []:
        annotation.text = annotation.text.replace("method_label=", "")

    path = output_dir / f"tradeoff_scatter_{dataset}.html"
    fig.write_html(str(path))
    print(f"  ✓ Trade-off scatter → {path.relative_to(output_dir)}")


def strong_asymmetry_bar(summary: pd.DataFrame, dataset: str, output_dir: Path) -> None:
    subset = summary[(summary["dataset_slug"] == dataset) & (summary["group"] == "overall")]
    if subset.empty:
        print(f"  ! Skipping strong-asymmetry bar for {dataset}: no overall rows.")
        return

    subset = subset.copy()
    subset["method_label"] = subset["method"].map(METHOD_LABELS).fillna(subset["method"])
    subset["graph_label"] = subset["graph"].map(GRAPH_LABELS).fillna(subset["graph"])

    records = []
    for _, row in subset.iterrows():
        records.append(
            {
                "method_label": row["method_label"],
                "graph_label": row["graph_label"],
                "direction": "Deletion-dominant",
                "percentage": row.get("strong_asymmetry_positive_pct", 0.0),
            }
        )
        records.append(
            {
                "method_label": row["method_label"],
                "graph_label": row["graph_label"],
                "direction": "Insertion-dominant",
                "percentage": row.get("strong_asymmetry_negative_pct", 0.0),
            }
        )

    plot_df = pd.DataFrame(records)
    plot_df["category"] = plot_df["method_label"] + "<br>" + plot_df["graph_label"]

    fig = px.bar(
        plot_df,
        x="category",
        y="percentage",
        color="direction",
        color_discrete_map={
            "Deletion-dominant": COLOR_PARADIGM["Deletion-dominant"],
            "Insertion-dominant": COLOR_PARADIGM["Insertion-dominant"],
        },
        barmode="group",
        title=(
            "<b>Strong Asymmetry Incidence</b><br>"
            f"<sub>{DATASET_LABELS.get(dataset, dataset)} · |F⁻ − F⁺| > {ASYMMETRY_THRESHOLD}</sub>"
        ),
    )
    fig.update_layout(
        template=PLOT_TEMPLATE,
        xaxis_title="Method · Graph",
        yaxis_title="Percentage of samples",
        legend_title="Dominance",
        width=max(900, 220 * plot_df["category"].nunique()),
        height=480,
    )
    path = output_dir / f"strong_asymmetry_{dataset}.html"
    fig.write_html(str(path))
    print(f"  ✓ Strong-asymmetry bar → {path.relative_to(output_dir)}")


def correctness_sensitivity_plot(summary: pd.DataFrame, dataset: str, output_dir: Path) -> None:
    subset = summary[(summary["dataset_slug"] == dataset) & (summary["group"].isin(["correct_true", "correct_false"]))]
    if subset.empty:
        print(f"  ! Skipping correctness sensitivity for {dataset}: missing correctness strata.")
        return

    subset = subset.copy()
    subset["method_label"] = subset["method"].map(METHOD_LABELS).fillna(subset["method"])
    subset["graph_label"] = subset["graph"].map(GRAPH_LABELS).fillna(subset["graph"])
    subset["correctness_label"] = subset["group"].map(
        {"correct_true": "Correct", "correct_false": "Incorrect"}
    )

    subset["category"] = subset["method_label"] + "<br>" + subset["graph_label"]

    fig = px.bar(
        subset,
        x="category",
        y="fidelity_asymmetry_mean",
        color="correctness_label",
        color_discrete_map=COLOR_CORRECTNESS,
        barmode="group",
        error_y="fidelity_asymmetry_std",
        title=(
            "<b>Correctness Sensitivity of Fidelity Asymmetry</b><br>"
            f"<sub>{DATASET_LABELS.get(dataset, dataset)} · Mean F⁻ − F⁺</sub>"
        ),
    )
    fig.update_layout(
        template=PLOT_TEMPLATE,
        xaxis_title="Method · Graph",
        yaxis_title="Mean asymmetry",
        legend_title="Correctness",
        width=max(900, 220 * subset["category"].nunique()),
        height=480,
    )
    fig.add_hline(y=0.0, line_dash="dot", line_color="#666", line_width=1)
    path = output_dir / f"correctness_sensitivity_{dataset}.html"
    fig.write_html(str(path))
    print(f"  ✓ Correctness sensitivity → {path.relative_to(output_dir)}")


def paradigm_breakdown_plot(instances: pd.DataFrame, dataset: str, output_dir: Path) -> None:
    subset = instances[instances["dataset_slug"] == dataset]
    if subset.empty:
        print(f"  ! Skipping paradigm breakdown for {dataset}: no samples.")
        return

    def classify(value: float) -> str:
        if np.isnan(value):
            return "Neutral"
        if value > ASYMMETRY_THRESHOLD:
            return "Deletion-dominant"
        if value < -ASYMMETRY_THRESHOLD:
            return "Insertion-dominant"
        return "Neutral"

    subset = subset.copy()
    subset["paradigm"] = subset["fidelity_asymmetry"].apply(classify)
    subset["method_label"] = subset["method"].map(METHOD_LABELS).fillna(subset["method"])
    subset["graph_label"] = subset["graph_type"].map(GRAPH_LABELS).fillna(subset["graph_type"])
    subset["category"] = subset["method_label"] + "<br>" + subset["graph_label"]

    breakdown = (
        subset.groupby(["category", "paradigm"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    totals = breakdown.groupby("category", dropna=False)["count"].transform("sum")
    breakdown["percentage"] = 100.0 * breakdown["count"] / totals.replace(0, np.nan)
    breakdown["percentage"] = breakdown["percentage"].fillna(0.0)
    breakdown["paradigm"] = pd.Categorical(breakdown["paradigm"], categories=PARADIGM_ORDER, ordered=True)

    fig = px.bar(
        breakdown,
        x="category",
        y="percentage",
        color="paradigm",
        color_discrete_map=COLOR_PARADIGM,
        title=(
            "<b>Fidelity Paradigm Breakdown</b><br>"
            f"<sub>{DATASET_LABELS.get(dataset, dataset)} · Relative share per experiment</sub>"
        ),
    )
    fig.update_layout(
        template=PLOT_TEMPLATE,
        xaxis_title="Method · Graph",
        yaxis_title="Percentage of samples",
        legend_title="Paradigm",
        barmode="stack",
        width=max(900, 220 * breakdown["category"].nunique()),
        height=480,
    )
    path = output_dir / f"paradigm_breakdown_{dataset}.html"
    fig.write_html(str(path))
    print(f"  ✓ Paradigm breakdown → {path.relative_to(output_dir)}")


# --------------------------------------------------------------------------- #
# CLI orchestration                                                           #
# --------------------------------------------------------------------------- #

def render_all(datasets: Iterable[str], instance_root: Path, summary_path: Path, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    instances = load_instances(instance_root)
    summary = load_summary(summary_path)

    available_datasets = sorted(instances["dataset_slug"].unique())
    target_datasets = [ds for ds in datasets if ds in available_datasets] if datasets else available_datasets
    if not target_datasets:
        raise ValueError("No matching datasets found for rendering.")

    for dataset in target_datasets:
        print(f"\nDataset: {DATASET_LABELS.get(dataset, dataset)} [{dataset}]")
        asymmetry_distribution_plot(instances, dataset, output_root)
        tradeoff_scatter_plot(instances, dataset, output_root)
        strong_asymmetry_bar(summary, dataset, output_root)
        correctness_sensitivity_plot(summary, dataset, output_root)
        paradigm_breakdown_plot(instances, dataset, output_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render fidelity visual analytics.")
    parser.add_argument(
        "--instance-root",
        type=Path,
        default=INSTANCE_ROOT,
        help="Root directory containing per-instance fidelity CSVs.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=SUMMARY_PATH,
        help="Path to fidelity_summary.csv.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset slugs to render (default: all available).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory where HTML plots will be written.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    render_all(args.datasets, args.instance_root, args.summary_path, args.output_root)


if __name__ == "__main__":
    main()
