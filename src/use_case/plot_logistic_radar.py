#!/usr/bin/env python3
"""Generate dual radar plots for logistic regression dimensions."""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import cycle
from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots

SRC_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from use_case.feature_config import dimension_for_feature  # type: ignore

DIMENSIONS = ["AUC", "Fidelity", "Consistency", "Progression"]
WEIGHT_SUMMARY_PATH = Path("outputs/use_case/module_datasets/coefficients/dimension_weight_summary.csv")
COEFF_ROOT = Path("outputs/use_case/module_datasets/coefficients")


def load_all_coefficients(dataset: str, coeff_root: Path, label: str | None) -> pd.DataFrame:
    path = coeff_root / dataset / f"all_logistic_coefficients_{dataset}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing coefficients file: {path}")
    df = pd.read_csv(path)
    if label is not None:
        df = df[df["label"].astype(str) == label]
    if df.empty:
        raise ValueError(f"No coefficient rows remain for dataset '{dataset}' and label '{label}'.")
    return df


def load_dimension_weights(dataset: str, label: str | None) -> pd.DataFrame:
    if not WEIGHT_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing dimension summary: {WEIGHT_SUMMARY_PATH}")
    df = pd.read_csv(WEIGHT_SUMMARY_PATH)
    df = df[df["dataset"] == dataset]
    if label is not None:
        df = df[df["label"].astype(str) == label]
    if df.empty:
        raise ValueError(f"No dimension weights for dataset '{dataset}' and label '{label}'.")
    return df


def compute_significance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["significant_95"] = df["significant_95"].astype(str).str.lower().isin(["true", "1", "yes"])
    df["dimension"] = df["feature"].map(lambda f: dimension_for_feature(str(f)))
    group = (
        df.groupby(["method", "graph", "label", "dimension"], dropna=False)
        .agg(sig_count=("significant_95", "sum"), total_features=("feature", "count"))
        .reset_index()
    )
    group["sig_rate"] = 100 * group["sig_count"] / group["total_features"].replace(0, pd.NA)
    pivot = group.pivot(index=["method", "graph", "label"], columns="dimension", values="sig_rate")
    pivot = pivot.reset_index().fillna(0.0)
    for dim in DIMENSIONS:
        if dim not in pivot.columns:
            pivot[dim] = 0.0
    pivot = pivot[["method", "graph", "label"] + DIMENSIONS]
    pivot = pivot.rename(columns={dim: f"sig_{dim}" for dim in DIMENSIONS})
    return pivot


def compute_weights(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(
        index=["method", "graph", "label"], columns="dimension", values="weight_pct", aggfunc="sum"
    ).reset_index()
    pivot = pivot.fillna(0.0)
    for dim in DIMENSIONS:
        if dim not in pivot.columns:
            pivot[dim] = 0.0
    pivot = pivot[["method", "graph", "label"] + DIMENSIONS]
    pivot = pivot.rename(columns={dim: f"weight_{dim}" for dim in DIMENSIONS})
    return pivot


def build_style_maps(keys: List[str]) -> tuple[Dict[str, str], Dict[str, str]]:
    colors = qualitative.Bold + qualitative.Safe + qualitative.Prism + qualitative.Light24
    color_cycle = cycle(colors)
    styles = ["solid", "dash", "dot", "dashdot"]
    style_cycle = cycle(styles)
    color_map: Dict[str, str] = {}
    line_map: Dict[str, str] = {}
    for key in keys:
        color_map[key] = next(color_cycle)
        line_map[key] = next(style_cycle)
    return color_map, line_map


def create_dual_radar_plot_closed(df: pd.DataFrame, dataset_name: str, filename: Path) -> None:
    color_keys = [str(row["graph"]) for _, row in df.iterrows()]
    color_map, _ = build_style_maps(sorted(set(color_keys)))

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scatterpolar"}, {"type": "scatterpolar"}]],
        column_widths=[0.48, 0.48],
        horizontal_spacing=0.04,
        subplot_titles=(
            "<b>Dimension Significance Rate (%)</b>",
            "<b>Dimension Weight (%)</b>",
        ),
    )

    for annotation in fig.layout.annotations:
        annotation.update(y=1.08, yanchor="bottom", font=dict(size=32))

    for _, row in df.iterrows():
        key = str(row["graph"])
        color = color_map.get(key, "#999999")
        linestyle = 'solid'
        theta_closed = DIMENSIONS + [DIMENSIONS[0]]
        legend_label = key

        sig_values = [row[f"sig_{dim}"] for dim in DIMENSIONS]
        sig_values_closed = sig_values + [sig_values[0]]
        fig.add_trace(
            go.Scatterpolar(
                r=sig_values_closed,
                theta=theta_closed,
                fill=None,
                name=legend_label,
                line=dict(color=color, width=3, dash=linestyle),
                marker=dict(size=8, color=color),
                legendgroup=key,
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        weight_values = [row[f"weight_{dim}"] for dim in DIMENSIONS]
        weight_values_closed = weight_values + [weight_values[0]]
        fig.add_trace(
            go.Scatterpolar(
                r=weight_values_closed,
                theta=theta_closed,
                fill=None,
                name=legend_label,
                line=dict(color=color, width=3, dash=linestyle),
                marker=dict(size=8, color=color),
                legendgroup=key,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    layout = dict(
        font=dict(size=26),
        template="plotly_white",
        polar=dict(
            radialaxis=dict(range=[0, 100], tickfont=dict(size=30), dtick=10, showline=True, showgrid=True, gridcolor='#666666', linewidth=1.4),
            angularaxis=dict(direction="clockwise"),
        ),
        polar2=dict(
            radialaxis=dict(range=[0, 100], tickfont=dict(size=24), dtick=10, showline=True, showgrid=True, gridcolor='#666666', linewidth=1.4),
            angularaxis=dict(direction="clockwise"),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.08, x=0.5, xanchor="center", font=dict(size=28)),
        margin=dict(t=200, b=160, l=120, r=120),
        height=650,
        width=1300,
    )
    config = {
        "toImageButtonOptions": {
            "format": "png",
            "filename": f"radar_{dataset_name}",
            "width": 3500,
            "height": 1800,
            "scale": 2,
        }
    }
    fig.update_layout(**layout)
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(filename, include_plotlyjs="cdn", config=config)


def build_dataset_frame(dataset: str, coeff_root: Path, label: str | None) -> pd.DataFrame:
    coeff_df = load_all_coefficients(dataset, coeff_root, label)
    weight_df = load_dimension_weights(dataset, label)

    sig_df = compute_significance(coeff_df)
    weight_df = compute_weights(weight_df)

    merged = sig_df.merge(weight_df, on=["method", "graph", "label"], how="inner")
    if merged.empty:
        raise ValueError(f"No overlapping methods/graphs for dataset '{dataset}'.")
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create radar plots for logistic regression dimensions.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["setfit_ag_news", "stanfordnlp_sst2"],
        help="Datasets to process (matching coefficient folder names).",
    )
    parser.add_argument(
        "--coeff-root",
        type=Path,
        default=COEFF_ROOT,
        help="Directory containing coefficient outputs.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="all",
        help="Label to visualise (default: 'all'). Use '' to include all labels.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory (defaults to each dataset folder).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label_value = args.label if args.label != "" else None

    for dataset in args.datasets:
        try:
            dataset_frame = build_dataset_frame(dataset, args.coeff_root, label_value)
        except Exception as exc:
            print(f"[warn] Skipping dataset '{dataset}': {exc}")
            continue

        out_dir = args.output_dir or (args.coeff_root / dataset)
        output_file = out_dir / f"radar_{dataset}.html"
        create_dual_radar_plot_closed(dataset_frame, dataset, output_file)
        csv_path = out_dir / f"radar_summary_{dataset}.csv"
        dataset_frame.to_csv(csv_path, index=False)
        print(f"[ok] Saved radar plot for {dataset} -> {output_file}")


if __name__ == "__main__":  # pragma: no cover
    main()
