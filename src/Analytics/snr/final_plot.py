#!/usr/bin/env python3
"""Publication-ready SNR visualisations for Dimension 5."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

INSTANCE_ROOT = Path("outputs/analytics/snr")
OUTPUT_ROOT = Path("outputs/analytics/snr/plots")

PLOT_FILENAMES = {
    "violin": "snr_violin_{dataset}.html",
    "effect": "snr_effect_size_{dataset}.html",
    "confidence": "snr_confidence_{dataset}.html",
    "summary": "snr_summary_{dataset}.csv",
    "dataset_comparison": "snr_dataset_comparison.html",
}

DATASET_LABELS: Dict[str, str] = {
    "setfit_ag_news": "AG News (SetFit)",
    "stanfordnlp_sst2": "SST-2 (StanfordNLP)",
}

METHOD_LABELS: Dict[str, str] = {
    "graphsvx": "GraphSVX",
    "subgraphx": "SubgraphX",
    "token_shap_llm": "TokenSHAP",
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

COLOR_CORRECT = "#1a9850"
COLOR_INCORRECT = "#d73027"
METHOD_COLORS = {
    "graphsvx": "#1f77b4",
    "subgraphx": "#2ca02c",
    "token_shap_llm": "#e67e22",
}
EPS = 1e-9


def discover_instance_csvs(root: Path) -> List[Path]:
    paths: List[Path] = []
    for path in root.glob("*/*/*.csv"):
        if path.name == "snr_summary.csv" or path.name.startswith("snr_summary_"):
            continue
        paths.append(path)
    return sorted(paths)


def load_instances(root: Path) -> pd.DataFrame:
    csv_paths = discover_instance_csvs(root)
    if not csv_paths:
        raise FileNotFoundError(f"No SNR instance CSVs found under {root}")

    frames: List[pd.DataFrame] = []
    for path in csv_paths:
        df = pd.read_csv(path)
        if df.empty:
            continue
        df = df.copy()
        df["method"] = df["method"].astype(str)
        df["graph_type"] = df["graph_type"].astype(str)
        dataset_slug = df.get("dataset_backbone", df.get("dataset", "unknown"))
        df["dataset_slug"] = dataset_slug.astype(str).str.replace("/", "_")
        df["correctness_label"] = df["is_correct"].map({True: "Correct", False: "Incorrect"}).fillna("Unknown")
        df["prediction_class"] = df["prediction_class"].astype(str)
        df["experiment"] = df["method"] + " · " + df["graph_type"]
        df["snr_db"] = 20.0 * np.log10(np.clip(df["snr_linear"], EPS, None))
        df["snr_db_clip"] = df["snr_db"].clip(-5, 5)
        frames.append(df)

    if not frames:
        raise ValueError("All SNR instance CSVs were empty.")
    return pd.concat(frames, ignore_index=True)


def sorted_classes(df: pd.DataFrame) -> List[str]:
    classes = sorted({cls for cls in df["prediction_class"].unique() if cls not in {"nan", "None", "NaN"}})
    return classes or ["All"]


def experiment_label(method: str, graph: str) -> str:
    method_name = METHOD_LABELS.get(method, method)
    graph_name = GRAPH_LABELS.get(graph, graph)
    return f"{method_name}\n{graph_name}"


def plot_violin(dataset: str, instances: pd.DataFrame, output_root: Path) -> None:
    subset = instances[instances["dataset_slug"] == dataset]
    if subset.empty:
        return

    classes = sorted_classes(subset)
    dataset_label = DATASET_LABELS.get(dataset, dataset.replace("_", " ").title())

    rows = 1
    cols = len(classes)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"Class {cls}" for cls in classes],
        vertical_spacing=0.12,
        horizontal_spacing=0.06,
    )

    ordered_labels = [experiment_label(m, g) for m, g in EXPERIMENT_ORDER]

    for col_idx, cls in enumerate(classes, start=1):
        class_subset = subset[subset["prediction_class"] == cls]
        if class_subset.empty:
            continue
        for method, graph in EXPERIMENT_ORDER:
            exp_data = class_subset[
                (class_subset["method"] == method)
                & (class_subset["graph_type"] == graph)
            ]
            if exp_data.empty:
                continue
            label = experiment_label(method, graph)
            for correctness, color in (
                ("Correct", COLOR_CORRECT),
                ("Incorrect", COLOR_INCORRECT),
            ):
                cohort = exp_data[exp_data["correctness_label"] == correctness]
                if cohort.empty:
                    continue
                fig.add_trace(
                    go.Violin(
                        x=[label] * len(cohort),
                        y=cohort["snr_db_clip"],
                        name=f"{label} · {correctness}" if col_idx == 1 else None,
                        legendgroup=f"{label} · {correctness}",
                        showlegend=(col_idx == 1),
                        line=dict(color=color, width=1.0),
                        fillcolor=color,
                        opacity=0.6,
                        points=False,
                        box=dict(visible=True),
                        meanline=dict(visible=True, color="#2c3e50"),
                        offsetgroup=f"{label}-{correctness}",
                        scalemode="count",
                    ),
                    row=1,
                    col=col_idx,
                )
        fig.update_xaxes(
            categoryorder="array",
            categoryarray=ordered_labels,
            tickfont=dict(size=11),
            row=1,
            col=col_idx,
        )
        fig.update_yaxes(
            range=[-5, 5],
            title_text="SNR (dB)" if col_idx == 1 else "",
            tickfont=dict(size=11),
            row=1,
            col=col_idx,
        )

    fig.update_layout(
        title=dict(
            text=(
                "<b>Signal-to-Noise by Method · Class</b><br>"
                f"<span style='font-size:14px'>{dataset_label}</span>"
            ),
            x=0.5,
            xanchor="center",
        ),
        legend=dict(
            title=dict(text="Method · Graph · Correctness"),
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.3)",
            borderwidth=1,
        ),
        font=dict(family="Arial, sans-serif", size=12),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=680,
        width=max(2000, 380 * cols),
        margin=dict(t=140, b=110, l=140, r=80),
    )

    output_path = output_root / dataset / PLOT_FILENAMES["violin"].format(dataset=dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))


def cohens_d(correct: np.ndarray, incorrect: np.ndarray) -> float:
    if len(correct) < 2 or len(incorrect) < 2:
        return float("nan")
    mean_diff = float(correct.mean() - incorrect.mean())
    pooled = np.sqrt(
        (
            (len(correct) - 1) * correct.var(ddof=1)
            + (len(incorrect) - 1) * incorrect.var(ddof=1)
        )
        / max(len(correct) + len(incorrect) - 2, 1)
    )
    if pooled <= 0:
        pooled = max(correct.std(ddof=1), incorrect.std(ddof=1), EPS)
    return mean_diff / pooled


def plot_effect_size(dataset: str, instances: pd.DataFrame, output_root: Path) -> None:
    subset = instances[instances["dataset_slug"] == dataset]
    if subset.empty:
        return

    classes = sorted_classes(subset)
    data_rows: List[Dict[str, object]] = []
    for cls in classes:
        class_subset = subset[subset["prediction_class"] == cls]
        for method, graph in EXPERIMENT_ORDER:
            exp_data = class_subset[(class_subset["method"] == method) & (class_subset["graph_type"] == graph)]
            if exp_data.empty:
                continue
            corr = exp_data[exp_data["correctness_label"] == "Correct"]["snr_db"].to_numpy()
            incorr = exp_data[exp_data["correctness_label"] == "Incorrect"]["snr_db"].to_numpy()
            d = cohens_d(corr, incorr)
            data_rows.append(
                {
                    "class": cls,
                    "method": method,
                    "graph": graph,
                    "experiment": experiment_label(method, graph),
                    "cohens_d": d,
                }
            )
    if not data_rows:
        return

    df = pd.DataFrame(data_rows)
    df.sort_values(["class", "experiment"], inplace=True)

    bars = []
    for (method, graph) in EXPERIMENT_ORDER:
        exp_label = experiment_label(method, graph)
        exp_rows = df[df["experiment"] == exp_label]
        if exp_rows.empty:
            continue
        bars.append(
            go.Bar(
                x=[f"Class {c}" for c in exp_rows["class"]],
                y=exp_rows["cohens_d"],
                name=exp_label,
                marker_color=METHOD_COLORS.get(method, "#7f8c8d"),
            )
        )

    fig = go.Figure(data=bars)
    for threshold, label in [(-0.2, "small"), (-0.5, "medium"), (-0.8, "large"), (0.2, "small"), (0.5, "medium"), (0.8, "large")]:
        fig.add_hline(y=threshold, line_dash="dot", line_color="#aaaaaa", annotation_text=str(threshold), annotation_position="top left")

    fig.update_layout(
        title=(
            "<b>Cohen's d of SNR (Correct vs Incorrect)</b><br>"
            f"<span style='font-size:14px'>{DATASET_LABELS.get(dataset, dataset)}</span>"
        ),
        xaxis_title="Prediction Class",
        yaxis_title="Cohen's d (Correct − Incorrect)",
        yaxis_range=[-2.0, 2.0],
        legend_title="Experiment",
        font=dict(family="Arial, sans-serif", size=12),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=max(1400, 280 * len(classes)),
        height=600,
    )

    output_path = output_root / dataset / PLOT_FILENAMES["effect"].format(dataset=dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))


def plot_confidence(dataset: str, instances: pd.DataFrame, output_root: Path) -> None:
    subset = instances[(instances["dataset_slug"] == dataset) & instances["prediction_confidence"].notna()]
    if subset.empty:
        return

    classes = sorted_classes(subset)
    rows = math.ceil(len(classes) / 2)
    cols = min(2, len(classes))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"Class {cls}" for cls in classes])

    for idx, cls in enumerate(classes):
        row = idx // cols + 1
        col = idx % cols + 1
        class_subset = subset[subset["prediction_class"] == cls]
        if class_subset.empty:
            continue
        for correctness, color in (("Correct", COLOR_CORRECT), ("Incorrect", COLOR_INCORRECT)):
            group = class_subset[class_subset["correctness_label"] == correctness]
            if group.empty:
                continue
            fig.add_trace(
                go.Scattergl(
                    x=group["prediction_confidence"],
                    y=group["snr_db"],
                    mode="markers",
                    marker=dict(color=color, size=6, opacity=0.35),
                    name=f"{correctness}" if (row == 1 and col == 1) else None,
                    legendgroup=correctness,
                    showlegend=(row == 1 and col == 1),
                    hovertext=group["experiment"],
                    hovertemplate="Confidence: %{x:.2f}<br>SNR (dB): %{y:.2f}<br>%{text}<extra></extra>",
                ),
                row=row,
                col=col,
            )
        if len(class_subset) >= 2:
            x = class_subset["prediction_confidence"].to_numpy(dtype=float)
            y = class_subset["snr_db"].to_numpy(dtype=float)
            coeffs = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = np.polyval(coeffs, x_line)
            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    line=dict(color="#2c3e50", dash="dash"),
                    name="Linear fit" if (row == 1 and col == 1) else None,
                    legendgroup="fit",
                    showlegend=(row == 1 and col == 1),
                ),
                row=row,
                col=col,
            )
        fig.update_xaxes(title_text="Prediction confidence", row=row, col=col)
        fig.update_yaxes(title_text="SNR (dB)" if col == 1 else "", row=row, col=col)

    fig.update_layout(
        title=(
            "<b>SNR vs Confidence</b><br>"
            f"<span style='font-size:14px'>{DATASET_LABELS.get(dataset, dataset)}</span>"
        ),
        font=dict(family="Arial, sans-serif", size=12),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=360 * rows,
        width=800 * cols,
        margin=dict(t=120, b=80, l=80, r=40),
    )

    output_path = output_root / dataset / PLOT_FILENAMES["confidence"].format(dataset=dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))


def build_summary_table(dataset: str, instances: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    subset = instances[instances["dataset_slug"] == dataset]
    if subset.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for method, graph in EXPERIMENT_ORDER:
        exp_subset = subset[(subset["method"] == method) & (subset["graph_type"] == graph)]
        if exp_subset.empty:
            continue
        correct = exp_subset[exp_subset["correctness_label"] == "Correct"]["snr_db"].to_numpy()
        incorrect = exp_subset[exp_subset["correctness_label"] == "Incorrect"]["snr_db"].to_numpy()
        row: Dict[str, object] = {
            "Dataset": DATASET_LABELS.get(dataset, dataset),
            "Method": METHOD_LABELS.get(method, method),
            "Graph": GRAPH_LABELS.get(graph, graph),
            "Samples": len(exp_subset),
        }
        if correct.size:
            row["SNR Correct (mean ± std)"] = f"{correct.mean():.3f} ± {correct.std(ddof=0):.3f}"
        else:
            row["SNR Correct (mean ± std)"] = "NA"
        if incorrect.size:
            row["SNR Incorrect (mean ± std)"] = f"{incorrect.mean():.3f} ± {incorrect.std(ddof=0):.3f}"
        else:
            row["SNR Incorrect (mean ± std)"] = "NA"
        d = cohens_d(correct, incorrect)
        row["Cohen's d"] = f"{d:.3f}" if not math.isnan(d) else "NA"
        rows.append(row)
    df = pd.DataFrame(rows)
    output_path = output_root / dataset / PLOT_FILENAMES["summary"].format(dataset=dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def build_dataset_comparison(instances: pd.DataFrame, output_root: Path) -> None:
    if instances.empty:
        return
    df = instances.copy()
    df["dataset_label"] = df["dataset_slug"].map(DATASET_LABELS).fillna(df["dataset_slug"])
    df["experiment_label"] = df.apply(
        lambda row: f"{METHOD_LABELS.get(row['method'], row['method'])} · {GRAPH_LABELS.get(row['graph_type'], row['graph_type'])}",
        axis=1,
    )
    fig = go.Figure()
    for dataset_label, group in df.groupby("dataset_label"):
        fig.add_trace(
            go.Violin(
                x=group["experiment_label"],
                y=group["snr_db"],
                name=dataset_label,
                box=dict(visible=True),
                meanline=dict(visible=True),
                opacity=0.6,
            )
        )
    fig.update_layout(
        title="<b>Dataset Comparison of SNR (dB)</b>",
        xaxis_title="Experiment",
        yaxis_title="SNR (dB)",
        font=dict(family="Arial, sans-serif", size=12),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=max(1400, 260 * df["experiment_label"].nunique()),
        height=600,
    )
    path = output_root / PLOT_FILENAMES["dataset_comparison"]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path))


def render_all(datasets: Iterable[str], instance_root: Path, output_root: Path) -> None:
    instances = load_instances(instance_root)
    available = sorted(instances["dataset_slug"].unique())
    targets = [ds for ds in datasets if ds in available] if datasets else available
    if not targets:
        raise ValueError("No matching datasets found for SNR plotting.")

    output_root.mkdir(parents=True, exist_ok=True)

    summary_frames: List[pd.DataFrame] = []
    for dataset in targets:
        plot_violin(dataset, instances, output_root)
        plot_effect_size(dataset, instances, output_root)
        plot_confidence(dataset, instances, output_root)
        df = build_summary_table(dataset, instances, output_root)
        if not df.empty:
            summary_frames.append(df)

    if summary_frames:
        combined = pd.concat(summary_frames, ignore_index=True)
        combined_path = output_root / "snr_summary_combined.csv"
        combined.to_csv(combined_path, index=False)
    build_dataset_comparison(instances, output_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render publication-ready SNR plots.")
    parser.add_argument(
        "--instance-root",
        type=Path,
        default=INSTANCE_ROOT,
        help="Root directory containing per-instance SNR CSVs.",
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
    render_all(args.datasets, args.instance_root, args.output_root)


if __name__ == "__main__":
    main()
