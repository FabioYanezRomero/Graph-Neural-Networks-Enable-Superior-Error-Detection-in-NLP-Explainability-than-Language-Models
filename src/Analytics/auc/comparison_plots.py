#!/usr/bin/env python3
"""Generate comparison plots across explainability methods for AUC analytics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import ttest_ind

DEFAULT_SUMMARY = Path("outputs/analytics/auc/auc_summary.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/analytics/auc/plots/comparisons")

METHOD_ORDER = ("graphsvx", "subgraphx", "token_shap_llm")
METHOD_LABELS: Dict[str, str] = {
    "graphsvx": "GraphSVX (GNN)",
    "subgraphx": "SubgraphX (GNN)",
    "token_shap_llm": "TokenSHAP (LLM)",
}
METHOD_FAMILY: Dict[str, str] = {
    "graphsvx": "gnn",
    "subgraphx": "gnn",
    "token_shap_llm": "llm",
}
METHOD_COLORS: Dict[str, str] = {
    "graphsvx": "#2E86AB",
    "subgraphx": "#A23B72",
    "token_shap_llm": "#F18F01",
    "gnn": "#5E60CE",
    "llm": "#F72585",
    "correct": "#06A77D",
    "incorrect": "#D62828",
}

DATASET_LABELS: Dict[str, str] = {
    "SetFit_ag_news": "AG News (SetFit)",
    "setfit_ag_news": "AG News (SetFit)",
    "stanfordnlp_sst2": "SST-2 (StanfordNLP)",
}


def dataset_label(name: str) -> str:
    return DATASET_LABELS.get(name, name.replace("_", " ").title())


def save_figure(fig: go.Figure, stem: Path) -> None:
    """Persist a Plotly figure to HTML (PDF optional if dependencies available)."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    html_path = stem.with_suffix(".html")
    fig.write_html(str(html_path))
    base_dir = stem.parent
    try:
        pdf_path = stem.with_suffix(".pdf")
        fig.write_image(str(pdf_path))
        try:
            print(f"  ✓ {pdf_path.relative_to(base_dir)}")
        except Exception:
            print(f"  ✓ {pdf_path}")
    except Exception as exc:
        print(f"  ! PDF failed ({stem.name}): {exc}")
    try:
        print(f"  ✓ {html_path.relative_to(base_dir)}")
    except Exception:
        print(f"  ✓ {html_path}")


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"AUC summary is empty: {path}")
    required = {"method", "dataset", "graph", "group", "deletion_auc_mean"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"AUC summary missing columns: {sorted(missing)}")
    return df


def filter_correctness(df: pd.DataFrame) -> pd.DataFrame:
    subset = df[df["group"].isin(["correct_true", "correct_false"])].copy()
    if subset.empty:
        raise ValueError("No correct/incorrect groups found in summary.")
    return subset


def cohens_d(correct_vals: np.ndarray, incorrect_vals: np.ndarray) -> float:
    if correct_vals.size < 2 or incorrect_vals.size < 2:
        return float("nan")
    var1 = np.var(correct_vals, ddof=1)
    var2 = np.var(incorrect_vals, ddof=1)
    pooled = np.sqrt(((correct_vals.size - 1) * var1 + (incorrect_vals.size - 1) * var2) /
                     (correct_vals.size + incorrect_vals.size - 2))
    if pooled == 0:
        return float("nan")
    return float((correct_vals.mean() - incorrect_vals.mean()) / pooled)


def calculate_method_metrics(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for method in METHOD_ORDER:
        method_df = df[df["method"] == method]
        correct = method_df[method_df["group"] == "correct_true"]
        incorrect = method_df[method_df["group"] == "correct_false"]
        if correct.empty or incorrect.empty:
            continue

        correct_vals = correct["deletion_auc_mean"].astype(float).values
        incorrect_vals = incorrect["deletion_auc_mean"].astype(float).values

        corr = float("nan")
        try:
            corr = float(
                correct["mean_prediction_confidence"]
                .astype(float)
                .corr(correct["deletion_auc_mean"].astype(float))
            )
        except Exception:
            corr = float("nan")

        metrics[method] = {
            "label": METHOD_LABELS[method],
            "family": METHOD_FAMILY[method],
            "correct_mean": float(correct_vals.mean()),
            "correct_std": float(correct_vals.std(ddof=0)),
            "incorrect_mean": float(incorrect_vals.mean()),
            "incorrect_std": float(incorrect_vals.std(ddof=0)),
            "mean_diff": float(correct_vals.mean() - incorrect_vals.mean()),
            "cohens_d": cohens_d(correct_vals, incorrect_vals),
            "corr_conf": corr,
            "n_correct_records": int(len(correct_vals)),
            "n_incorrect_records": int(len(incorrect_vals)),
            "n_correct_samples": int(correct.get("sample_size", pd.Series(dtype=int)).sum()
                                     if "sample_size" in correct.columns else len(correct_vals)),
            "n_incorrect_samples": int(incorrect.get("sample_size", pd.Series(dtype=int)).sum()
                                       if "sample_size" in incorrect.columns else len(incorrect_vals)),
        }
    if not metrics:
        raise ValueError("Unable to compute metrics for any method.")
    return metrics


def plot_gnn_vs_llm(metrics: Dict[str, Dict[str, float]], output_dir: Path) -> None:
    available_methods = [m for m in METHOD_ORDER if m in metrics]
    labels = [metrics[m]["label"] for m in available_methods]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Correct",
            x=labels,
            y=[metrics[m]["correct_mean"] for m in available_methods],
            marker_color=METHOD_COLORS["correct"],
            opacity=0.82,
            error_y=dict(
                type="data",
                array=[metrics[m]["correct_std"] for m in available_methods],
                visible=True,
            ),
        )
    )
    fig.add_trace(
        go.Bar(
            name="Incorrect",
            x=labels,
            y=[metrics[m]["incorrect_mean"] for m in available_methods],
            marker_color=METHOD_COLORS["incorrect"],
            opacity=0.82,
            error_y=dict(
                type="data",
                array=[metrics[m]["incorrect_std"] for m in available_methods],
                visible=True,
            ),
        )
    )
    fig.update_layout(
        title="<b>Deletion AUC by Correctness</b><br><sub>Mean ± stdev per method</sub>",
        xaxis_title="Method",
        yaxis_title="Deletion AUC (mean)",
        template="plotly_white",
        barmode="group",
        height=620,
        width=980,
        legend=dict(title="Prediction"),
    )

    gnn_diffs = [metrics[m]["mean_diff"] for m in available_methods if metrics[m]["family"] == "gnn"]
    llm_diff = next((metrics[m]["mean_diff"] for m in available_methods if metrics[m]["family"] == "llm"), float("nan"))
    annotation_lines: List[str] = []
    if gnn_diffs:
        annotation_lines.append(f"GNN Δ: {np.mean(gnn_diffs):.4f}")
    if np.isfinite(llm_diff):
        annotation_lines.append(f"LLM Δ: {llm_diff:.4f}")
        if gnn_diffs and llm_diff != 0:
            annotation_lines.append(f"GNN advantage ×{np.mean(gnn_diffs)/llm_diff:.2f}")

    if annotation_lines:
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.98,
            y=0.97,
            text="<br>".join(annotation_lines),
            showarrow=False,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#555",
            borderwidth=1,
            align="right",
        )

    save_figure(fig, output_dir / "08_gnn_vs_llm_metrics")


def plot_discrimination_power(metrics: Dict[str, Dict[str, float]], output_dir: Path) -> None:
    available_methods = [m for m in METHOD_ORDER if m in metrics]
    labels = [metrics[m]["label"] for m in available_methods]
    families = [metrics[m]["family"] for m in available_methods]
    colors = [METHOD_COLORS[fam] for fam in families]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Cohen’s d (Effect Size)", "Mean Difference (Correct − Incorrect)"],
        specs=[[{"type": "bar"}, {"type": "bar"}]],
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=[metrics[m]["cohens_d"] for m in available_methods],
            marker_color=colors,
            text=[f"{metrics[m]['cohens_d']:.2f}" for m in available_methods],
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=[metrics[m]["mean_diff"] for m in available_methods],
            marker_color=colors,
            text=[f"{metrics[m]['mean_diff']:.4f}" for m in available_methods],
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_yaxes(title_text="Cohen’s d", row=1, col=1)
    fig.update_yaxes(title_text="Mean difference", row=1, col=2)
    fig.update_layout(
        title="<b>Discrimination Power</b>",
        template="plotly_white",
        height=540,
        width=1180,
    )
    save_figure(fig, output_dir / "09_discrimination_power")


def plot_effect_size_heatmap(metrics: Dict[str, Dict[str, float]], output_dir: Path) -> None:
    available_methods = [m for m in METHOD_ORDER if m in metrics]
    z = []
    for m in available_methods:
        row = [
            metrics[m]["cohens_d"],
            metrics[m]["mean_diff"],
            metrics[m]["corr_conf"],
            metrics[m]["n_correct_samples"],
        ]
        z.append(row)
    z_array = np.array(z, dtype=float)
    text = np.empty_like(z_array, dtype=object)
    for idx in np.ndindex(z_array.shape):
        value = z_array[idx]
        text[idx] = "n/a" if not np.isfinite(value) else f"{value:.3f}"
    fig = go.Figure(
        data=go.Heatmap(
            z=z_array,
            x=["Cohen’s d", "Mean diff", "Corr(conf)", "Sample size (correct)"],
            y=[metrics[m]["label"] for m in available_methods],
            colorscale="RdYlGn",
            text=text,
            texttemplate="%{text}",
            colorbar=dict(title="Value"),
        )
    )
    fig.update_layout(
        title="<b>Effect Size & Diagnostics</b>",
        template="plotly_white",
        height=520,
        width=960,
    )
    save_figure(fig, output_dir / "10_effect_size_heatmap")


def plot_mean_diff_by_graph(df: pd.DataFrame, output_dir: Path) -> None:
    methods = sorted(df["method"].unique())
    graphs = sorted(df["graph"].unique())
    z = []
    for method in methods:
        row = []
        for graph in graphs:
            correct = df[
                (df["method"] == method)
                & (df["graph"] == graph)
                & (df["group"] == "correct_true")
            ]
            incorrect = df[
                (df["method"] == method)
                & (df["graph"] == graph)
                & (df["group"] == "correct_false")
            ]
            if correct.empty or incorrect.empty:
                row.append(np.nan)
                continue
            row.append(float(correct["deletion_auc_mean"].iloc[0] - incorrect["deletion_auc_mean"].iloc[0]))
        z.append(row)
    fig = go.Figure(
        data=go.Heatmap(
            z=np.array(z, dtype=float),
            x=graphs,
            y=[METHOD_LABELS.get(method, method) for method in methods],
            colorscale="RdYlGn",
            text=np.round(np.array(z, dtype=float), 4),
            texttemplate="%{text}",
            colorbar=dict(title="Mean diff"),
        )
    )
    fig.update_layout(
        title="<b>Deletion AUC Mean Difference by Graph</b>",
        template="plotly_white",
        height=540,
        width=1020,
    )
    save_figure(fig, output_dir / "11_mean_diff_by_graph")


def plot_statistical_summary(df: pd.DataFrame, metrics: Dict[str, Dict[str, float]], output_dir: Path) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        if method not in metrics:
            continue
        correct = df[
            (df["method"] == method) & (df["group"] == "correct_true")
        ]["deletion_auc_mean"].astype(float)
        incorrect = df[
            (df["method"] == method) & (df["group"] == "correct_false")
        ]["deletion_auc_mean"].astype(float)
        if len(correct) < 2 or len(incorrect) < 2:
            continue
        t_stat, p_val = ttest_ind(correct, incorrect, equal_var=False)
        if p_val < 0.001:
            sig = "***"
        elif p_val < 0.01:
            sig = "**"
        elif p_val < 0.05:
            sig = "*"
        else:
            sig = "ns"
        rows.append(
            {
                "Method": METHOD_LABELS[method],
                "t-statistic": t_stat,
                "p-value": p_val,
                "Significance": sig,
                "Cohen’s d": metrics[method]["cohens_d"],
            }
        )
    table_df = pd.DataFrame(rows)
    if table_df.empty:
        print("  ! Statistical summary skipped (insufficient data)")
        return table_df
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["Method", "t-statistic", "p-value", "Significance", "Cohen’s d"],
                    fill_color="#2E86AB",
                    font=dict(color="white", size=12),
                    align="center",
                ),
                cells=dict(
                    values=[
                        table_df["Method"],
                        table_df["t-statistic"].round(4),
                        table_df["p-value"].apply(lambda v: f"{v:.2e}"),
                        table_df["Significance"],
                        table_df["Cohen’s d"].round(3),
                    ],
                    fill_color="lavender",
                    align="center",
                    font=dict(size=11),
                ),
            )
        ]
    )
    fig.update_layout(
        title="<b>Statistical Significance Summary</b><br><sub>Welch t-test (correct vs incorrect)</sub>",
        height=520,
        width=880,
        template="plotly_white",
    )
    save_figure(fig, output_dir / "12_statistical_summary")
    return table_df


def plot_gnn_consistency(metrics: Dict[str, Dict[str, float]], output_dir: Path) -> None:
    required = [m for m in ("graphsvx", "subgraphx") if m in metrics]
    if len(required) < 2:
        print("  ! Skipping GNN consistency plot (missing methods)")
        return
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Deletion AUC (Correct vs Incorrect)", "Effect size (Cohen’s d)"],
        specs=[[{"type": "bar"}, {"type": "bar"}]],
    )
    categories = ["Correct", "Incorrect"]
    for method in required:
        fig.add_trace(
            go.Bar(
                name=METHOD_LABELS[method].split(" ")[0],
                x=categories,
                y=[metrics[method]["correct_mean"], metrics[method]["incorrect_mean"]],
                marker_color=METHOD_COLORS[method],
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Bar(
            x=[METHOD_LABELS[m].split(" ")[0] for m in required],
            y=[metrics[m]["cohens_d"] for m in required],
            marker_color=[METHOD_COLORS[m] for m in required],
            showlegend=False,
            text=[f"{metrics[m]['cohens_d']:.2f}" for m in required],
            textposition="outside",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        title="<b>GNN Method Consistency</b>",
        template="plotly_white",
        height=540,
        width=1120,
        barmode="group",
    )
    save_figure(fig, output_dir / "13_gnn_consistency")


def plot_per_dataset(df: pd.DataFrame, output_dir: Path) -> None:
    records = []
    for dataset in sorted(df["dataset"].unique()):
        for method in METHOD_ORDER:
            correct = df[
                (df["dataset"] == dataset)
                & (df["method"] == method)
                & (df["group"] == "correct_true")
            ]["deletion_auc_mean"]
            incorrect = df[
                (df["dataset"] == dataset)
                & (df["method"] == method)
                & (df["group"] == "correct_false")
            ]["deletion_auc_mean"]
            if correct.empty or incorrect.empty:
                continue
            records.append(
                {
                    "Dataset": dataset_label(dataset),
                    "Method": METHOD_LABELS[method],
                    "Family": METHOD_FAMILY[method],
                    "Mean diff": correct.mean() - incorrect.mean(),
                }
            )
    if not records:
        print("  ! Skipping per-dataset plot (insufficient data)")
        return
    df_plot = pd.DataFrame(records)
    fig = px.bar(
        df_plot,
        x="Dataset",
        y="Mean diff",
        color="Method",
        barmode="group",
        title="<b>Mean Deletion AUC Difference by Dataset</b>",
        color_discrete_map={METHOD_LABELS[k]: METHOD_COLORS[k] for k in METHOD_ORDER if k in METHOD_COLORS},
    )
    fig.update_layout(template="plotly_white", height=540, width=960)
    save_figure(fig, output_dir / "14_per_dataset_comparison")


def plot_per_graph(df: pd.DataFrame, output_dir: Path) -> None:
    records = []
    for graph in sorted(df["graph"].unique()):
        for method in METHOD_ORDER:
            correct = df[
                (df["graph"] == graph)
                & (df["method"] == method)
                & (df["group"] == "correct_true")
            ]["deletion_auc_mean"]
            incorrect = df[
                (df["graph"] == graph)
                & (df["method"] == method)
                & (df["group"] == "correct_false")
            ]["deletion_auc_mean"]
            if correct.empty or incorrect.empty:
                continue
            records.append(
                {
                    "Graph": graph,
                    "Method": METHOD_LABELS[method],
                    "Mean diff": correct.mean() - incorrect.mean(),
                    "Family": METHOD_FAMILY[method],
                }
            )
    if not records:
        print("  ! Skipping per-graph plot (insufficient data)")
        return
    df_plot = pd.DataFrame(records)
    fig = px.bar(
        df_plot,
        x="Graph",
        y="Mean diff",
        color="Method",
        barmode="group",
        title="<b>Mean Deletion AUC Difference by Graph Type</b>",
        color_discrete_map={METHOD_LABELS[k]: METHOD_COLORS[k] for k in METHOD_ORDER if k in METHOD_COLORS},
    )
    fig.update_layout(template="plotly_white", height=540, width=1160)
    save_figure(fig, output_dir / "15_per_graph_type_comparison")


def plot_meta_summary(metrics: Dict[str, Dict[str, float]], output_dir: Path) -> None:
    available_methods = [m for m in METHOD_ORDER if m in metrics]
    labels = [metrics[m]["label"] for m in available_methods]
    families = [metrics[m]["family"] for m in available_methods]
    colors = [METHOD_COLORS[fam] for fam in families]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Cohen’s d",
            "Deletion AUC (Correct/Incorrect)",
            "Correlation with confidence",
            "Record counts",
        ],
        specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "bar"}, {"type": "bar"}]],
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=[metrics[m]["cohens_d"] for m in available_methods],
            marker_color=colors,
            showlegend=False,
            text=[f"{metrics[m]['cohens_d']:.2f}" for m in available_methods],
            textposition="outside",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            name="Correct",
            x=labels,
            y=[metrics[m]["correct_mean"] for m in available_methods],
            marker_color=METHOD_COLORS["correct"],
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            name="Incorrect",
            x=labels,
            y=[metrics[m]["incorrect_mean"] for m in available_methods],
            marker_color=METHOD_COLORS["incorrect"],
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=[metrics[m]["corr_conf"] for m in available_methods],
            marker_color=colors,
            showlegend=False,
            text=[f"{metrics[m]['corr_conf']:.3f}" if np.isfinite(metrics[m]["corr_conf"]) else "n/a"
                  for m in available_methods],
            textposition="outside",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            name="Correct",
            x=labels,
            y=[metrics[m]["n_correct_records"] for m in available_methods],
            marker_color=METHOD_COLORS["correct"],
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            name="Incorrect",
            x=labels,
            y=[metrics[m]["n_incorrect_records"] for m in available_methods],
            marker_color=METHOD_COLORS["incorrect"],
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    fig.update_layout(
        title="<b>Meta-summary dashboard</b>",
        template="plotly_white",
        height=920,
        width=1280,
        barmode="group",
    )
    save_figure(fig, output_dir / "16_meta_summary_dashboard")


def plot_discrimination_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    methods = sorted(df["method"].unique())
    datasets = sorted(df["dataset"].unique())
    graphs = sorted(df["graph"].unique())
    z = []
    row_labels = []
    for method in methods:
        for dataset in datasets:
            row = []
            for graph in graphs:
                correct = df[
                    (df["method"] == method)
                    & (df["dataset"] == dataset)
                    & (df["graph"] == graph)
                    & (df["group"] == "correct_true")
                ]["deletion_auc_mean"]
                incorrect = df[
                    (df["method"] == method)
                    & (df["dataset"] == dataset)
                    & (df["graph"] == graph)
                    & (df["group"] == "correct_false")
                ]["deletion_auc_mean"]
                if correct.empty or incorrect.empty:
                    row.append(np.nan)
                    continue
                row.append(float(correct.mean() - incorrect.mean()))
            z.append(row)
            row_labels.append(f"{METHOD_LABELS.get(method, method)} · {dataset_label(dataset)}")
    fig = go.Figure(
        data=go.Heatmap(
            z=np.array(z, dtype=float),
            x=graphs,
            y=row_labels,
            colorscale="RdYlGn",
            text=np.round(np.array(z, dtype=float), 4),
            texttemplate="%{text}",
            colorbar=dict(title="Mean diff"),
        )
    )
    fig.update_layout(
        title="<b>Discrimination ability by method, dataset & graph</b>",
        template="plotly_white",
        height=680,
        width=1180,
    )
    save_figure(fig, output_dir / "17_discrimination_heatmap")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate comparison plots for AUC analytics summary."
    )
    parser.add_argument(
        "--summary",
        default=str(DEFAULT_SUMMARY),
        help="Path to auc_summary.csv (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write comparison plots (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_summary(summary_path)
    df_correctness = filter_correctness(df)
    metrics = calculate_method_metrics(df_correctness)

    print("=" * 72)
    print("AUC COMPARISON PLOTS")
    print("=" * 72)
    print(f"Records: {len(df)}  |  Correctness rows: {len(df_correctness)}")

    plot_gnn_vs_llm(metrics, output_dir)
    plot_discrimination_power(metrics, output_dir)
    plot_effect_size_heatmap(metrics, output_dir)
    plot_mean_diff_by_graph(df_correctness, output_dir)
    stats_df = plot_statistical_summary(df_correctness, metrics, output_dir)
    plot_gnn_consistency(metrics, output_dir)
    plot_per_dataset(df_correctness, output_dir)
    plot_per_graph(df_correctness, output_dir)
    plot_meta_summary(metrics, output_dir)
    plot_discrimination_heatmap(df_correctness, output_dir)

    if not stats_df.empty:
        print("\nStatistical summary (Welch t-test):")
        print(stats_df.to_string(index=False))
    print(f"\nPlots saved to {output_dir}")


if __name__ == "__main__":
    main()
