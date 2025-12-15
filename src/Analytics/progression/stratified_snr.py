#!/usr/bin/env python3
"""
Generate stratified SNR plots for MULTIPLE progression fields.

This is a generalized version of stratified_snr.py that works with the
multi-field aggregated data structure:

  outputs/analytics/progression/
    ├── maskout_progression_drop/
    │   └── concentration_summary.csv
    ├── maskout_progression_confidence/
    │   └── concentration_summary.csv
    ├── sufficiency_progression_drop/
    │   └── concentration_summary.csv
    └── sufficiency_progression_confidence/
        └── concentration_summary.csv

For each field, generates the same SNR analysis independently.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_PROGRESSION_FIELDS_ROOT = Path("outputs/analytics/progression")
SUMMARY_FILENAME = "concentration_summary.csv"
PLOTS_DIRNAME = "plots"

DEFAULT_METRICS: Sequence[str] = (
    "half_threshold_step_fraction",
    "normalized_area_under_curve",
)

FIELD_LABELS: Dict[str, str] = {
    "maskout_progression_drop": "Maskout Drop (Error Detection)",
    "maskout_progression_confidence": "Maskout Confidence",
    "sufficiency_progression_drop": "Sufficiency Drop (Recovery)",
    "sufficiency_progression_confidence": "Sufficiency Confidence",
}

DEFAULT_METHOD_ORDER: Sequence[str] = ("token_shap_llm", "graphsvx", "subgraphx")

DEFAULT_METHOD_LABELS: Dict[str, str] = {
    "token_shap_llm": "TokenSHAP (LLM)",
    "graphsvx": "GraphSVX (GNN)",
    "subgraphx": "SubgraphX (GNN)",
}

DEFAULT_METHOD_COLORS: Dict[str, str] = {
    "token_shap_llm": "#3498db",
    "graphsvx": "#2ecc71",
    "subgraphx": "#e74c3c",
}

DEFAULT_DATASET_ORDER: Sequence[str] = ("setfit_ag_news", "stanfordnlp_sst2")

DEFAULT_DATASET_LABELS: Dict[str, str] = {
    "setfit_ag_news": "AG News",
    "stanfordnlp_sst2": "SST-2",
}

DATASET_CLASS_LABELS: Dict[str, Dict[int, str]] = {
    "setfit_ag_news": {
        0: "World",
        1: "Sports",
        2: "Business",
        3: "Sci/Tech",
    },
    "stanfordnlp_sst2": {
        0: "Negative",
        1: "Positive",
    },
}


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def compute_snr(
    correct_mean: float,
    correct_std: float,
    incorrect_mean: float,
    incorrect_std: float,
) -> float:
    """Compute Signal-to-Noise Ratio."""
    gap = abs(incorrect_mean - correct_mean)
    noise = (correct_std + incorrect_std) / 2.0
    
    if noise <= 1e-10 or not np.isfinite(noise):
        return 0.0
    
    snr = gap / noise
    return float(snr) if np.isfinite(snr) else 0.0


def _resolve_class_metadata(dataset: str, df: pd.DataFrame) -> List[Tuple[int, str]]:
    """Return sorted list of (class_id, label) for the given dataset subset."""
    
    if dataset in DATASET_CLASS_LABELS:
        mapping = DATASET_CLASS_LABELS[dataset]
        return sorted(mapping.items(), key=lambda item: item[0])
    
    # Fallback: extract from data
    class_ids: Dict[int, str] = {}
    for group in df.get("group", []):
        if isinstance(group, str) and group.startswith("class_"):
            try:
                class_part = group.split("_", 2)[1]
                class_id = int(class_part)
                class_ids[class_id] = str(class_id)
            except (IndexError, ValueError):
                continue
    
    return sorted(class_ids.items(), key=lambda item: item[0])


def extract_snr_per_class(
    df: pd.DataFrame,
    method: str,
    metric: str,
    dataset: str,
    class_meta: List[Tuple[int, str]],
) -> Tuple[float, Dict[int, float]]:
    """Extract SNR for method across classes."""
    
    method_data = df[
        (df["method"] == method) &
        (df["metric"] == metric) &
        (df["dataset"] == dataset)
    ]
    
    if method_data.empty:
        return 0.0, {}
    
    # Overall SNR
    correct = method_data[method_data["group"] == "correct_True"]
    incorrect = method_data[method_data["group"] == "correct_False"]
    
    overall_snr = 0.0
    if not correct.empty and not incorrect.empty:
        c_mean = float(correct["mean"].mean())
        c_std = float(correct["std"].mean())
        i_mean = float(incorrect["mean"].mean())
        i_std = float(incorrect["std"].mean())
        
        overall_snr = compute_snr(c_mean, c_std, i_mean, i_std)
    
    # Per-class SNRs
    class_snrs: Dict[int, float] = {}
    for class_id, _ in class_meta:
        class_correct = method_data[
            method_data["group"] == f"class_{class_id}_correct_True"
        ]
        class_incorrect = method_data[
            method_data["group"] == f"class_{class_id}_correct_False"
        ]
        
        if not class_correct.empty and not class_incorrect.empty:
            c_mean = float(class_correct["mean"].mean())
            c_std = float(class_correct["std"].mean())
            i_mean = float(class_incorrect["mean"].mean())
            i_std = float(class_incorrect["std"].mean())
            
            class_snrs[class_id] = compute_snr(c_mean, c_std, i_mean, i_std)
    
    return overall_snr, class_snrs


def _save_figure(fig: go.Figure, output_stem: Path, width: int, height: int) -> None:
    """Save figure to PDF and HTML."""
    
    pdf_path = output_stem.with_suffix(".pdf")
    html_path = output_stem.with_suffix(".html")
    
    try:
        fig.write_image(str(pdf_path), width=width, height=height)
        print(f"  ✓ {pdf_path.name}")
    except Exception as exc:
        print(f"  ! Skipped PDF: {exc}")
    
    try:
        fig.write_html(str(html_path))
        print(f"  ✓ {html_path.name}")
    except Exception as exc:
        print(f"  ! Skipped HTML: {exc}")


# ============================================================================
# PLOT GENERATION
# ============================================================================

def _create_overall_snr_comparison(
    df: pd.DataFrame,
    dataset: str,
    class_meta: List[Tuple[int, str]],
    field_name: str,
) -> go.Figure:
    """Create bar chart comparing overall SNR across methods."""
    
    metric = DEFAULT_METRICS[0]
    
    methods = []
    snrs = []
    colors = []
    
    for method in DEFAULT_METHOD_ORDER:
        overall_snr, _ = extract_snr_per_class(df, method, metric, dataset, class_meta)
        
        methods.append(DEFAULT_METHOD_LABELS.get(method, method))
        snrs.append(overall_snr)
        colors.append(DEFAULT_METHOD_COLORS.get(method, "#999999"))
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            x=methods,
            y=snrs,
            marker=dict(
                color=colors,
                line=dict(color="black", width=2),
            ),
            text=[f"{snr:.3f}" for snr in snrs],
            textposition="outside",
            showlegend=False,
            hovertemplate="<b>%{x}</b><br>SNR: %{y:.4f}<extra></extra>",
        )
    )
    
    fig.add_hline(y=2.0, line_dash="dash", line_color="green", annotation_text="Excellent (>2.0)", annotation_position="right")
    fig.add_hline(y=1.0, line_dash="dash", line_color="orange", annotation_text="Good (>1.0)", annotation_position="right")
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Moderate (>0.5)", annotation_position="right")
    
    fig.update_layout(
        title=dict(
            text=(
                f"<b>SNR: {FIELD_LABELS.get(field_name, field_name)}</b><br>"
                f"<sub>{DEFAULT_DATASET_LABELS.get(dataset, dataset)} | Higher SNR = Better Error Discrimination</sub>"
            ),
            x=0.5,
            xanchor="center",
        ),
        yaxis_title="<b>Signal-to-Noise Ratio (SNR)</b>",
        height=600,
        width=900,
        plot_bgcolor="rgba(245,245,245,0.7)",
        paper_bgcolor="white",
        font=dict(size=12, family="Times New Roman"),
        margin=dict(l=80, r=150, t=130, b=80),
        hovermode="closest",
    )
    
    return fig


def _create_snr_heatmap(
    df: pd.DataFrame,
    dataset: str,
    class_meta: List[Tuple[int, str]],
    field_name: str,
) -> go.Figure:
    """Create heatmap of SNR values (methods × classes)."""
    
    metric = DEFAULT_METRICS[0]
    methods = DEFAULT_METHOD_ORDER
    class_ids = [cid for cid, _ in class_meta]
    class_labels = [label for _, label in class_meta]
    
    snr_matrix = []
    
    for method in methods:
        _, class_snrs = extract_snr_per_class(df, method, metric, dataset, class_meta)
        row = [class_snrs.get(cid, 0.0) for cid in class_ids]
        snr_matrix.append(row)
    
    snr_array = np.array(snr_matrix)
    
    fig = go.Figure(
        data=go.Heatmap(
            z=snr_array,
            x=[f"Class {cid}\n({label})" for cid, label in zip(class_ids, class_labels)],
            y=[DEFAULT_METHOD_LABELS.get(m, m) for m in methods],
            colorscale="RdYlGn",
            text=np.round(snr_array, 3),
            texttemplate="%{text:.3f}",
            textfont={"size": 12, "color": "black"},
            colorbar=dict(title="SNR", thickness=20),
            hovertemplate="<b>%{y}</b><br>%{x}<br>SNR: %{z:.3f}<extra></extra>",
        )
    )
    
    fig.update_layout(
        title=dict(
            text=(
                f"<b>Class-Stratified SNR: {FIELD_LABELS.get(field_name, field_name)}</b><br>"
                f"<sub>{DEFAULT_DATASET_LABELS.get(dataset, dataset)} | Red (low) → Green (high)</sub>"
            ),
            x=0.5,
            xanchor="center",
        ),
        xaxis_title="<b>Class</b>",
        yaxis_title="<b>Method</b>",
        height=600,
        width=900,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=12, family="Times New Roman"),
        margin=dict(l=150, r=80, t=130, b=100),
    )
    
    return fig


def _create_snr_consistency(
    df: pd.DataFrame,
    dataset: str,
    class_meta: List[Tuple[int, str]],
    field_name: str,
) -> go.Figure:
    """Create SNR consistency plot (std dev across classes)."""
    
    metric = DEFAULT_METRICS[0]
    
    methods = []
    snr_means = []
    snr_stds = []
    colors = []
    
    for method in DEFAULT_METHOD_ORDER:
        overall_snr, class_snrs = extract_snr_per_class(df, method, metric, dataset, class_meta)
        
        if class_snrs:
            snr_values = list(class_snrs.values())
            mean_snr = float(np.mean(snr_values))
            std_snr = float(np.std(snr_values))
        else:
            mean_snr = overall_snr
            std_snr = 0.0
        
        methods.append(DEFAULT_METHOD_LABELS.get(method, method))
        snr_means.append(mean_snr)
        snr_stds.append(std_snr)
        colors.append(DEFAULT_METHOD_COLORS.get(method, "#999999"))
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            x=methods,
            y=snr_means,
            error_y=dict(type="data", array=snr_stds, visible=True, thickness=3),
            marker=dict(
                color=colors,
                line=dict(color="black", width=2),
            ),
            text=[f"{mean:.3f}" for mean in snr_means],
            textposition="outside",
            showlegend=False,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Mean SNR: %{y:.4f}<br>"
                "Std Dev: %{error_y.array:.4f}<extra></extra>"
            ),
        )
    )
    
    fig.update_layout(
        title=dict(
            text=(
                f"<b>SNR Consistency: {FIELD_LABELS.get(field_name, field_name)}</b><br>"
                f"<sub>{DEFAULT_DATASET_LABELS.get(dataset, dataset)} | Smaller error bars = Better generalization</sub>"
            ),
            x=0.5,
            xanchor="center",
        ),
        yaxis_title="<b>Mean SNR ± Std Dev</b>",
        height=600,
        width=900,
        plot_bgcolor="rgba(245,245,245,0.7)",
        paper_bgcolor="white",
        font=dict(size=12, family="Times New Roman"),
        margin=dict(l=80, r=80, t=130, b=80),
        hovermode="closest",
    )
    
    return fig


def _create_distribution_comparison(
    df: pd.DataFrame,
    dataset: str,
    metric: str,
    class_meta: List[Tuple[int, str]],
    field_name: str,
) -> go.Figure:
    """Create box plots comparing correct vs incorrect distributions."""
    
    metric_data = df[(df["metric"] == metric) & (df["dataset"] == dataset)]
    
    if metric_data.empty:
        return go.Figure()
    
    class_ids = [cid for cid, _ in class_meta]
    class_labels = [label for _, label in class_meta]
    
    fig = make_subplots(
        rows=len(DEFAULT_METHOD_ORDER),
        cols=len(class_ids),
        subplot_titles=[f"Class {cid}\n({label})" for cid, label in zip(class_ids, class_labels)],
        specs=[
            [{"type": "bar"} for _ in class_ids]
            for _ in DEFAULT_METHOD_ORDER
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )
    
    for method_idx, method in enumerate(DEFAULT_METHOD_ORDER):
        method_data = metric_data[metric_data["method"] == method]
        
        for class_pos, class_id in enumerate(class_ids):
            class_correct = method_data[
                method_data["group"] == f"class_{class_id}_correct_True"
            ]
            class_incorrect = method_data[
                method_data["group"] == f"class_{class_id}_correct_False"
            ]
            
            # Add correct bar
            if not class_correct.empty:
                c_mean = float(class_correct["mean"].mean())
                c_std = float(class_correct["std"].mean())
                
                fig.add_trace(
                    go.Bar(
                        x=["Correct"],
                        y=[c_mean],
                        name="Correct",
                        marker=dict(color="#2ecc71"),
                        showlegend=(method_idx == 0 and class_pos == 0),
                        error_y=dict(type="data", array=[c_std], thickness=2),
                        hovertemplate=f"Correct<br>Mean: {c_mean:.4f}<extra></extra>",
                    ),
                    row=method_idx + 1,
                    col=class_pos + 1,
                )
            
            # Add incorrect bar
            if not class_incorrect.empty:
                i_mean = float(class_incorrect["mean"].mean())
                i_std = float(class_incorrect["std"].mean())
                
                fig.add_trace(
                    go.Bar(
                        x=["Incorrect"],
                        y=[i_mean],
                        name="Incorrect",
                        marker=dict(color="#e74c3c"),
                        showlegend=(method_idx == 0 and class_pos == 0),
                        error_y=dict(type="data", array=[i_std], thickness=2),
                        hovertemplate=f"Incorrect<br>Mean: {i_mean:.4f}<extra></extra>",
                    ),
                    row=method_idx + 1,
                    col=class_pos + 1,
                )
        
        # Add method label annotation
        fig.add_annotation(
            text=f"<b>{DEFAULT_METHOD_LABELS[method]}</b>",
            xref="paper",
            yref="paper",
            x=-0.08,
            y=1.0 - (method_idx + 0.5) / len(DEFAULT_METHOD_ORDER),
            showarrow=False,
            font=dict(size=13, color=DEFAULT_METHOD_COLORS[method], family="Times New Roman"),
            xanchor="right",
            yanchor="middle",
        )
    
    metric_name = metric.replace("_", " ").title()
    
    fig.update_layout(
        title=dict(
            text=(
                f"<b>Distribution Comparison: {metric_name}</b><br>"
                f"<sub>{FIELD_LABELS.get(field_name, field_name)} | {DEFAULT_DATASET_LABELS.get(dataset, dataset)}</sub><br>"
                f"<sub>Green (Correct) vs Red (Incorrect)</sub>"
            ),
            x=0.5,
            xanchor="center",
            font=dict(size=12, family="Times New Roman"),
        ),
        height=900,
        width=1200,
        plot_bgcolor="rgba(245,245,245,0.7)",
        paper_bgcolor="white",
        font=dict(size=10, family="Times New Roman"),
        margin=dict(l=150, r=50, t=150, b=80),
        showlegend=True,
        legend=dict(x=1.01, y=1.0),
        barmode="group",
    )
    
    return fig


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def generate_snr_plots_for_field(
    csv_path: Path,
    output_dir: Path,
    field_name: str,
) -> None:
    """Generate SNR plots for a single progression field."""
    
    df = pd.read_csv(csv_path)
    
    if df.empty:
        print(f"  Warning: No data in {csv_path}")
        return
    
    datasets = sorted(
        df["dataset"].unique(),
        key=lambda x: (x not in DEFAULT_DATASET_ORDER, 
                      DEFAULT_DATASET_ORDER.index(x) if x in DEFAULT_DATASET_ORDER else 999)
    )
    
    for dataset in datasets:
        ds_df = df[df["dataset"] == dataset]
        class_meta = _resolve_class_metadata(dataset, ds_df)
        
        plot_dir = output_dir / PLOTS_DIRNAME
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n  Generating plots for {dataset}...")
        
        # Overall SNR
        fig1 = _create_overall_snr_comparison(ds_df, dataset, class_meta, field_name)
        output_stem = plot_dir / f"snr_overall_{dataset}"
        _save_figure(fig1, output_stem, width=900, height=600)
        
        # Heatmap
        fig2 = _create_snr_heatmap(ds_df, dataset, class_meta, field_name)
        output_stem = plot_dir / f"snr_heatmap_{dataset}"
        _save_figure(fig2, output_stem, width=900, height=600)
        
        # Consistency
        fig3 = _create_snr_consistency(ds_df, dataset, class_meta, field_name)
        output_stem = plot_dir / f"snr_consistency_{dataset}"
        _save_figure(fig3, output_stem, width=900, height=600)
        
        # Distributions
        for metric in DEFAULT_METRICS:
            fig4 = _create_distribution_comparison(ds_df, dataset, metric, class_meta, field_name)
            metric_name = metric.replace("_", "_").lower()
            output_stem = plot_dir / f"snr_distribution_{dataset}_{metric_name}"
            _save_figure(fig4, output_stem, width=1200, height=900)


def main() -> None:
    """Main entry point: process all progression fields."""
    
    parser = argparse.ArgumentParser(
        description="Generate stratified SNR plots for all progression fields."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_PROGRESSION_FIELDS_ROOT,
        help=f"Root directory containing progression field folders (default: {DEFAULT_PROGRESSION_FIELDS_ROOT})",
    )
    parser.add_argument(
        "--fields",
        type=str,
        nargs="+",
        default=list(FIELD_LABELS.keys()),
        help="Specific fields to process (default: all)",
    )
    
    args = parser.parse_args()
    root_dir = args.root.resolve()
    
    print("=" * 140)
    print("STRATIFIED SNR ANALYSIS: Multi-Field Progression Evaluation")
    print("=" * 140)
    print(f"\nInput root: {root_dir}")
    print(f"Processing fields: {', '.join(args.fields)}\n")
    
    for field_name in args.fields:
        field_dir = root_dir / field_name
        
        if not field_dir.exists():
            print(f"⚠️  Field directory not found: {field_dir}")
            continue
        
        csv_path = field_dir / SUMMARY_FILENAME
        
        if not csv_path.exists():
            print(f"⚠️  Summary CSV not found: {csv_path}")
            continue
        
        print(f"\n{'=' * 140}")
        print(f"Processing: {FIELD_LABELS.get(field_name, field_name)}")
        print(f"{'=' * 140}")
        
        generate_snr_plots_for_field(csv_path, field_dir, field_name)
    
    print("\n" + "=" * 140)
    print("COMPLETE")
    print("=" * 140)


if __name__ == "__main__":
    main()
