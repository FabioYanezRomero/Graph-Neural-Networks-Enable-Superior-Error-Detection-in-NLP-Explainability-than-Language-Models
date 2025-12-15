#!/usr/bin/env python3
"""
Decision margin consistency visualisations.

This script consumes the aggregated consistency summaries produced by
``aggregate_consistency.py`` (now housed under the consistency analytics
namespace) and generates four dashboards that surface how well explanation
methods preserve or disrupt the decision margin:

  A. Margin Preservation Cascade (baseline vs. sufficiency vs. necessity)
  B. Sufficiency–Necessity Trade-off Scatter
  C. Ratio Comparison Heatmap
  D. Correctness Stratification Violins

Outputs are stored under ``outputs/analytics/consistency/plots/<dataset>/``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DEFAULT_ROOT = Path("outputs/analytics/consistency")
SUMMARY_FILENAME = "consistency_summary.csv"
PLOTS_ROOT = "plots"
RATIO_LIMIT = (-1.0, 1.0)

METHOD_ORDER: Sequence[str] = ("graphsvx", "subgraphx", "token_shap_llm")
METHOD_LABELS: Dict[str, str] = {
    "graphsvx": "GraphSVX (GNN)",
    "subgraphx": "SubgraphX (GNN)",
    "token_shap_llm": "TokenSHAP (LLM)",
}
METHOD_COLORS: Dict[str, str] = {
    "graphsvx": "#3498db",
    "subgraphx": "#9b59b6",
    "token_shap_llm": "#2ecc71",
}
GRAPH_ORDER = ["skipgrams", "window", "constituency", "syntactic", "tokens"]
GRAPH_COLORS: Dict[str, str] = {
    "skipgrams": "#1f77b4",
    "window": "#ff7f0e",
    "constituency": "#2ca02c",
    "syntactic": "#d62728",
    "tokens": "#9467bd",
}

DATASET_LABELS: Dict[str, str] = {
    "setfit_ag_news": "AG News (SetFit)",
    "stanfordnlp_sst2": "SST-2 (StanfordNLP)",
}


def dataset_label(dataset: str) -> str:
    return DATASET_LABELS.get(dataset, dataset.replace("_", " ").title())


def graph_title_text(graph: str) -> str:
    return graph.replace("_", " ").title()


def order_graphs(graphs: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for candidate in GRAPH_ORDER:
        if candidate in graphs and candidate not in seen:
            ordered.append(candidate)
            seen.add(candidate)
    for graph in graphs:
        if graph not in seen:
            ordered.append(graph)
            seen.add(graph)
    return ordered


def lighten_to_rgba(hex_color: str, *, factor: float = 0.65, alpha: float = 0.2) -> str:
    """Lighten a hex colour by blending toward white and return as rgba string."""
    color = hex_color.lstrip("#")
    if len(color) != 6:
        raise ValueError(f"Expected hex colour #RRGGBB, received {hex_color}")
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    factor_clamped = max(0.0, min(1.0, factor))
    r_l = int(r + (255 - r) * factor_clamped)
    g_l = int(g + (255 - g) * factor_clamped)
    b_l = int(b + (255 - b) * factor_clamped)
    alpha_clamped = max(0.0, min(1.0, alpha))
    return f"rgba({r_l},{g_l},{b_l},{alpha_clamped:.3f})"


def with_alpha(hex_color: str, alpha: float) -> str:
    color = hex_color.lstrip("#")
    if len(color) != 6:
        raise ValueError(f"Expected hex colour #RRGGBB, received {hex_color}")
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    alpha_clamped = max(0.0, min(1.0, alpha))
    return f"rgba({r},{g},{b},{alpha_clamped:.3f})"


def load_summary(root: Path, field: str) -> pd.DataFrame:
    path = root / field / SUMMARY_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Field summary missing: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Field summary is empty: {path}")
    return df


def load_instance_records(root: Path, dataset: str) -> pd.DataFrame:
    """
    Load instance-level records from method/dataset/graph.csv files.
    The extract_gnn.py and extract_llm.py scripts write these CSVs with all necessary columns.
    """
    rows: List[pd.DataFrame] = []
    
    # Map from display dataset names (like "setfit_ag_news") to the various possible CSV dataset patterns
    # The CSVs use dataset names like "ag-news" (with hyphens), not "setfit_ag_news"
    dataset_map = {
        "setfit_ag_news": "setfit_ag_news",
        "stanfordnlp_sst2": "stanfordnlp_sst2",
    }
    dataset_slug = dataset_map.get(dataset, dataset)
    
    for method in METHOD_ORDER:
        method_dir = root / method / dataset_slug
        if not method_dir.exists():
            continue
        for csv_path in method_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"Warning: Failed to read {csv_path}: {e}")
                continue
            if df.empty:
                continue
            df = df.copy()
            # Ensure method, dataset, and graph_type columns exist
            if "method" not in df.columns:
                df["method"] = method
            if "dataset" not in df.columns:
                df["dataset"] = dataset
            if "graph_type" not in df.columns:
                df["graph_type"] = csv_path.stem
            rows.append(df)
    
    if not rows:
        return pd.DataFrame()
    
    data = pd.concat(rows, ignore_index=True)
    
    # Ensure sufficiency_ratio and necessity_ratio exist and are properly computed
    if "sufficiency_ratio" not in data.columns or data["sufficiency_ratio"].isna().all():
        denominator = data.get("baseline_margin", np.nan).replace(0, np.nan)
        if "preservation_sufficiency" in data.columns:
            data["sufficiency_ratio"] = data["preservation_sufficiency"] / denominator
    
    if "necessity_ratio" not in data.columns or data["necessity_ratio"].isna().all():
        denominator = data.get("baseline_margin", np.nan).replace(0, np.nan)
        if "preservation_necessity" in data.columns:
            data["necessity_ratio"] = data["preservation_necessity"] / denominator
    
    # Clip ratios to valid range
    data["sufficiency_ratio"] = data["sufficiency_ratio"].clip(*RATIO_LIMIT)
    data["necessity_ratio"] = data["necessity_ratio"].clip(*RATIO_LIMIT)
    data["dataset_slug"] = dataset
    
    return data


def save_figure(fig: go.Figure, stem: Path, width: int, height: int) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = stem.with_suffix(".pdf")
    html_path = stem.with_suffix(".html")
    upscale = 4
    download_config = {
        "toImageButtonOptions": {
            "format": "png",
            "filename": stem.name,
            "height": height,
            "width": width,
            "scale": upscale,
        }
    }
    try:
        fig.write_image(str(pdf_path), width=width, height=height)
        print(f"  ✓ {pdf_path}")
    except Exception as exc:  # pragma: no cover - environment specific
        print(f"  ! PDF failed ({pdf_path.name}): {exc}")
    try:
        fig.write_html(
            str(html_path),
            include_plotlyjs="cdn",
            full_html=True,
            config=download_config,
        )
        print(f"  ✓ {html_path}")
    except Exception as exc:  # pragma: no cover - environment specific
        print(f"  ! HTML failed ({html_path.name}): {exc}")


def _confidence_interval(std: float, count: float) -> float:
    if not np.isfinite(std) or not np.isfinite(count) or count <= 1:
        return 0.0
    return 1.96 * std / math.sqrt(count)


def create_margin_preservation_cascade(
    baseline_df: pd.DataFrame,
    sufficiency_df: pd.DataFrame,
    necessity_df: pd.DataFrame,
    dataset: str,
    output_dir: Path,
) -> None:
    metric_specs = [
        ("baseline_margin", baseline_df, "Baseline Margin"),
        ("preservation_sufficiency", sufficiency_df, "Preservation Sufficiency"),
        ("preservation_necessity", necessity_df, "Preservation Necessity"),
    ]

    fig = make_subplots(
        rows=1,
        cols=len(metric_specs),
        subplot_titles=[title for _, _, title in metric_specs],
        shared_yaxes=True,
    )

    has_any = False

    for col_idx, (metric_key, source_df, title) in enumerate(metric_specs, start=1):
        subset = source_df[
            (source_df["dataset"] == dataset)
            & (source_df["group"] == "overall")
        ].copy()
        if subset.empty:
            continue

        subset["graph_title"] = subset["graph"].apply(graph_title_text)
        graphs = order_graphs(subset["graph"].unique())
        if not graphs:
            continue

        has_any = True

        for method in METHOD_ORDER:
            method_subset = subset[subset["method"] == method]
            if method_subset.empty:
                continue

            prepared = method_subset.copy()
            prepared.set_index("graph", inplace=True)
            means = [float(prepared.loc[g, "mean"]) if g in prepared.index else np.nan for g in graphs]
            cis = [
                _confidence_interval(
                    float(prepared.loc[g, "std"]) if g in prepared.index else np.nan,
                    float(prepared.loc[g, "count"]) if g in prepared.index else np.nan,
                )
                for g in graphs
            ]
            upper = [m + c if np.isfinite(m) else np.nan for m, c in zip(means, cis)]
            lower = [m - c if np.isfinite(m) else np.nan for m, c in zip(means, cis)]

            color = METHOD_COLORS.get(method, "#7f8c8d")
            method_label = METHOD_LABELS.get(method, method)

            fig.add_trace(
                go.Scatter(
                    x=[graph_title_text(g) for g in graphs],
                    y=upper,
                    mode="lines",
                    line=dict(width=0),
                    legendgroup=method_label,
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=1,
                col=col_idx,
            )
            fig.add_trace(
                go.Scatter(
                    x=[graph_title_text(g) for g in graphs],
                    y=lower,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=with_alpha(color, 0.25),
                    legendgroup=method_label,
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=1,
                col=col_idx,
            )
            fig.add_trace(
                go.Scatter(
                    x=[graph_title_text(g) for g in graphs],
                    y=means,
                    mode="lines+markers",
                    name=method_label if col_idx == 1 else None,
                    legendgroup=method_label,
                    line=dict(color=color, width=3),
                    marker=dict(size=8, symbol="circle", line=dict(color="#333333", width=1)),
                    showlegend=col_idx == 1,
                ),
                row=1,
                col=col_idx,
            )

        fig.add_hline(y=0, line_dash="dot", line_color="#7f8c8d", row=1, col=col_idx)
        fig.update_xaxes(title_text="Graph Type", row=1, col=col_idx, tickangle=-30)

    if not has_any:
        return

    fig.update_yaxes(title_text="Decision Margin", range=[-1.05, 1.05])
    fig.update_layout(
        title=dict(
            text=(
                "<b>Margin Preservation Cascade</b><br>"
                f"<sub>{dataset_label(dataset)} · lines = methods, columns = margin view</sub>"
            ),
            x=0.5,
            xanchor="center",
        ),
        height=1080,
        width=1920,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=11),
    )

    save_figure(fig, output_dir / f"vizA_margin_preservation_cascade_{dataset}", width=1300, height=520)


def build_tradeoff_scatter_figure(
    instances: pd.DataFrame,
    dataset: str,
    output_dir: Optional[Path] = None,
) -> Optional[go.Figure]:
    """Return the trade-off scatter figure for a dataset (or None if no data)."""
    if instances.empty:
        return None

    subset = instances[instances.get("dataset_slug") == dataset].copy()
    if subset.empty:
        return None

    # Get unique classes
    classes = sorted(subset["prediction_class"].dropna().unique().tolist())
    if not classes:
        classes = ["All"]
        subset = subset.assign(prediction_class="All")

    class_titles = [f"<b>Class {cls}</b>" for cls in classes]
    cols = len(classes)
    rows = 1
    subplot_titles = class_titles
    left_margin, right_margin = 90, 70
    top_margin, bottom_margin = 140, 80
    horizontal_spacing = 0.05
    subplot_target_px = 620
    domain_width = (1 - horizontal_spacing * max(cols - 1, 0)) / cols
    domain_width = max(domain_width, 0.2)
    usable_width = subplot_target_px / domain_width
    figure_width = max(960, usable_width + left_margin + right_margin)
    subplot_height = subplot_target_px
    figure_height = max(720, top_margin + bottom_margin + subplot_height)

    # Summarise quadrant distribution for this dataset
    quadrant_subset = subset[
        np.isfinite(subset["sufficiency_ratio"]) & np.isfinite(subset["necessity_ratio"])
    ].copy()
    if not quadrant_subset.empty:
        suff_nonneg = quadrant_subset["sufficiency_ratio"] >= 0
        nec_nonneg = quadrant_subset["necessity_ratio"] >= 0
        quadrant_subset["quadrant"] = np.select(
            [
                suff_nonneg & ~nec_nonneg,
                suff_nonneg & nec_nonneg,
                ~suff_nonneg & ~nec_nonneg,
                ~suff_nonneg & nec_nonneg,
            ],
            [
                "Sufficient-Necessary",
                "Sufficient-Redundant",
                "Insufficient-Necessary",
                "Insufficient-Redundant",
            ],
            default="Unknown",
        )
        group_cols = ["prediction_class", "method", "graph_type", "is_correct"]
        summary = (
            quadrant_subset.groupby(group_cols + ["quadrant"], dropna=False)
            .size()
            .rename("count")
            .reset_index()
        )
        if not summary.empty and output_dir is not None:
            summary["dataset"] = dataset
            summary["total_count"] = summary.groupby(group_cols, dropna=False)["count"].transform("sum")
            summary["proportion"] = summary["count"] / summary["total_count"]
            summary.sort_values(group_cols + ["quadrant"], inplace=True)
            summary_path = output_dir / f"vizB_quadrant_summary_{dataset}.csv"
            summary.to_csv(summary_path, index=False)
            print(f"  ✓ {summary_path}")

    fig = make_subplots(
        rows=1,
        cols=cols,
        subplot_titles=subplot_titles,
        shared_xaxes=False,
        shared_yaxes=False,
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=0.08,
    )

    class_annotation_idx = 0
    for annotation in fig.layout.annotations:
        if isinstance(annotation.text, str) and annotation.text.startswith("<b>Class"):
            axis_suffix = "" if class_annotation_idx == 0 else str(class_annotation_idx + 1)
            xref = "x domain" if axis_suffix == "" else f"x{axis_suffix} domain"
            annotation.update(
                x=0.5,
                xref=xref,
                yref="paper",
                y=1.01,
                yshift=-5,
            )
            class_annotation_idx += 1

    has_data = False
    correctness_colors = {True: "#2ecc71", False: "#e74c3c"}
    symbol_cycle = [
        "circle", "square", "diamond", "triangle-up", "triangle-down", "star", "cross", "hexagon",
        "triangle-left", "triangle-right", "x", "asterisk", "hexagon2", "bowtie", "hourglass"
    ]
    combo_symbols: Dict[Tuple[str, str], str] = {}
    combo_index = 0
    for method in METHOD_ORDER:
        method_graphs = order_graphs(subset[subset["method"] == method]["graph_type"].unique())
        for graph in method_graphs:
            combo_symbols[(method, graph)] = symbol_cycle[combo_index % len(symbol_cycle)]
            combo_index += 1

    legend_shown: set[Tuple[str, str, bool]] = set()
    axis_meta: List[Tuple[str, str, int, int]] = []

    quadrant_fills = {
        "Sufficient-Necessary": lighten_to_rgba("#1a9850", factor=0.45, alpha=0.25),
        "Sufficient-Redundant": lighten_to_rgba("#91bfdb", factor=0.4, alpha=0.25),
        "Insufficient-Necessary": lighten_to_rgba("#fc8d59", factor=0.45, alpha=0.25),
        "Insufficient-Redundant": lighten_to_rgba("#d73027", factor=0.4, alpha=0.25),
    }

    for idx, cls in enumerate(classes):
        row_idx = 1
        col_idx = idx + 1
        class_subset = subset[subset["prediction_class"] == cls]
        if class_subset.empty:
            continue
        axis_index = idx + 1
        axis_suffix = "" if axis_index == 1 else str(axis_index)
        x_axis_name = f"x{axis_suffix}" if axis_suffix else "x"
        y_axis_name = f"y{axis_suffix}" if axis_suffix else "y"
        axis_meta.append((x_axis_name, y_axis_name, row_idx, col_idx))

        # Background quadrants
        fig.add_shape(
            type="rect",
            x0=0,
            x1=1,
            y0=-1,
            y1=0,
            xref=x_axis_name,
            yref=y_axis_name,
            fillcolor=quadrant_fills["Sufficient-Necessary"],
            line=dict(width=0),
            layer="below",
        )
        fig.add_shape(
            type="rect",
            x0=0,
            x1=1,
            y0=0,
            y1=1,
            xref=x_axis_name,
            yref=y_axis_name,
            fillcolor=quadrant_fills["Sufficient-Redundant"],
            line=dict(width=0),
            layer="below",
        )
        fig.add_shape(
            type="rect",
            x0=-1,
            x1=0,
            y0=-1,
            y1=0,
            xref=x_axis_name,
            yref=y_axis_name,
            fillcolor=quadrant_fills["Insufficient-Necessary"],
            line=dict(width=0),
            layer="below",
        )
        fig.add_shape(
            type="rect",
            x0=-1,
            x1=0,
            y0=0,
            y1=1,
            xref=x_axis_name,
            yref=y_axis_name,
            fillcolor=quadrant_fills["Insufficient-Redundant"],
            line=dict(width=0),
            layer="below",
        )

        for method in METHOD_ORDER:
            method_subset = class_subset[class_subset["method"] == method]
            if method_subset.empty:
                continue
            graphs = order_graphs(method_subset["graph_type"].unique())
            for graph in graphs:
                combo_subset = method_subset[method_subset["graph_type"] == graph]
                if combo_subset.empty:
                    continue
                symbol = combo_symbols.get((method, graph), "circle")
                for correctness, color in correctness_colors.items():
                    status_subset = combo_subset[combo_subset.get("is_correct") == correctness]
                    if status_subset.empty:
                        continue
                    has_data = True
                    legend_key = (method, graph, bool(correctness))
                    showlegend = legend_key not in legend_shown
                    if showlegend:
                        legend_shown.add(legend_key)
                    fig.add_trace(
                        go.Scattergl(
                            x=status_subset["sufficiency_ratio"],
                            y=status_subset["necessity_ratio"],
                            mode="markers",
                            marker=dict(
                                size=7,
                                color=color,
                                symbol=symbol,
                                opacity=0.75,
                                line=dict(color="#333333", width=0.55),
                            ),
                            customdata=status_subset[["baseline_margin"]].to_numpy(),
                            hovertemplate=(
                                f"{'Correct' if correctness else 'Incorrect'} prediction<br>"
                                f"Method: {METHOD_LABELS.get(method, method)}<br>"
                                f"Graph: {graph_title_text(graph)}<br>"
                                "Suff. ratio: %{x:.3f}<br>"
                                "Nec. ratio: %{y:.3f}<br>"
                                "Baseline margin: %{customdata[0]:.3f}<extra></extra>"
                            ),
                            showlegend=showlegend,
                            legendgroup=f"{graph}_{correctness}",
                            name=(
                                "<b>"
                                f"{graph_title_text(graph)} · "
                                f"{'Correct' if correctness else 'Incorrect'}"
                                "</b>"
                            )
                            if showlegend
                            else None,
                        ),
                        row=row_idx,
                        col=col_idx,
                    )

    if not has_data:
        return None

    for x_axis_name, y_axis_name, row_idx, col_idx in axis_meta:
        fig.update_xaxes(
            range=RATIO_LIMIT,
            row=row_idx,
            col=col_idx,
            constrain="domain",
            ticks="outside",
            ticklen=5,
            tickwidth=1,
        )
        fig.update_yaxes(
            range=RATIO_LIMIT,
            row=row_idx,
            col=col_idx,
            title_text="<b>Necessity Ratio</b>" if col_idx == 1 else None,
            title_font=dict(size=22),
            scaleanchor=x_axis_name,
            scaleratio=1,
            ticks="outside",
            ticklen=5,
            tickwidth=1,
        )

    fig.add_annotation(
        text="<b>Sufficiency Ratio</b>",
        x=0.5,
        y=-0.1,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=22),
    )

    fig.update_layout(
        height=figure_height,
        width=figure_width,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="center",
            x=0.5,
            font=dict(size=20),
            traceorder="grouped",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.25)",
            borderwidth=1,
        ),
        margin=dict(t=top_margin, b=bottom_margin + 60, l=left_margin, r=right_margin),
    )

    return fig


def create_tradeoff_scatter(instances: pd.DataFrame, dataset: str, output_dir: Path) -> None:
    """
    Create scatter plots showing the sufficiency-necessity trade-off, stratified by:
    - Rows: Correctness (correct vs incorrect predictions)
    - Columns: Predicted class
    - Within each subplot: Different graph types shown with different symbols,
      and methods shown with different colors.
    """
    fig = build_tradeoff_scatter_figure(instances, dataset, output_dir=output_dir)
    if fig is None:
        return

    width = int(fig.layout.width) if fig.layout.width else 1920
    height = int(fig.layout.height) if fig.layout.height else 1080

    save_figure(
        fig,
        output_dir / f"vizB_tradeoff_scatter_{dataset}",
        width=width,
        height=height,
    )


def create_ratio_heatmap(
    sufficiency_ratio_df: pd.DataFrame,
    necessity_ratio_df: pd.DataFrame,
    margin_coherence_df: pd.DataFrame,
    dataset: str,
    output_dir: Path,
) -> None:
    """
    Create 2D heatmaps showing three metrics (sufficiency ratio, necessity ratio, margin coherence)
    with methods as rows and graph types as columns.
    """
    def lookup_map(df: pd.DataFrame) -> Dict[tuple[str, str], float]:
        subset = df[
            (df["dataset"] == dataset)
            & (df["group"] == "overall")
        ][["method", "graph", "mean"]]
        return {(row["method"], row["graph"]): float(row["mean"]) for _, row in subset.iterrows()}

    suff_map = lookup_map(sufficiency_ratio_df)
    nec_map = lookup_map(necessity_ratio_df)
    coh_map = lookup_map(margin_coherence_df)

    # Get all methods and graphs that exist in the data
    all_graphs = set()
    for mapping in (suff_map, nec_map, coh_map):
        all_graphs.update(graph for _, graph in mapping.keys())
    
    if not all_graphs:
        return
    
    # Order graphs and filter methods that have data
    ordered_graphs = order_graphs(all_graphs)
    available_methods = [m for m in METHOD_ORDER if any((m, g) in suff_map or (m, g) in nec_map or (m, g) in coh_map for g in ordered_graphs)]
    
    if not available_methods or not ordered_graphs:
        return

    # Build 2D matrices: rows = methods, columns = graph types
    def build_matrix(mapping: Dict[tuple[str, str], float]) -> np.ndarray:
        matrix = np.zeros((len(available_methods), len(ordered_graphs)))
        matrix[:] = np.nan
        for i, method in enumerate(available_methods):
            for j, graph in enumerate(ordered_graphs):
                matrix[i, j] = mapping.get((method, graph), np.nan)
        return matrix

    suff_matrix = build_matrix(suff_map)
    nec_matrix = build_matrix(nec_map)
    coh_matrix = build_matrix(coh_map)

    # Create row and column labels
    row_labels = [METHOD_LABELS.get(method, method) for method in available_methods]
    col_labels = [graph_title_text(graph) for graph in ordered_graphs]

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["Sufficiency Ratio", "Necessity Ratio", "Margin Coherence"],
        shared_yaxes=True,
        horizontal_spacing=0.12,
    )

    # Sufficiency ratio heatmap (0 to 1, green is good)
    fig.add_trace(
        go.Heatmap(
            z=suff_matrix,
            x=col_labels,
            y=row_labels,
            colorscale=[
                [0.0, "#d73027"],  # Red for low
                [0.5, "#fee08b"],  # Yellow for mid
                [1.0, "#1a9850"],  # Green for high
            ],
            zmin=0,
            zmax=1,
            colorbar=dict(
                title="Ratio",
                x=0.30,
                len=0.9,
            ),
            showscale=True,
            hovertemplate="Method: %{y}<br>Graph: %{x}<br>Suff. Ratio: %{z:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Necessity ratio heatmap (-1 to 1, negative is good, centered at 0)
    fig.add_trace(
        go.Heatmap(
            z=nec_matrix,
            x=col_labels,
            y=row_labels,
            colorscale=[
                [0.0, "#1a9850"],    # Green for -1 (good)
                [0.5, "#ffffbf"],    # Light yellow for 0
                [1.0, "#d73027"],    # Red for +1 (bad)
            ],
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar=dict(
                title="Ratio",
                x=0.64,
                len=0.9,
            ),
            showscale=True,
            hovertemplate="Method: %{y}<br>Graph: %{x}<br>Nec. Ratio: %{z:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # Margin coherence heatmap (0 to 1, higher is better)
    fig.add_trace(
        go.Heatmap(
            z=coh_matrix,
            x=col_labels,
            y=row_labels,
            colorscale=[
                [0.0, "#d73027"],  # Red for low
                [0.5, "#fee08b"],  # Yellow for mid
                [1.0, "#1a9850"],  # Green for high
            ],
            zmin=0,
            zmax=1,
            colorbar=dict(
                title="Coherence",
                x=1.0,
                len=0.9,
            ),
            showscale=True,
            hovertemplate="Method: %{y}<br>Graph: %{x}<br>Coherence: %{z:.3f}<extra></extra>",
        ),
        row=1,
        col=3,
    )

    # Update axes
    for col_idx in range(1, 4):
        fig.update_xaxes(
            tickangle=-45,
            row=1,
            col=col_idx,
            side="bottom",
        )
        fig.update_yaxes(
            row=1,
            col=col_idx,
        )

    fig.update_layout(
        title=dict(
            text=(
                "<b>Ratio Comparison Heatmap</b><br>"
                f"<sub>{dataset_label(dataset)} · Methods (rows) × Graph Types (columns)</sub>"
            ),
            x=0.5,
            xanchor="center",
        ),
        height=max(1080, 200 * len(available_methods) + 200),
        width=1400,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=11),
    )

    save_figure(
        fig,
        output_dir / f"vizC_ratio_comparison_heatmap_{dataset}",
        width=1920,
        height=max(1080, 200 * len(available_methods) + 200),
    )


def create_correctness_violins(instances: pd.DataFrame, dataset: str, output_dir: Path) -> None:
    if instances.empty:
        return

    subset = instances[instances.get("dataset_slug") == dataset]
    if subset.empty:
        return

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=["Correct predictions", "Incorrect predictions"],
        shared_yaxes=True,
    )

    has_data = False
    ratio_specs = [
        ("sufficiency_ratio", "Sufficiency Ratio", "#1890ff", "negative"),
        ("necessity_ratio", "Necessity Ratio", "#f39c12", "positive"),
    ]

    for row_idx, correctness in enumerate((True, False), start=1):
        row_subset = subset[subset.get("is_correct") == correctness]
        if row_subset.empty:
            continue
        has_data = True
        for method in METHOD_ORDER:
            method_subset = row_subset[row_subset["method"] == method]
            if method_subset.empty:
                continue
            graphs = order_graphs(method_subset["graph_type"].unique())
            for graph in graphs:
                exp_subset = method_subset[method_subset["graph_type"] == graph]
                if exp_subset.empty:
                    continue
                x_label = f"{METHOD_LABELS.get(method, method)}<br>{graph_title_text(graph)}"
                for metric, name, color, side in ratio_specs:
                    fig.add_trace(
                        go.Violin(
                            x=[x_label] * len(exp_subset),
                            y=exp_subset[metric],
                            legendgroup=name,
                            name=name if (row_idx == 1 and method == METHOD_ORDER[0] and graph == graphs[0]) else None,
                            line=dict(color=color, width=1.5),
                            fillcolor=with_alpha(color, 0.45),
                            meanline=dict(visible=True),
                            spanmode="hard",
                            span=RATIO_LIMIT,
                            side=side,
                            width=0.6,
                            points=False,
                            showlegend=(row_idx == 1 and method == METHOD_ORDER[0] and graph == graphs[0]),
                        ),
                        row=row_idx,
                        col=1,
                    )

    if not has_data:
        return

    fig.update_yaxes(title_text="Ratio", range=RATIO_LIMIT, row=1, col=1)
    fig.update_xaxes(title_text="Method · Graph", row=1, col=1)
    fig.update_xaxes(title_text="Method · Graph", row=2, col=1)
    fig.update_layout(
        title=dict(
            text=(
                "<b>Correctness Stratification Violins</b><br>"
                f"<sub>{dataset_label(dataset)} · left = sufficiency, right = necessity</sub>"
            ),
            x=0.5,
            xanchor="center",
        ),
        height=1080,
        width=1920,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=11),
    )

    save_figure(fig, output_dir / f"vizD_correctness_violins_{dataset}", width=1920, height=1080)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate decision margin consistency dashboards from aggregated summaries.")
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root directory containing decision margin summaries (default: outputs/analytics/consistency).",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    baseline_df = load_summary(root, "baseline_margin")
    sufficiency_df = load_summary(root, "preservation_sufficiency")
    necessity_df = load_summary(root, "preservation_necessity")
    suff_ratio_df = load_summary(root, "sufficiency_ratio")
    nec_ratio_df = load_summary(root, "necessity_ratio")
    coherence_df = load_summary(root, "margin_coherence")

    datasets = sorted(sufficiency_df["dataset"].unique())
    plots_root = root / PLOTS_ROOT

    print("=" * 120)
    print("Decision margin consistency visualisations")
    print("=" * 120)
    print(f"Root: {root}")
    print(f"Datasets: {datasets}")

    for dataset in datasets:
        dataset_dir = plots_root / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nDataset: {dataset_label(dataset)}")

        instance_records = load_instance_records(root, dataset)

        create_margin_preservation_cascade(baseline_df, sufficiency_df, necessity_df, dataset, dataset_dir)
        create_tradeoff_scatter(instance_records, dataset, dataset_dir)
        create_ratio_heatmap(suff_ratio_df, nec_ratio_df, coherence_df, dataset, dataset_dir)
        create_correctness_violins(instance_records, dataset, dataset_dir)

    print("\n" + "=" * 120)
    print("All decision margin consistency plots generated.")
    print("=" * 120)


if __name__ == "__main__":
    main()
