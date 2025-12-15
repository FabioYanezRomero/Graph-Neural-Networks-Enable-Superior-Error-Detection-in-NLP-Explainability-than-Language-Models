#!/usr/bin/env python3
"""
Generate top-k concentration bar plots that compare correct vs. incorrect subsets.

You can render a single dataset/field figure or stack multiple datasets (e.g.,
AG News on top, SST-2 below) into the same canvas for a paper-ready layout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure repo root on sys.path for imports when running as a script
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from Analytics.progression.importance_accumulation_plot import (  # noqa: E402
    DEFAULT_DATASET_LABELS,
    DEFAULT_GRAPH_LABELS,
    DEFAULT_GRAPH_TYPE_ORDER,
    DEFAULT_METHOD_BASE_COLORS,
    DEFAULT_METHOD_LABELS,
    DEFAULT_METHOD_ORDER,
    DEFAULT_TOPK_METRICS,
    FIELD_LABELS,
    METHOD_GRAPH_COLORS,
    TOPK_LABELS,
)

PROGRESSION_ROOT = Path("outputs/analytics/progression")
SUMMARY_FILENAME = "concentration_summary.csv"
DEFAULT_STACK_DATASETS = ["setfit_ag_news", "stanfordnlp_sst2"]
BARGAP = 0.15
BARGROUPGAP = 0.05
BAR_WIDTH = 0.36


def _load_field_dataframe(field_dir: Path) -> pd.DataFrame:
    summary = field_dir / SUMMARY_FILENAME
    if not summary.exists():
        raise FileNotFoundError(f"Missing concentration summary: {summary}")
    df = pd.read_csv(summary)
    mask = df["metric"].isin(DEFAULT_TOPK_METRICS)
    focus = df[mask].copy()
    if focus.empty:
        raise ValueError(f"No top-k data found in {summary}")
    return focus


def _clip_unit_interval(value: Optional[float]) -> Optional[float]:
    if value is None or not np.isfinite(value):
        return None
    return float(np.clip(value, 0.0, 1.0))


def _aggregate_stats(rows: pd.DataFrame) -> tuple[Optional[float], Optional[float]]:
    if rows.empty:
        return None, None
    counts = rows["count"].to_numpy(dtype=float)
    means = rows["mean"].to_numpy(dtype=float)
    stds = rows["std"].to_numpy(dtype=float)
    mask = np.isfinite(counts) & np.isfinite(means) & (counts > 0)
    if not np.any(mask):
        return None, None
    counts = counts[mask]
    means = means[mask]
    stds = stds[mask]
    stds = np.where(np.isfinite(stds), stds, 0.0)
    total = counts.sum()
    if total <= 0:
        return None, None
    mean = float(np.dot(means, counts) / total)
    var_num = np.sum((counts - 1.0) * (stds ** 2))
    var_num += np.sum(counts * (means - mean) ** 2)
    dof = total - 1.0
    variance = var_num / dof if dof > 0 else np.nan
    std = float(np.sqrt(variance)) if np.isfinite(variance) and variance >= 0 else None
    return _clip_unit_interval(mean), std


def _series_for_dataset(df: pd.DataFrame, dataset: str) -> List[Dict[str, object]]:
    dataset_df = df[df["dataset"] == dataset]
    if dataset_df.empty:
        return []

    entries: List[Dict[str, object]] = []
    for method in DEFAULT_METHOD_ORDER:
        method_slice = dataset_df[dataset_df["method"] == method]
        if method_slice.empty:
            continue
        graph_order = DEFAULT_GRAPH_TYPE_ORDER.get(
            method, tuple(sorted(method_slice["graph"].unique()))
        )
        for graph in graph_order:
            graph_slice = method_slice[method_slice["graph"] == graph]
            if graph_slice.empty:
                continue
            metrics_map: Dict[str, Dict[str, float]] = {}
            for metric in DEFAULT_TOPK_METRICS:
                metric_row = graph_slice[graph_slice["metric"] == metric]
                if metric_row.empty:
                    continue
                overall_row = metric_row[metric_row["group"] == "overall"]
                overall_mean = None
                overall_std = None
                if not overall_row.empty:
                    overall_mean = _clip_unit_interval(
                        float(overall_row["mean"].iloc[0])
                    )
                    std_val = overall_row["std"].iloc[0]
                    overall_std = float(std_val) if np.isfinite(std_val) else None
                correct_rows = metric_row[
                    metric_row["group"].str.contains("correct_True", na=False)
                ]
                incorrect_rows = metric_row[
                    metric_row["group"].str.contains("correct_False", na=False)
                ]
                correct_mean, correct_std = _aggregate_stats(correct_rows)
                incorrect_mean, incorrect_std = _aggregate_stats(incorrect_rows)

                metrics_map[metric] = {
                    "metric": metric,
                    "label": TOPK_LABELS[metric],
                    "overall_mean": overall_mean,
                    "overall_std": overall_std,
                    "correct_mean": correct_mean,
                    "correct_std": correct_std,
                    "incorrect_mean": incorrect_mean,
                    "incorrect_std": incorrect_std,
                }
            if not metrics_map:
                continue
            method_label = DEFAULT_METHOD_LABELS.get(method, method)
            graph_label = DEFAULT_GRAPH_LABELS.get(graph, graph)
            display_name = f"{method_label} — {graph_label}"
            color = METHOD_GRAPH_COLORS.get(
                (method, graph),
                DEFAULT_METHOD_BASE_COLORS.get(method, "#636EFA"),
            )
            entries.append(
                {
                    "name": display_name,
                    "color": color,
                    "metrics_by_name": metrics_map,
                }
            )
    return entries


def _format_bar_text(value: Optional[float]) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return f"{value:.3f}".rstrip("0").rstrip(".") or "0"


def _format_hover(
    module_name: str,
    metric_label: str,
    subset_label: str,
    mean_value: Optional[float],
    std_value: Optional[float],
    overall_value: Optional[float],
) -> str:
    def fmt(val: Optional[float]) -> str:
        return "N/A" if val is None or not np.isfinite(val) else f"{val:.4f}"

    return (
        f"<b>{module_name}</b><br>"
        f"Metric: {metric_label}<br>"
        f"Set: {subset_label}<br>"
        f"Clipped mean: {fmt(mean_value)}<br>"
        f"Std: {fmt(std_value)}<br>"
        f"Overall mean: {fmt(overall_value)}"
    )


def _lighten_color(hex_color: str, factor: float = 0.5) -> str:
    color = hex_color.lstrip("#")
    if len(color) != 6:
        return hex_color
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def _error_array(values: Sequence[Optional[float]]) -> List[float]:
    result: List[float] = []
    for val in values:
        if val is None or not np.isfinite(val):
            result.append(0.0)
        else:
            result.append(float(abs(val)))
    return result


def _add_dataset_row(
    fig: go.Figure,
    row: int,
    entries: Sequence[Dict[str, object]],
    showlegend: bool,
) -> bool:
    row_has_data = False
    for col, metric in enumerate(DEFAULT_TOPK_METRICS, start=1):
        module_labels: List[str] = []
        colors: List[str] = []
        correct_vals: List[Optional[float]] = []
        incorrect_vals: List[Optional[float]] = []
        correct_errs: List[Optional[float]] = []
        incorrect_errs: List[Optional[float]] = []
        hover_correct: List[str] = []
        hover_incorrect: List[str] = []

        for entry in entries:
            metrics_map = entry["metrics_by_name"]
            info = metrics_map.get(metric)
            if not info:
                continue
            module_labels.append(f"<b>{entry['name'].replace(' — ', '<br>')}</b>")
            colors.append(entry["color"])
            correct_mean = info.get("correct_mean")
            incorrect_mean = info.get("incorrect_mean")
            correct_vals.append(correct_mean)
            incorrect_vals.append(incorrect_mean)
            correct_errs.append(info.get("correct_std"))
            incorrect_errs.append(info.get("incorrect_std"))
            hover_correct.append(
                _format_hover(
                    entry["name"],
                    info["label"],
                    "Correct",
                    correct_mean,
                    info.get("correct_std"),
                    info.get("overall_mean"),
                )
            )
            hover_incorrect.append(
                _format_hover(
                    entry["name"],
                    info["label"],
                    "Incorrect",
                    incorrect_mean,
                    info.get("incorrect_std"),
                    info.get("overall_mean"),
                )
            )

        if not module_labels:
            continue
        row_has_data = True
        legend_correct = showlegend and (col == 1)
        legend_incorrect = showlegend and (col == 1)

        fig.add_trace(
            go.Bar(
                x=module_labels,
                y=correct_vals,
                name="Correct",
                legendgroup="correct",
                showlegend=legend_correct,
                marker=dict(color=colors, line=dict(color="black", width=1.2)),
                text=[_format_bar_text(v) for v in correct_vals],
                textposition="outside",
                texttemplate="%{text}",
                width=BAR_WIDTH,
                cliponaxis=False,
                offsetgroup=f"correct_{row}_{col}",
                customdata=hover_correct,
                offset=-BAR_WIDTH / 2,
                hovertemplate="%{customdata}<extra></extra>",
                error_y=dict(
                    type="data",
                    array=_error_array(correct_errs),
                    visible=True,
                    color="rgba(20,20,20,0.75)",
                    thickness=1.2,
                    width=2,
                ),
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Bar(
                x=module_labels,
                y=incorrect_vals,
                name="Incorrect",
                legendgroup="incorrect",
                showlegend=legend_incorrect,
                marker=dict(
                    color=[_lighten_color(c, 0.6) for c in colors],
                    line=dict(color="black", width=1.2),
                    opacity=0.85,
                ),
                text=[_format_bar_text(v) for v in incorrect_vals],
                textposition="outside",
                texttemplate="%{text}",
                width=BAR_WIDTH,
                cliponaxis=False,
                offsetgroup=f"incorrect_{row}_{col}",
                customdata=hover_incorrect,
                offset=BAR_WIDTH / 2,
                hovertemplate="%{customdata}<extra></extra>",
                error_y=dict(
                    type="data",
                    array=_error_array(incorrect_errs),
                    visible=True,
                    color="rgba(20,20,20,0.55)",
                    thickness=1.1,
                    width=2,
                ),
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(
            tickangle=-45,
            row=row,
            col=col,
            matches="x1" if (row, col) != (1, 1) else None,
        )
        fig.update_yaxes(
            range=[0, 1],
            row=row,
            col=col,
            matches="y1" if (row, col) != (1, 1) else None,
        )
    return row_has_data


def build_bar_figure(
    entries: Sequence[Dict[str, object]],
    dataset_label: str,
    field_label: str,
) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=4,
        shared_yaxes=True,
        subplot_titles=[f"<b>{TOPK_LABELS[m]}</b>" for m in DEFAULT_TOPK_METRICS],
        horizontal_spacing=0.04,
    )
    _add_dataset_row(fig, 1, entries, showlegend=True)

    fig.update_yaxes(title="<b>Clipped mean concentration</b>", row=1, col=1)
    fig.update_layout(
        barmode="group",
        bargap=BARGAP,
        bargroupgap=BARGROUPGAP,
        title=(
            f"<b>Top-k Concentration — {dataset_label}</b><br>"
            f"<sub>{field_label}</sub>"
        ),
        template="plotly_white",
        height=520,
        width=1800,
        margin=dict(t=120, b=80, l=80, r=40),
    )
    return fig


def build_stacked_figure(
    dataset_entries: Sequence[Sequence[Dict[str, object]]],
    dataset_labels: Sequence[str],
    field_label: str,
) -> go.Figure:
    rows = len(dataset_entries)
    if rows == 0:
        raise ValueError("No dataset entries provided for stacked figure.")

    subplot_titles: List[str] = []
    for label in dataset_labels:
        for metric in DEFAULT_TOPK_METRICS:
            subplot_titles.append(f"<b>{label} — {TOPK_LABELS[metric]}</b>")

    fig = make_subplots(
        rows=rows,
        cols=4,
        shared_yaxes=True,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.04,
        vertical_spacing=0.2,
    )
    for idx, (entries, _) in enumerate(zip(dataset_entries, dataset_labels), start=1):
        _add_dataset_row(fig, idx, entries, showlegend=(idx == 1))

    fig.update_yaxes(title="<b>Clipped mean concentration</b>", row=1, col=1)
    fig.update_layout(
        title=f"<b>Top-k Concentration — {field_label}</b>",
        template="plotly_white",
        barmode="group",
        bargap=BARGAP,
        bargroupgap=BARGROUPGAP,
        legend=dict(
            orientation="h",
            x=0.5,
            y=1.02,
            xanchor="center",
            yanchor="bottom",
        ),
        height=520 * rows,
        width=1800,
        margin=dict(t=120, b=80, l=90, r=40),
    )
    return fig


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a focused top-k bar plot for a single dataset."
    )
    parser.add_argument(
        "--field",
        type=str,
        default="maskout_progression_drop",
        choices=list(FIELD_LABELS.keys()),
        help="Progression field to visualize.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="setfit_ag_news",
        help="Dataset identifier when generating a single figure.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Datasets to include when stacking (default: AG News then SST-2).",
    )
    parser.add_argument(
        "--stack-datasets",
        action="store_true",
        help="Stack the datasets vertically into a single figure.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path for the HTML figure.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    field_dir = PROGRESSION_ROOT / args.field
    df = _load_field_dataframe(field_dir)
    field_label = FIELD_LABELS.get(args.field, args.field)

    if args.stack_datasets:
        dataset_ids = args.datasets if args.datasets else DEFAULT_STACK_DATASETS
        entries_by_dataset: List[Sequence[Dict[str, object]]] = []
        label_list: List[str] = []
        for dataset in dataset_ids:
            entry = _series_for_dataset(df, dataset)
            if not entry:
                print(f"Warning: no data for dataset '{dataset}' in field '{args.field}'. Skipping.")
                continue
            entries_by_dataset.append(entry)
            label_list.append(DEFAULT_DATASET_LABELS.get(dataset, dataset))
        if not entries_by_dataset:
            raise SystemExit("No datasets available for stacked figure.")
        fig = build_stacked_figure(entries_by_dataset, label_list, field_label)
        if args.output:
            output_path = args.output
        else:
            plot_dir = field_dir / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            output_path = plot_dir / f"topk_concentration_{args.field}_stacked.html"
    else:
        entries = _series_for_dataset(df, args.dataset)
        if not entries:
            raise SystemExit(
                f"No data for dataset '{args.dataset}' in field '{args.field}'."
            )
        dataset_label = DEFAULT_DATASET_LABELS.get(args.dataset, args.dataset)
        fig = build_bar_figure(entries, dataset_label, field_label)
        if args.output:
            output_path = args.output
        else:
            plot_dir = field_dir / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            slug = dataset_label.replace(" ", "_").lower()
            output_path = plot_dir / f"topk_concentration_{slug}.html"

    config = {
        "toImageButtonOptions": {
            "format": "png",
            "filename": f"{args.field}_topk_concentration",
            "width": 1800,
            "height": fig.layout.height or 520,
            "scale": 2,
        }
    }
    fig.write_html(str(output_path), include_plotlyjs="cdn", config=config)
    print(f"Wrote top-k concentration bar plot to: {output_path}")


if __name__ == "__main__":
    main()
