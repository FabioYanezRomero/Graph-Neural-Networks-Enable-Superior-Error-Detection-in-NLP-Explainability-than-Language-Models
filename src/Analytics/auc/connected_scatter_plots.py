from __future__ import annotations

import argparse
import os
import sys
from itertools import cycle
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots


DEFAULT_INPUT_CSV = Path(
    "outputs/analytics/auc/plots/final_metrics/detection_rates_by_dataset.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/analytics/auc/plots/final_metrics/connected_scatter"
)
DEFAULT_COMBINED_LAYOUTS = [
    {
        "datasets": [
            ("ag-news", "AG News"),
            ("sst-2", "SST-2"),
        ],
        "title": "Error and Correctness Detection Rates — AG News vs. SST-2",
        "filename": "ag_news_vs_sst2_connected_scatter.html",
    }
]

INSERTION_INPUT_CSV = Path(
    "outputs/analytics/auc/plots/insertion_metrics/detection_rates_by_dataset.csv"
)

AUTOMATIC_EXPORT_BATCHES = [
    {
        "label": "Deletion AUC",
        "input_csv": DEFAULT_INPUT_CSV,
        "output_dir": DEFAULT_OUTPUT_DIR,
        "variant_label": "Deletion AUC",
        "filename_suffix": "deletion",
    },
    {
        "label": "Insertion AUC",
        "input_csv": INSERTION_INPUT_CSV,
        "output_dir": DEFAULT_OUTPUT_DIR,
        "variant_label": "Insertion AUC",
        "filename_suffix": "insertion",
    },
]

LEGEND_BELOW = {
    "orientation": "h",
    "yanchor": "top",
    "y": -0.2,
    "xanchor": "center",
    "x": 0.5,
    "groupclick": "togglegroup",
    "font": {"size": 13},
    "itemwidth": 50,
}

STANDARD_MARGINS = {"t": 40, "l": 60, "r": 20, "b": 120}

STYLE_HINT_TEXT = "Solid lines = Error detection • Dashed lines = Correctness detection"
STYLE_HINT_OFFSET = -0.32


def _apply_suffix(filename: str, suffix: str) -> str:
    if not suffix:
        return filename
    stem, ext = os.path.splitext(filename)
    return f"{stem}_{suffix}{ext or '.html'}"


def _append_style_hint(fig: go.Figure) -> None:
    fig.add_annotation(
        text=STYLE_HINT_TEXT,
        xref="paper",
        yref="paper",
        x=0.5,
        y=STYLE_HINT_OFFSET,
        showarrow=False,
        font=dict(size=11, color="#4a4a4a"),
        align="center",
    )


def _compute_intersections(x: np.ndarray, y_err: np.ndarray, y_corr: np.ndarray) -> list[tuple[float, float]]:
    intersections: list[tuple[float, float]] = []
    if len(x) < 2:
        return intersections
    for i in range(len(x) - 1):
        e1, e2 = y_err[i], y_err[i + 1]
        c1, c2 = y_corr[i], y_corr[i + 1]
        if any(np.isnan([e1, e2, c1, c2])):
            continue
        diff1 = e1 - c1
        diff2 = e2 - c2
        if diff1 == 0:
            intersections.append((x[i], e1))
        elif diff1 * diff2 < 0:
            ratio = diff1 / (diff1 - diff2)
            x_cross = x[i] + ratio * (x[i + 1] - x[i])
            y_cross = e1 + ratio * (e2 - e1)
            intersections.append((x_cross, y_cross))
    return intersections




def _ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_detection_rates(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Detection rate data not found: {csv_path}")
    df = pd.read_csv(csv_path)
    expected_columns = {
        "method",
        "dataset",
        "graph_type",
        "threshold",
        "mispredictions",
        "detections",
        "rate",
        "correct_predictions",
        "correct_retained",
        "correct_rate",
    }
    missing = expected_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in detection CSV: {sorted(missing)}")
    return df


def build_connected_scatter(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    datasets: Optional[Sequence[str]] = None,
    methods: Optional[Sequence[str]] = None,
    graph_types: Optional[Sequence[str]] = None,
    variant_label: Optional[str] = None,
    filename_suffix: str = "",
) -> list[Path]:
    output_dir = _ensure_output_dir(output_dir)
    dataset_filter = set(datasets) if datasets else None
    method_filter = set(methods) if methods else None
    graph_filter = set(graph_types) if graph_types else None
    created_files: list[Path] = []

    palette = qualitative.Bold + qualitative.Safe + qualitative.Light24
    color_lookup: dict[str, str] = {}
    palette_cycle = cycle(palette)

    for dataset, dataset_group in df.groupby("dataset", sort=True):
        if dataset_filter and dataset not in dataset_filter:
            continue

        fig = go.Figure()
        legend_rank = 0
        for (method, graph_type), method_group in dataset_group.groupby(
            ["method", "graph_type"], sort=True
        ):
            if method_filter and method not in method_filter:
                continue
            if graph_filter and graph_type not in graph_filter:
                continue
            ordered = method_group.sort_values("threshold").reset_index(drop=True)
            display_name = method
            if pd.notna(graph_type):
                display_name = f"{method} ({graph_type})"
            if display_name not in color_lookup:
                color_lookup[display_name] = next(palette_cycle)
            color = color_lookup[display_name]

            error_series = ordered["rate"]
            correct_series = ordered["correct_rate"]
            any_error = error_series.notna().any()
            any_correct = correct_series.notna().any()
            value_arrays: list[np.ndarray] = []
            if any_error:
                value_arrays.append(error_series.to_numpy())
            if any_correct:
                value_arrays.append(correct_series.to_numpy())
            if value_arrays:
                concatenated = np.concatenate(value_arrays)
                finite_values = concatenated[~np.isnan(concatenated)]
            else:
                finite_values = np.array([])

            legend_entry_added = False
            if any_error:
                fig.add_trace(
                    go.Scatter(
                        x=ordered["threshold"],
                        y=error_series,
                        mode="lines",
                        name=display_name,
                        legendgroup=display_name,
                        legendrank=legend_rank,
                        showlegend=True,
                        line=dict(color=color, dash="solid"),
                        hovertemplate=(
                            "Method: %{customdata[0]}<br>"
                            "Graph type: %{customdata[1]}<br>"
                            "Threshold: %{x:.2f}<br>"
                            "Error detection rate: %{y:.1%}<extra></extra>"
                        ),
                        customdata=ordered[["method", "graph_type"]].to_numpy(),
                    )
                )
                legend_entry_added = True
                legend_rank += 1
            if any_correct:
                showlegend = not legend_entry_added
                fig.add_trace(
                    go.Scatter(
                        x=ordered["threshold"],
                        y=correct_series,
                        mode="lines",
                        name=display_name if showlegend else f"{display_name} — Correct retention",
                        legendgroup=display_name,
                        legendrank=legend_rank if showlegend else None,
                        showlegend=showlegend,
                        line=dict(color=color, dash="dash"),
                        hovertemplate=(
                            "Method: %{customdata[0]}<br>"
                            "Graph type: %{customdata[1]}<br>"
                            "Threshold: %{x:.2f}<br>"
                            "Correct retention rate: %{y:.1%}<extra></extra>"
                        ),
                        customdata=ordered[["method", "graph_type"]].to_numpy(),
                    )
                )
                if showlegend:
                    legend_rank += 1
            if any_error and any_correct:
                intersections = _compute_intersections(
                    ordered["threshold"].to_numpy(),
                    error_series.to_numpy(),
                    correct_series.to_numpy(),
                )
                if intersections:
                    xs, ys = zip(*intersections)
                    fig.add_trace(
                        go.Scatter(
                            x=xs,
                            y=ys,
                            mode="markers",
                            marker=dict(color=color, symbol="circle", size=18, opacity=0.65),
                            name=f"{display_name} — Intersection",
                            legendgroup=display_name,
                            showlegend=False,
                            hovertemplate="Threshold: %{x:.2f}<br>Rate: %{y:.1%}<extra></extra>",
                        )
                    )

        fig.update_layout(
            xaxis_title="AUC cutoff (Precision-Recall tradeoff)",
            yaxis_title="Error-Correctness Rate",
            yaxis_tickformat=".0%",
            template="plotly_white",
            legend=LEGEND_BELOW.copy(),
            margin=STANDARD_MARGINS.copy(),
        )
        _append_style_hint(fig)

        config = {
            "toImageButtonOptions": {
                "format": "png",
                "filename": f"{dataset}_connected_scatter{('_' + filename_suffix) if filename_suffix else ''}",
                "width": 1920,
                "height": 1080,
                "scale": 3,
            },
        }
        filename = _apply_suffix(f"{dataset}_connected_scatter.html", filename_suffix)
        output_path = output_dir / filename
        fig.write_html(output_path, include_plotlyjs="cdn", config=config)
        created_files.append(output_path)

    return created_files


def build_combined_connected_scatter(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    specs: Sequence[dict] = DEFAULT_COMBINED_LAYOUTS,
    variant_label: Optional[str] = None,
    filename_suffix: str = "",
) -> list[Path]:
    if not specs:
        return []

    output_dir = _ensure_output_dir(output_dir)
    created_files: list[Path] = []

    for spec in specs:
        dataset_pairs: Sequence[tuple[str, str]] = spec.get("datasets", [])
        if not dataset_pairs:
            continue

        fig = make_subplots(
            rows=1,
            cols=len(dataset_pairs),
            shared_yaxes=True,
            horizontal_spacing=0.08,
            subplot_titles=[label for _, label in dataset_pairs],
        )

        palette = qualitative.Bold + qualitative.Safe + qualitative.Light24
        color_lookup: dict[str, str] = {}
        palette_cycle = cycle(palette)
        legend_rank = 0
        legend_methods_shown: set[str] = set()

        for col_index, (dataset_key, dataset_label) in enumerate(dataset_pairs, start=1):
            dataset_df = df[df["dataset"] == dataset_key]
            if dataset_df.empty:
                continue
            for (method, graph_type), method_df in dataset_df.groupby(
                ["method", "graph_type"], sort=True
            ):
                ordered = method_df.sort_values("threshold")
                if ordered.empty:
                    continue
                series_name = method
                if pd.notna(graph_type):
                    series_name = f"{method} ({graph_type})"
                if series_name not in color_lookup:
                    color_lookup[series_name] = next(palette_cycle)
                color = color_lookup[series_name]
                error_series = ordered["rate"]
                correct_series = ordered["correct_rate"]
                any_error = error_series.notna().any()
                any_correct = correct_series.notna().any()
                value_arrays: list[np.ndarray] = []
                if any_error:
                    value_arrays.append(error_series.to_numpy())
                if any_correct:
                    value_arrays.append(correct_series.to_numpy())
                if value_arrays:
                    concatenated = np.concatenate(value_arrays)
                    finite_values = concatenated[~np.isnan(concatenated)]
                else:
                    finite_values = np.array([])
                method_has_entry = series_name in legend_methods_shown
                if any_error:
                    showlegend = not method_has_entry
                    fig.add_trace(
                        go.Scatter(
                            x=ordered["threshold"],
                            y=error_series,
                            mode="lines",
                            name=series_name if showlegend else f"{series_name} — Error detection",
                            legendgroup=series_name,
                            legendrank=legend_rank if showlegend else None,
                            showlegend=showlegend,
                            line=dict(color=color, dash="solid"),
                            hovertemplate=(
                                f"Dataset: {dataset_label}<br>"
                                "Method: %{customdata[0]}<br>"
                                "Graph type: %{customdata[1]}<br>"
                                "Threshold: %{x:.2f}<br>"
                                "Error detection rate: %{y:.1%}<extra></extra>"
                            ),
                            customdata=ordered[["method", "graph_type"]].to_numpy(),
                        ),
                        row=1,
                        col=col_index,
                    )
                    if showlegend:
                        legend_methods_shown.add(series_name)
                        legend_rank += 1
                        method_has_entry = True
                if any_correct:
                    showlegend = not method_has_entry
                    fig.add_trace(
                        go.Scatter(
                            x=ordered["threshold"],
                            y=correct_series,
                            mode="lines",
                            name=series_name if showlegend else f"{series_name} — Correct retention",
                            legendgroup=series_name,
                            legendrank=legend_rank if showlegend else None,
                            showlegend=showlegend,
                            line=dict(color=color, dash="dash"),
                            hovertemplate=(
                                f"Dataset: {dataset_label}<br>"
                                "Method: %{customdata[0]}<br>"
                                "Graph type: %{customdata[1]}<br>"
                                "Threshold: %{x:.2f}<br>"
                                "Correct retention rate: %{y:.1%}<extra></extra>"
                            ),
                            customdata=ordered[["method", "graph_type"]].to_numpy(),
                        ),
                        row=1,
                        col=col_index,
                    )
                    if showlegend:
                        legend_methods_shown.add(series_name)
                        legend_rank += 1
                        method_has_entry = True
                if any_error and any_correct:
                    intersections = _compute_intersections(
                        ordered["threshold"].to_numpy(),
                        error_series.to_numpy(),
                        correct_series.to_numpy(),
                    )
                    if intersections:
                        xs, ys = zip(*intersections)
                        fig.add_trace(
                            go.Scatter(
                                x=xs,
                                y=ys,
                                mode="markers",
                                marker=dict(color=color, symbol="circle", size=18, opacity=0.65),
                                name=f"{series_name} — Intersection",
                                legendgroup=series_name,
                                showlegend=False,
                                hovertemplate=(
                                    f"Dataset: {dataset_label}<br>"
                                    "Threshold: %{x:.2f}<br>"
                                    "Rate: %{y:.1%}<extra></extra>"
                                ),
                            ),
                            row=1,
                            col=col_index,
                        )
            fig.update_xaxes(
                title_text="AUC cutoff (Precision-Recall tradeoff)", row=1, col=col_index, range=[0, 1]
            )
            fig.update_yaxes(
                title_text="Error-Correctness Rate", row=1, col=col_index, tickformat=".0%"
            )

        layout_height = spec.get("height", 450)
        layout_width = spec.get("width", 1200)
        fig.update_layout(
            template="plotly_white",
            legend=LEGEND_BELOW.copy(),
            height=layout_height,
            width=layout_width,
            margin=STANDARD_MARGINS.copy(),
        )
        _append_style_hint(fig)

        filename = spec.get("filename", "combined_connected_scatter.html")
        filename = _apply_suffix(filename, filename_suffix)
        base_filename = filename.removesuffix(".html")
        config = {
            "toImageButtonOptions": {
                "format": "png",
                "filename": base_filename,
                "width": layout_width,
                "height": layout_height,
                "scale": 2,
            }
        }
        output_path = output_dir / filename
        fig.write_html(output_path, include_plotlyjs="cdn", config=config)
        created_files.append(output_path)

    return created_files


def _generate_metric_exports(
    input_csv: Path,
    output_dir: Path,
    *,
    datasets: Optional[Sequence[str]] = None,
    methods: Optional[Sequence[str]] = None,
    graph_types: Optional[Sequence[str]] = None,
    variant_label: Optional[str] = None,
    filename_suffix: str = "",
) -> list[Path]:
    """Load detection data and build both standalone and combined connected-scatter plots."""
    detection_rates = load_detection_rates(input_csv)
    plots = build_connected_scatter(
        detection_rates,
        output_dir,
        datasets=datasets,
        methods=methods,
        graph_types=graph_types,
        variant_label=variant_label,
        filename_suffix=filename_suffix,
    )
    plots.extend(
        build_combined_connected_scatter(
            detection_rates,
            output_dir,
            variant_label=variant_label,
            filename_suffix=filename_suffix,
        )
    )
    return plots


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate connected scatter plots (threshold vs. detection rate) per dataset."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Path to detection_rates_by_dataset.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where Plotly HTML files will be written.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Optional list of datasets to include.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        help="Optional list of methods to include.",
    )
    parser.add_argument(
        "--graph-types",
        nargs="+",
        help="Optional list of graph types to include.",
    )
    parser.add_argument(
        "--variant-label",
        type=str,
        help="Optional label appended to titles to denote the underlying metric (e.g., 'Deletion AUC').",
    )
    parser.add_argument(
        "--filename-suffix",
        type=str,
        default="",
        help="Optional suffix appended to output filenames to distinguish different metrics.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    if raw_args:
        options = parse_args(raw_args)
        plots = _generate_metric_exports(
            options.input_csv,
            options.output_dir,
            datasets=options.datasets,
            methods=options.methods,
            graph_types=options.graph_types,
            variant_label=options.variant_label,
            filename_suffix=options.filename_suffix,
        )
        print("Generated connected scatter plots:")
        for path in plots:
            print(f" - {path}")
        return

    print(
        "No CLI arguments supplied; generating default deletion and insertion AUC connected-scatter plots..."
    )
    for batch in AUTOMATIC_EXPORT_BATCHES:
        label = batch["label"]
        print(f"\n[{label}]")
        plots = _generate_metric_exports(
            batch["input_csv"],
            batch["output_dir"],
            variant_label=batch.get("variant_label"),
            filename_suffix=batch.get("filename_suffix", ""),
        )
        for path in plots:
            print(f" - {path}")


if __name__ == "__main__":
    main()
