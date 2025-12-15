#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from importance_accumulation_plot import (
    DEFAULT_DATASET_LABELS,
    DEFAULT_GRAPH_LABELS,
    DEFAULT_GRAPH_TYPE_ORDER,
    DEFAULT_METHOD_LABELS,
    DEFAULT_METHOD_ORDER,
    DEFAULT_TOPK_METRICS,
    FIELD_LABELS,
    METHOD_GRAPH_COLORS,
    SUMMARY_FILENAME,
    TOPK_LABELS,
)

DEFAULT_ROOT = Path("outputs/analytics/progression")
DEFAULT_OUTPUT = Path(
    "outputs/analytics/progression/combined_plots/topk_progression_clipped_grid.html"
)


SUBPLOT_SPEC = [
    {
        "field": "maskout_progression_drop",
        "dataset": "setfit_ag_news",
        "row": 1,
        "col": 1,
    },
    {
        "field": "maskout_progression_drop",
        "dataset": "stanfordnlp_sst2",
        "row": 1,
        "col": 2,
    },
    {
        "field": "sufficiency_progression_drop",
        "dataset": "setfit_ag_news",
        "row": 2,
        "col": 1,
    },
    {
        "field": "sufficiency_progression_drop",
        "dataset": "stanfordnlp_sst2",
        "row": 2,
        "col": 2,
    },
]


def _load_field_dataframe(root: Path, field: str) -> pd.DataFrame:
    csv_path = root / field / SUMMARY_FILENAME
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing concentration summary: {csv_path}")
    df = pd.read_csv(csv_path)
    focus = df[
        (df["metric"].isin(DEFAULT_TOPK_METRICS))
        & (df["group"] == "overall")
    ].copy()
    return focus


def _series_for_dataset(df: pd.DataFrame, dataset: str) -> List[Dict[str, object]]:
    dataset_df = df[df["dataset"] == dataset]
    if dataset_df.empty:
        return []

    series_entries: List[Dict[str, object]] = []
    for method in DEFAULT_METHOD_ORDER:
        method_slice = dataset_df[dataset_df["method"] == method]
        if method_slice.empty:
            continue
        graph_order = DEFAULT_GRAPH_TYPE_ORDER.get(
            method,
            tuple(sorted(method_slice["graph"].unique())),
        )
        for graph in graph_order:
            graph_slice = method_slice[method_slice["graph"] == graph]
            if graph_slice.empty:
                continue
            values = []
            hover = []
            for metric in DEFAULT_TOPK_METRICS:
                metric_row = graph_slice[graph_slice["metric"] == metric]
                if metric_row.empty:
                    values.append(np.nan)
                    hover.append(f"Top-k: {TOPK_LABELS.get(metric, metric)}<br>Value: N/A")
                    continue
                raw_mean = float(metric_row["mean"].iloc[0])
                clipped = float(np.clip(raw_mean, 0.0, 1.0))
                values.append(clipped)
                hover.append(
                    f"Top-k: {TOPK_LABELS.get(metric, metric)}<br>"
                    f"Clipped mean: {clipped:.3f}"
                )
            method_label = DEFAULT_METHOD_LABELS.get(method, method)
            graph_label = DEFAULT_GRAPH_LABELS.get(graph, graph)
            name = f"{method_label} — {graph_label}"
            color = METHOD_GRAPH_COLORS.get(
                (method, graph),
                "#636EFA",
            )
            series_entries.append(
                {
                    "name": name,
                    "color": color,
                    "values": values,
                    "hover": hover,
                }
            )
    return series_entries


def build_combined_figure(root: Path, output_path: Path) -> Path:
    field_frames: Dict[str, pd.DataFrame] = {}
    for spec in SUBPLOT_SPEC:
        field = spec["field"]
        if field in field_frames:
            continue
        field_frames[field] = _load_field_dataframe(root, field)

    fig = make_subplots(
        rows=2,
        cols=2,
        shared_yaxes=True,
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
        subplot_titles=[
            (
                "<b>"
                + f"{FIELD_LABELS.get(spec['field'], spec['field'])} — "
                + f"{DEFAULT_DATASET_LABELS.get(spec['dataset'], spec['dataset'])}"
                + "</b>"
            )
            for spec in SUBPLOT_SPEC
        ],
        specs=[[{}, {}], [{}, {}]],
    )

    x_positions = list(range(len(DEFAULT_TOPK_METRICS)))
    x_labels = [f"<b>{TOPK_LABELS[m]}</b>" for m in DEFAULT_TOPK_METRICS]

    for idx, spec in enumerate(SUBPLOT_SPEC):
        field_df = field_frames[spec["field"]]
        dataset = spec["dataset"]
        dataset_label = DEFAULT_DATASET_LABELS.get(dataset, dataset)
        series_entries = _series_for_dataset(field_df, dataset)
        if not series_entries:
            continue

        for series in series_entries:
            fig.add_trace(
                go.Scatter(
                    x=x_positions,
                    y=series["values"],
                    mode="lines+markers",
                    name=series["name"],
                    legendgroup=series["name"],
                    showlegend=(idx == 0),
                    line=dict(color=series["color"], width=2),
                    marker=dict(color=series["color"], size=7),
                    hovertemplate=(
                        f"Dataset: {dataset_label}<br>"
                        "%{text}<extra></extra>"
                    ),
                    text=series["hover"],
                ),
                row=spec["row"],
                col=spec["col"],
            )

        fig.update_xaxes(
            tickmode="array",
            tickvals=x_positions,
            ticktext=x_labels,
            tickangle=-45,
            title="<b>Top-k concentration</b>" if spec["row"] == 2 else "",
            title_standoff=10,
            row=spec["row"],
            col=spec["col"],
        )
        fig.update_yaxes(
            title="<b>Clipped mean concentration</b>",
            range=[0, 1],
            row=spec["row"],
            col=spec["col"],
        )

    fig.update_layout(
        title=dict(
            text="<b>Top-k Concentration Comparison — Maskout vs. Sufficiency Drops (Clipped)</b>",
            x=0.5,
            xanchor="center",
        ),
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.08,
            xanchor="center",
            x=0.5,
            title=dict(text=""),
            font=dict(size=10),
        ),
        height=650,
        width=1100,
        margin=dict(t=90, l=70, r=40, b=110),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "toImageButtonOptions": {
            "format": "png",
            "filename": output_path.stem,
            "width": 1100,
            "height": 650,
            "scale": 2,
        }
    }
    fig.write_html(str(output_path), include_plotlyjs="cdn", config=config)
    return output_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build combined top-k progression grid (clipped) for maskout/sufficiency drops."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root directory containing progression field folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write the combined HTML figure.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    output = build_combined_figure(args.root, args.output)
    print(f"Wrote combined progression grid to: {output}")


if __name__ == "__main__":
    main()
