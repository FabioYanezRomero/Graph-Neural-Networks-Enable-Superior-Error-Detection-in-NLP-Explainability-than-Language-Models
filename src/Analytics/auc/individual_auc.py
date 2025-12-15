#!/usr/bin/env python3
"""Generate AUC visualisations for every raw per-graph CSV."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import pearsonr
from sklearn.metrics import auc, roc_curve

import warnings

warnings.filterwarnings("ignore")

DEFAULT_INPUT_ROOT = Path("outputs/analytics/auc")
PLOTS_SUBDIR = "plots"

REQUIRED_COLUMNS = {
    "method",
    "dataset",
    "graph_type",
    "label",
    "is_correct",
    "prediction_confidence",
    "deletion_auc",
    "insertion_auc",
    "normalised_deletion_auc",
}

METRIC_COLUMNS: Sequence[str] = (
    "deletion_auc",
    "insertion_auc",
    "normalised_deletion_auc",
)

CORRELATION_COLUMNS: Sequence[str] = (
    "deletion_auc",
    "insertion_auc",
    "normalised_deletion_auc",
    "prediction_confidence",
)

COLOR_MAP: Dict[str, str] = {
    "correct": "#06A77D",
    "incorrect": "#D62828",
    "deletion_auc": "#2E86AB",
    "insertion_auc": "#A23B72",
    "normalised_deletion_auc": "#F18F01",
}


def discover_auc_csvs(root: Path) -> List[Path]:
    """Return all method/dataset/graph CSV paths."""

    csvs: List[Path] = []
    for path in root.glob("**/*.csv"):
        try:
            parts = path.relative_to(root).parts
        except ValueError:
            continue
        if len(parts) == 3 and path.is_file():
            csvs.append(path)
    return sorted(csvs)


def ensure_required_columns(df: pd.DataFrame, csv_path: Path) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing columns {sorted(missing)}")


def normalize_correctness(value: object) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def dataset_slug(df: pd.DataFrame) -> str:
    for column in ("dataset_backbone", "dataset_raw", "dataset"):
        if column in df.columns:
            value = df[column].iloc[0]
            if isinstance(value, str) and value:
                return value.replace("/", "_")
    return "unknown_dataset"


class IndividualAUCVisualizer:
    """Container for generating the seven AUC plots for a single CSV."""

    def __init__(self, csv_path: Path, df: pd.DataFrame, output_dir: Path) -> None:
        self.csv_path = csv_path
        self.df = df.copy()
        self.output_dir = output_dir

        ensure_required_columns(self.df, csv_path)
        self.df["is_correct"] = self.df["is_correct"].apply(normalize_correctness)
        self.df = self.df[self.df["is_correct"].notna()].reset_index(drop=True)

        if self.df.empty:
            raise ValueError(f"{csv_path}: no valid rows after cleaning correctness column")

        self.method = str(self.df["method"].iloc[0])
        self.dataset = dataset_slug(self.df)
        self.graph = str(self.df["graph_type"].iloc[0])

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ helpers

    def _save(self, fig: go.Figure, stem: str) -> None:
        path = self.output_dir / stem
        fig.write_html(str(path))
        print(f"    ✓ {path.relative_to(self.output_dir.parent)}")

    def _has_two_classes(self, mask: np.ndarray) -> bool:
        y = mask.astype(int)
        positives = y.sum()
        negatives = len(y) - positives
        return positives > 0 and negatives > 0

    # ------------------------------------------------------------------ plots

    def plot_roc_overall(self) -> None:
        frame = self.df[["is_correct", "deletion_auc"]].dropna()
        if frame.empty or not self._has_two_classes(frame["is_correct"].values):
            print("    ! Skipping ROC overall (insufficient class diversity)")
            return

        y_true = frame["is_correct"].astype(int).values
        y_scores = frame["deletion_auc"].values
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        score = auc(fpr, tpr)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"Deletion AUC (AUC = {score:.4f})",
                line=dict(color=COLOR_MAP["deletion_auc"], width=3),
                fill="tozeroy",
                fillcolor="rgba(46, 134, 171, 0.18)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Classifier",
                line=dict(color="#7f8c8d", width=2, dash="dash"),
            )
        )
        fig.update_layout(
            title=(
                "<b>ROC: Deletion AUC vs Correctness</b><br>"
                f"<sub>{self.method} · {self.dataset} · {self.graph} (n={len(frame)})</sub>"
            ),
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            template="plotly_white",
            height=640,
            width=900,
            legend=dict(x=0.65, y=0.15),
        )
        self._save(fig, "01_roc_deletion_auc_overall.html")

    def plot_roc_multi_metrics(self) -> None:
        y_true = self.df["is_correct"].astype(int).values
        if not self._has_two_classes(y_true):
            print("    ! Skipping ROC comparison (insufficient class diversity)")
            return

        fig = go.Figure()
        added = False
        for column in METRIC_COLUMNS:
            frame = self.df[["is_correct", column]].dropna()
            if frame.empty:
                continue
            scores = frame[column].values
            labels = frame["is_correct"].astype(int).values
            if not self._has_two_classes(labels):
                continue

            # guard against extreme values that break ROC
            if column == "normalised_deletion_auc":
                scores = np.clip(scores, -10, 10)

            fpr, tpr, _ = roc_curve(labels, scores)
            score = auc(fpr, tpr)
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"{column.replace('_', ' ').title()} (AUC = {score:.4f})",
                    line=dict(color=COLOR_MAP.get(column, "#444"), width=3),
                )
            )
            added = True

        if not added:
            print("    ! Skipping ROC comparison (no metric produced a valid curve)")
            return

        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random (AUC = 0.5)",
                line=dict(color="#7f8c8d", width=2, dash="dash"),
            )
        )
        fig.update_layout(
            title=(
                "<b>ROC Comparison: AUC Metrics</b><br>"
                f"<sub>{self.method} · {self.dataset} · {self.graph}</sub>"
            ),
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            template="plotly_white",
            height=640,
            width=960,
            legend=dict(x=0.62, y=0.15),
        )
        self._save(fig, "02_roc_multi_metrics_comparison.html")

    def plot_scatter_deletion_vs_confidence(self) -> None:
        frame = self.df[["is_correct", "prediction_confidence", "deletion_auc"]].dropna()
        if frame.empty:
            print("    ! Skipping scatter (deletion vs confidence) due to missing data")
            return

        correct = frame[frame["is_correct"] == True]  # noqa: E712
        incorrect = frame[frame["is_correct"] == False]  # noqa: E712

        fig = go.Figure()
        if not correct.empty:
            fig.add_trace(
                go.Scatter(
                    x=correct["prediction_confidence"],
                    y=correct["deletion_auc"],
                    mode="markers",
                    name="Correct",
                    marker=dict(
                        size=6,
                        color=COLOR_MAP["correct"],
                        opacity=0.65,
                        line=dict(width=0.4, color="#146356"),
                    ),
                    hovertemplate="Conf: %{x:.3f}<br>Del AUC: %{y:.3f}<extra></extra>",
                )
            )

        if not incorrect.empty:
            fig.add_trace(
                go.Scatter(
                    x=incorrect["prediction_confidence"],
                    y=incorrect["deletion_auc"],
                    mode="markers",
                    name="Incorrect",
                    marker=dict(
                        size=8,
                        color=COLOR_MAP["incorrect"],
                        opacity=0.7,
                        symbol="diamond",
                        line=dict(width=0.4, color="#8e1c1c"),
                    ),
                    hovertemplate="Conf: %{x:.3f}<br>Del AUC: %{y:.3f}<extra></extra>",
                )
            )

        def add_trend(subframe: pd.DataFrame, color: str, label: str) -> float:
            if len(subframe) < 2 or subframe["prediction_confidence"].nunique() < 2:
                return math.nan
            try:
                coeffs = np.polyfit(
                    subframe["prediction_confidence"], subframe["deletion_auc"], 1
                )
                poly = np.poly1d(coeffs)
                x_coords = np.linspace(
                    subframe["prediction_confidence"].min(),
                    subframe["prediction_confidence"].max(),
                    100,
                )
                fig.add_trace(
                    go.Scatter(
                        x=x_coords,
                        y=poly(x_coords),
                        mode="lines",
                        name=f"Trend ({label})",
                        line=dict(color=color, width=2, dash="dash"),
                        hoverinfo="skip",
                    )
                )
            except (np.linalg.LinAlgError, ValueError):
                return math.nan

            try:
                corr, _ = pearsonr(
                    subframe["prediction_confidence"], subframe["deletion_auc"]
                )
            except ValueError:
                corr = math.nan
            return corr

        corr_correct = add_trend(correct, "#1b8f57", "Correct")
        corr_incorrect = add_trend(incorrect, "#b83c3c", "Incorrect")

        annotation = "<b>Correlation</b>"
        if not math.isnan(corr_correct):
            annotation += f"<br>Correct: {corr_correct:.4f}"
        if not math.isnan(corr_incorrect):
            annotation += f"<br>Incorrect: {corr_incorrect:.4f}"

        fig.update_layout(
            title=(
                "<b>Deletion AUC vs Prediction Confidence</b><br>"
                f"<sub>{self.method} · {self.dataset} · {self.graph}</sub>"
            ),
            xaxis_title="Prediction Confidence",
            yaxis_title="Deletion AUC",
            template="plotly_white",
            height=640,
            width=960,
            hovermode="closest",
        )
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.01,
            y=0.99,
            text=annotation,
            showarrow=False,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#555",
            borderwidth=1,
            align="left",
        )
        self._save(fig, "03_scatter_deletion_vs_confidence.html")

    def plot_pairwise_metrics(self) -> None:
        available = [col for col in METRIC_COLUMNS if col in self.df.columns]
        if not available:
            print("    ! Skipping pairwise metrics (no columns available)")
            return

        correct = self.df[self.df["is_correct"] == True]  # noqa: E712
        incorrect = self.df[self.df["is_correct"] == False]  # noqa: E712

        fig = make_subplots(
            rows=len(available),
            cols=len(available),
            subplot_titles=[
                f"{a.replace('_', ' ').title()} vs {b.replace('_', ' ').title()}"
                for a in available
                for b in available
            ],
            specs=[[{} for _ in available] for _ in available],
            horizontal_spacing=0.05,
            vertical_spacing=0.08,
        )

        for i, row_metric in enumerate(available, start=1):
            for j, col_metric in enumerate(available, start=1):
                if i == j:
                    fig.add_trace(
                        go.Histogram(
                            x=correct[row_metric],
                            name="Correct",
                            marker_color=COLOR_MAP["correct"],
                            opacity=0.65,
                            showlegend=(i == 1 and j == 1),
                        ),
                        row=i,
                        col=j,
                    )
                    fig.add_trace(
                        go.Histogram(
                            x=incorrect[row_metric],
                            name="Incorrect",
                            marker_color=COLOR_MAP["incorrect"],
                            opacity=0.65,
                            showlegend=(i == 1 and j == 1),
                        ),
                        row=i,
                        col=j,
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=correct[col_metric],
                            y=correct[row_metric],
                            mode="markers",
                            name="Correct",
                            marker=dict(size=4, color=COLOR_MAP["correct"], opacity=0.5),
                            showlegend=(i == 1 and j == 2),
                            hovertemplate="%{x:.3f}, %{y:.3f}<extra></extra>",
                        ),
                        row=i,
                        col=j,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=incorrect[col_metric],
                            y=incorrect[row_metric],
                            mode="markers",
                            name="Incorrect",
                            marker=dict(
                                size=5,
                                color=COLOR_MAP["incorrect"],
                                opacity=0.65,
                                symbol="diamond",
                            ),
                            showlegend=(i == 1 and j == 2),
                            hovertemplate="%{x:.3f}, %{y:.3f}<extra></extra>",
                        ),
                        row=i,
                        col=j,
                    )

                fig.update_xaxes(title_text=row_metric.replace("_", " ").title(), row=i, col=j)
                fig.update_yaxes(title_text=col_metric.replace("_", " ").title(), row=i, col=j)

        size = 360 * len(available)
        fig.update_layout(
            title=(
                "<b>Pairwise AUC Metric Relationships</b><br>"
                f"<sub>{self.method} · {self.dataset} · {self.graph}</sub>"
            ),
            template="plotly_white",
            height=max(720, size),
            width=max(720, size),
        )
        self._save(fig, "04_scatter_metrics_pairwise.html")

    def plot_correlation_overall(self) -> None:
        columns = [col for col in CORRELATION_COLUMNS if col in self.df.columns]
        frame = self.df[columns].copy()
        frame["is_correct"] = self.df["is_correct"].astype(int)
        if len(frame) < 2:
            print("    ! Skipping overall correlation (insufficient rows)")
            return

        corr = frame.corr()
        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=[c.replace("_", " ").title() for c in corr.columns],
                y=[c.replace("_", " ").title() for c in corr.index],
                colorscale="RdBu",
                zmid=0,
                zmin=-1,
                zmax=1,
                text=np.round(corr.values, 3),
                texttemplate="%{text:.2f}",
                colorbar=dict(title="Correlation"),
            )
        )
        fig.update_layout(
            title=(
                "<b>Correlation Matrix</b><br>"
                f"<sub>{self.method} · {self.dataset} · {self.graph}</sub>"
            ),
            template="plotly_white",
            height=600,
            width=780,
        )
        self._save(fig, "05_correlation_heatmap_overall.html")

    def plot_correlation_stratified(self) -> None:
        columns = [col for col in CORRELATION_COLUMNS if col in self.df.columns]
        correct = self.df[self.df["is_correct"] == True][columns]  # noqa: E712
        incorrect = self.df[self.df["is_correct"] == False][columns]  # noqa: E712

        if len(correct) < 2 and len(incorrect) < 2:
            print("    ! Skipping stratified correlation (no cohorts with >=2 rows)")
            return

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                f"Correct (n={len(correct)})",
                f"Incorrect (n={len(incorrect)})",
            ),
            specs=[[{"type": "heatmap"}, {"type": "heatmap"}]],
        )

        def add_heatmap(sub: pd.DataFrame, col_idx: int) -> None:
            if len(sub) < 2:
                fig.add_trace(
                    go.Heatmap(
                        z=[[0]],
                        x=["No data"],
                        y=["No data"],
                        showscale=False,
                        colorscale="Greys",
                        zmin=-1,
                        zmax=1,
                    ),
                    row=1,
                    col=col_idx,
                )
                return
            corr = sub.corr()
            fig.add_trace(
                go.Heatmap(
                    z=corr.values,
                    x=[c.replace("_", " ").title() for c in corr.columns],
                    y=[c.replace("_", " ").title() for c in corr.index],
                    colorscale="RdBu",
                    zmid=0,
                    zmin=-1,
                    zmax=1,
                    text=np.round(corr.values, 3),
                    texttemplate="%{text:.2f}",
                    colorbar=dict(title="Corr.", len=0.8),
                ),
                row=1,
                col=col_idx,
            )

        add_heatmap(correct, 1)
        add_heatmap(incorrect, 2)
        fig.update_layout(
            title=(
                "<b>Correlation by Correctness</b><br>"
                f"<sub>{self.method} · {self.dataset} · {self.graph}</sub>"
            ),
            template="plotly_white",
            height=540,
            width=960,
        )
        self._save(fig, "06_correlation_heatmap_stratified.html")

    def plot_box_swarm_deletion(self) -> None:
        frame = self.df[["is_correct", "deletion_auc"]].dropna()
        correct = frame[frame["is_correct"] == True]  # noqa: E712
        incorrect = frame[frame["is_correct"] == False]  # noqa: E712

        if frame.empty:
            print("    ! Skipping box+swarm (no deletion AUC data)")
            return

        fig = go.Figure()
        if not correct.empty:
            fig.add_trace(
                go.Box(
                    y=correct["deletion_auc"],
                    name="Correct",
                    marker_color=COLOR_MAP["correct"],
                    boxmean="sd",
                    opacity=0.7,
                )
            )
            jitter = np.random.normal(-0.12, 0.035, size=len(correct))
            fig.add_trace(
                go.Scatter(
                    x=jitter,
                    y=correct["deletion_auc"],
                    mode="markers",
                    name="Correct (points)",
                    marker=dict(size=4, color=COLOR_MAP["correct"], opacity=0.32),
                    showlegend=False,
                    hovertemplate="Deletion AUC: %{y:.4f}<extra></extra>",
                )
            )

        if not incorrect.empty:
            fig.add_trace(
                go.Box(
                    y=incorrect["deletion_auc"],
                    name="Incorrect",
                    marker_color=COLOR_MAP["incorrect"],
                    boxmean="sd",
                    opacity=0.7,
                )
            )
            jitter = np.random.normal(1.12, 0.035, size=len(incorrect))
            fig.add_trace(
                go.Scatter(
                    x=jitter,
                    y=incorrect["deletion_auc"],
                    mode="markers",
                    name="Incorrect (points)",
                    marker=dict(
                        size=5,
                        color=COLOR_MAP["incorrect"],
                        opacity=0.4,
                        symbol="diamond",
                    ),
                    showlegend=False,
                    hovertemplate="Deletion AUC: %{y:.4f}<extra></extra>",
                )
            )

        def describe(values: pd.Series) -> str:
            if values.empty:
                return "n/a"
            return f"μ={values.mean():.4f}, σ={values.std(ddof=0):.4f}"

        stats = "<b>Deletion AUC</b>"
        stats += f"<br>Correct: {describe(correct['deletion_auc'])}"
        stats += f"<br>Incorrect: {describe(incorrect['deletion_auc'])}"

        fig.update_layout(
            title=(
                "<b>Deletion AUC Distribution</b><br>"
                f"<sub>{self.method} · {self.dataset} · {self.graph}</sub>"
            ),
            yaxis_title="Deletion AUC",
            xaxis=dict(showticklabels=False),
            template="plotly_white",
            height=640,
            width=820,
        )
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.98,
            y=0.02,
            text=stats,
            showarrow=False,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#555",
            borderwidth=1,
            align="right",
        )
        self._save(fig, "07_box_swarm_deletion_auc.html")

    # ------------------------------------------------------------------ public

    def run(self) -> None:
        print(
            f"  • {self.method}/{self.dataset}/{self.graph} "
            f"(records={len(self.df)})"
        )
        self.plot_roc_overall()
        self.plot_roc_multi_metrics()
        self.plot_scatter_deletion_vs_confidence()
        self.plot_pairwise_metrics()
        self.plot_correlation_overall()
        self.plot_correlation_stratified()
        self.plot_box_swarm_deletion()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate individual-level AUC plots for each raw CSV."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_ROOT),
        help="Root directory with method/dataset/graph CSVs (default: %(default)s)",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional path to a single CSV to process.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.csv:
        csv_paths = [Path(args.csv)]
    else:
        csv_paths = discover_auc_csvs(Path(args.input))

    if not csv_paths:
        raise FileNotFoundError("No AUC CSV files were found.")

    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"! Skipping {csv_path}: failed to load ({exc})")
            continue

        try:
            graph_dir = csv_path.parent / csv_path.stem
            plots_dir = graph_dir / PLOTS_SUBDIR
            viz = IndividualAUCVisualizer(csv_path, df, plots_dir)
            viz.run()
        except Exception as exc:
            print(f"! Failed to generate plots for {csv_path}: {exc}")


if __name__ == "__main__":
    main()
