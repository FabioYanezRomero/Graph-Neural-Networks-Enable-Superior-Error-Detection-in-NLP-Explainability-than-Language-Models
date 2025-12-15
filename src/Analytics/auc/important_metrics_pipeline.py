"""
Generate the key AUC metrics and plots highlighted in
`outputs/analytics/auc/important_metrics and plots.md`.

The script consolidates per-instance explainability CSVs for the supported
methods (GraphSVX, SubgraphX, TokenSHAP LLM), computes the headline metrics,
and emits the corresponding visualisations used in the paper.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DEFAULT_INPUT_ROOT = Path("outputs/analytics/auc")
DEFAULT_OUTPUT_ROOT = Path("outputs/analytics/auc/plots/final_metrics")
DEFAULT_THRESHOLDS: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


BOOLEAN_MAP = {"true": True, "false": False}

PALETTE = {
    "graphsvx": "#2060FF",
    "subgraphx": "#4E89FF",
    "token_shap_llm": "#FF4B4B",
}

CORRECTNESS_LABELS = {True: "Correct", False: "Incorrect"}
CORRECTNESS_COLORS_BOOL = {True: "#00CC96", False: "#EF553B"}
CORRECTNESS_PALETTE = {
    label: CORRECTNESS_COLORS_BOOL[key] for key, label in CORRECTNESS_LABELS.items()
}


@dataclass(frozen=True)
class DetectionRecord:
    method: str
    dataset: str
    graph_type: str
    threshold: float
    mispredictions: int
    detections: int
    correct_predictions: int
    correct_retained: int

    @property
    def rate(self) -> float:
        if self.mispredictions == 0:
            return float("nan")
        return self.detections / self.mispredictions

    @property
    def correct_rate(self) -> float:
        if self.correct_predictions == 0:
            return float("nan")
        return self.correct_retained / self.correct_predictions


def _discover_instance_csvs(root: Path) -> List[Path]:
    """Locate per-instance analytics CSV files."""
    candidates: List[Path] = []
    for csv_path in root.rglob("*.csv"):
        try:
            rel_parts = csv_path.relative_to(root).parts
        except ValueError:
            continue
        if len(rel_parts) == 3:
            candidates.append(csv_path)
    return sorted(candidates)


def _normalise_filters(values: Optional[Iterable[str]]) -> Optional[set[str]]:
    if values is None:
        return None
    filtered = {str(v).strip().lower() for v in values if str(v).strip()}
    return filtered or None


def load_per_instance_data(
    input_root: Path,
    *,
    methods: Optional[Sequence[str]] = None,
    datasets: Optional[Sequence[str]] = None,
    graph_types: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Load and concatenate per-instance explainability CSV files."""
    root = Path(input_root)
    if not root.exists():
        raise FileNotFoundError(f"Input root not found: {root}")

    method_filter = _normalise_filters(methods)
    dataset_filter = _normalise_filters(datasets)
    graph_filter = _normalise_filters(graph_types)

    frames: List[pd.DataFrame] = []
    for csv_path in _discover_instance_csvs(root):
        df = pd.read_csv(csv_path)

        if method_filter is not None and "method" in df.columns:
            df = df[df["method"].str.lower().isin(method_filter)]
        if dataset_filter is not None and "dataset" in df.columns:
            df = df[df["dataset"].str.lower().isin(dataset_filter)]
        if graph_filter is not None and "graph_type" in df.columns:
            df = df[df["graph_type"].str.lower().isin(graph_filter)]

        if df.empty:
            continue

        df["source_csv"] = str(csv_path)
        frames.append(df)

    if not frames:
        raise ValueError("No CSV files matched the requested filters.")

    data = pd.concat(frames, ignore_index=True)
    data["is_correct"] = (
        data["is_correct"]
        .astype(str)
        .str.lower()
        .map(BOOLEAN_MAP)
    )
    data["del_ins_spread"] = data["deletion_auc"] - data["insertion_auc"]
    return data


def compute_detection_records(
    df: pd.DataFrame,
    thresholds: Sequence[float],
    metric_column: str,
) -> List[DetectionRecord]:
    if metric_column not in df.columns:
        raise ValueError(f"Requested detection metric '{metric_column}' is missing.")
    records: List[DetectionRecord] = []

    for (method, dataset, graph_type), group in df.groupby(
        ["method", "dataset", "graph_type"], dropna=False
    ):
        mispred = group[group["is_correct"] == False]  # noqa: E712
        correct = group[group["is_correct"] == True]  # noqa: E712
        mispredictions = len(mispred)
        correct_predictions = len(correct)
        for threshold in thresholds:
            detections = int((mispred[metric_column] < threshold).sum()) if mispredictions else 0
            retained = (
                int((correct[metric_column] >= threshold).sum()) if correct_predictions else 0
            )
            records.append(
                DetectionRecord(
                    method=str(method),
                    dataset=str(dataset),
                    graph_type=str(graph_type),
                    threshold=float(threshold),
                    mispredictions=mispredictions,
                    detections=detections,
                    correct_predictions=correct_predictions,
                    correct_retained=retained,
                )
            )
    return records


def summarise_detection(records: Sequence[DetectionRecord]) -> pd.DataFrame:
    rows = []
    for rec in records:
        rows.append(
            {
                "method": rec.method,
                "dataset": rec.dataset,
                "graph_type": rec.graph_type,
                "threshold": rec.threshold,
                "mispredictions": rec.mispredictions,
                "detections": rec.detections,
                "rate": rec.rate,
                "correct_predictions": rec.correct_predictions,
                "correct_retained": rec.correct_retained,
                "correct_rate": rec.correct_rate,
            }
        )
    df = pd.DataFrame(rows)

    grouped = (
        df.groupby(["method", "threshold"])
        .agg(
            rate_mean=("rate", "mean"),
            rate_std=("rate", "std"),
            weighted_rate=("detections", "sum"),
            weighted_total=("mispredictions", "sum"),
             correct_rate_mean=("correct_rate", "mean"),
             correct_rate_std=("correct_rate", "std"),
             weighted_correct_retained=("correct_retained", "sum"),
             weighted_correct_total=("correct_predictions", "sum"),
            groups=("rate", "size"),
        )
        .reset_index()
    )
    grouped["weighted_rate"] = grouped["weighted_rate"] / grouped["weighted_total"].where(
        grouped["weighted_total"] != 0, np.nan
    )
    grouped["weighted_correct_rate"] = grouped["weighted_correct_retained"] / grouped[
        "weighted_correct_total"
    ].where(grouped["weighted_correct_total"] != 0, np.nan)
    return df, grouped


def compute_auc_separation(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, dataset, graph_type), group in df.groupby(
        ["method", "dataset", "graph_type"], dropna=False
    ):
        correct = group[group["is_correct"] == True]["deletion_auc"]  # noqa: E712
        incorrect = group[group["is_correct"] == False]["deletion_auc"]  # noqa: E712
        if correct.empty or incorrect.empty:
            continue
        separation = correct.mean() - incorrect.mean()
        rows.append(
            {
                "method": method,
                "dataset": dataset,
                "graph_type": graph_type,
                "correct_mean": correct.mean(),
                "incorrect_mean": incorrect.mean(),
                "separation": separation,
            }
        )
    combo = pd.DataFrame(rows)
    summary = (
        combo.groupby("method")
        .agg(
            correct_mean=("correct_mean", "mean"),
            incorrect_mean=("incorrect_mean", "mean"),
            separation_mean=("separation", "mean"),
            separation_std=("separation", "std"),
        )
        .reset_index()
    )
    return combo, summary


def compute_divergence_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in df.groupby("method"):
        correct = group[group["is_correct"] == True]["del_ins_spread"]  # noqa: E712
        incorrect = group[group["is_correct"] == False]["del_ins_spread"]  # noqa: E712
        if correct.empty or incorrect.empty:
            continue
        rows.append(
            {
                "method": method,
                "divergence_correct_mean": correct.mean(),
                "divergence_incorrect_mean": incorrect.mean(),
                "divergence_gap": correct.mean() - incorrect.mean(),
            }
        )
    return pd.DataFrame(rows)


def _ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _format_percentage(value: float) -> str:
    if np.isnan(value):
        return "N/A"
    return f"{value * 100:.1f}%"


def create_error_detection_bar_plot(
    summary: pd.DataFrame,
    output_dir: Path,
    threshold: float,
    metric_label: str,
) -> Path:
    plot_data = summary[summary["threshold"] == threshold].copy()
    plot_data["rate_std"] = plot_data["rate_std"].fillna(0.0)
    plot_data.sort_values("weighted_rate", ascending=False, inplace=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        plot_data["method"],
        plot_data["weighted_rate"] * 100,
        yerr=plot_data["rate_std"] * 100,
        color=[PALETTE.get(m, "#888888") for m in plot_data["method"]],
        capsize=6,
    )
    ax.set_ylabel("Error detection rate (%)")
    ax.set_title(f"Error Detection at {metric_label} < {threshold}")
    ax.set_ylim(0, 100)

    for bar, value in zip(bars, plot_data["weighted_rate"]):
        ax.annotate(
            _format_percentage(value),
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    sns.despine(ax=ax, left=True)

    output_path = output_dir / f"error_detection_bar_auc_{threshold:.1f}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def create_auc_separation_boxplots(
    df: pd.DataFrame,
    output_dir: Path,
    methods: Sequence[str],
) -> Path:
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, len(methods), figsize=(12, 4), sharey=True)
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        subset = df[df["method"] == method].copy()
        subset["correctness"] = subset["is_correct"].map(CORRECTNESS_LABELS)
        sns.boxplot(
            data=subset,
            x="correctness",
            y="deletion_auc",
            hue="correctness",
            dodge=False,
            palette=CORRECTNESS_PALETTE,
            ax=ax,
        )
        sns.stripplot(
            data=subset,
            x="correctness",
            y="deletion_auc",
            hue="correctness",
            dodge=False,
            alpha=0.25,
            palette=CORRECTNESS_PALETTE,
            ax=ax,
        )
        ax.set_title(method.replace("_", " ").title())
        ax.set_xlabel("")
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

        correct_mean = subset[subset["is_correct"] == True]["deletion_auc"].mean()  # noqa: E712
        incorrect_mean = subset[subset["is_correct"] == False]["deletion_auc"].mean()  # noqa: E712
        separation = correct_mean - incorrect_mean
        ax.annotate(
            f"Δ = {separation:.3f}",
            xy=(0.5, 0.05),
            xycoords="axes fraction",
            ha="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#444444", alpha=0.8),
        )

    axes[0].set_ylabel("Deletion AUC")
    fig.suptitle("Deletion AUC Separation by Prediction Outcome", y=1.02)

    output_path = output_dir / "auc_separation_boxplots.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def create_divergence_scatter(
    df: pd.DataFrame,
    output_dir: Path,
    methods: Sequence[str],
    sample_size: int = 5000,
) -> Path:
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, len(methods), figsize=(12, 4), sharex=True, sharey=True)
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        subset = df[df["method"] == method].copy()
        if sample_size and len(subset) > sample_size:
            subset = subset.sample(sample_size, random_state=42)
        subset["correctness"] = subset["is_correct"].map(CORRECTNESS_LABELS)
        sns.scatterplot(
            data=subset,
            x="deletion_auc",
            y="del_ins_spread",
            hue="correctness",
            palette=CORRECTNESS_PALETTE,
            alpha=0.4,
            edgecolor="none",
            ax=ax,
        )
        ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1)
        ax.set_title(method.replace("_", " ").title())
        ax.set_xlabel("Deletion AUC")
        ax.set_ylabel("Deletion - Insertion AUC")
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Divergence vs. Deletion AUC", y=1.02)

    output_path = output_dir / "divergence_scatter.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def create_detection_heatmap(
    df: pd.DataFrame,
    output_dir: Path,
    threshold: float,
    metric_label: str,
    metric_column: str,
) -> Path:
    mispred = df[df["is_correct"] == False]  # noqa: E712
    rows = []
    for (method, dataset), group in mispred.groupby(["method", "dataset"]):
        total = len(group)
        detections = int((group[metric_column] < threshold).sum())
        rate = detections / total if total else float("nan")
        rows.append(
            {
                "method": method,
                "dataset": dataset,
                "rate": rate,
            }
        )
    heatmap_df = pd.pivot_table(
        pd.DataFrame(rows),
        values="rate",
        index="method",
        columns="dataset",
    )

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    sns.heatmap(
        heatmap_df * 100,
        annot=heatmap_df.map(_format_percentage),
        fmt="",
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Detection rate (%)"},
        ax=ax,
    )
    ax.set_title(f"Detection Rate Heatmap ({metric_label} < {threshold})")

    output_path = output_dir / f"detection_heatmap_auc_{threshold:.1f}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_summary_markdown(
    detection_summary: pd.DataFrame,
    separation_summary: pd.DataFrame,
    divergence_summary: pd.DataFrame,
    *,
    metric_label: str,
) -> str:
    lines = [
        "# Important AUC Metrics",
        "",
        f"_Detection metric_: {metric_label}",
        "",
        "## Error Detection Rates",
    ]
    for _, row in detection_summary.sort_values(["threshold", "method"]).iterrows():
        std_pp = "n/a"
        if not np.isnan(row["rate_std"]):
            std_pp = f"{row['rate_std'] * 100:.1f} pp"
        lines.append(
            f"- `{row['method']}` at {metric_label} < {row['threshold']:.1f}: "
            f"{_format_percentage(row['weighted_rate'])} "
            f"(mean={_format_percentage(row['rate_mean'])}, std={std_pp}, n={int(row['groups'])})"
        )

    lines.extend(
        [
            "",
            "## AUC Separation (mean correct − mean incorrect)",
        ]
    )
    for _, row in separation_summary.sort_values("separation_mean", ascending=False).iterrows():
        lines.append(
            f"- `{row['method']}`: Δ={row['separation_mean']:.3f} "
            f"(correct={row['correct_mean']:.3f}, incorrect={row['incorrect_mean']:.3f})"
        )

    lines.extend(
        [
            "",
            "## Deletion–Insertion Divergence Gap",
        ]
    )
    for _, row in divergence_summary.sort_values("divergence_gap", ascending=False).iterrows():
        lines.append(
            f"- `{row['method']}`: Δ={row['divergence_gap']:.3f} "
            f"(correct={row['divergence_correct_mean']:.3f}, incorrect={row['divergence_incorrect_mean']:.3f})"
        )

    return "\n".join(lines) + "\n"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the headline AUC metrics and plots for the paper."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root directory that contains the per-instance explainability CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where metrics and plots will be written.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        help="Optional subset of methods to include.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Optional subset of datasets to include.",
    )
    parser.add_argument(
        "--graph-types",
        nargs="+",
        help="Optional subset of graph types to include.",
    )
    parser.add_argument(
        "--detection-metric",
        type=str,
        default="deletion_auc",
        help="Column name to threshold for error detection (default: deletion_auc).",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_THRESHOLDS),
        help="Detection thresholds to evaluate (default: 0.6 0.7).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Number of points per method for the divergence scatter plot.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    options = parse_args(argv)

    sns.set_theme(style="whitegrid")
    output_dir = _ensure_output_dir(options.output_dir)

    # 1. Load data.
    data = load_per_instance_data(
        options.input_root,
        methods=options.methods,
        datasets=options.datasets,
        graph_types=options.graph_types,
    )
    metric_column = options.detection_metric
    if metric_column not in data.columns:
        raise ValueError(
            f"Requested detection metric '{metric_column}' not present in loaded data."
        )
    metric_label = metric_column.replace("_", " ").title()

    # 2. Compute metrics.
    detection_records = compute_detection_records(data, options.thresholds, metric_column)
    detection_details, detection_summary = summarise_detection(detection_records)
    separation_details, separation_summary = compute_auc_separation(data)
    divergence_summary = compute_divergence_summary(data)

    # 3. Persist metric tables.
    detection_details.to_csv(output_dir / "detection_rates_by_dataset.csv", index=False)
    detection_summary.to_csv(output_dir / "detection_rates_summary.csv", index=False)
    separation_details.to_csv(output_dir / "auc_separation_by_dataset.csv", index=False)
    separation_summary.to_csv(output_dir / "auc_separation_summary.csv", index=False)
    divergence_summary.to_csv(output_dir / "divergence_summary.csv", index=False)

    summary_markdown = build_summary_markdown(
        detection_summary,
        separation_summary,
        divergence_summary,
        metric_label=metric_label,
    )
    (output_dir / "metrics_summary.md").write_text(summary_markdown, encoding="utf-8")

    # 4. Generate plots.
    created_plots = []

    for threshold in options.thresholds:
        created_plots.append(
            create_error_detection_bar_plot(
                detection_summary,
                output_dir,
                threshold=threshold,
                metric_label=metric_label,
            )
        )

    created_plots.append(
        create_auc_separation_boxplots(
            data,
            output_dir,
            methods=["graphsvx", "token_shap_llm"],
        )
    )
    created_plots.append(
        create_divergence_scatter(
            data,
            output_dir,
            methods=["graphsvx", "token_shap_llm"],
            sample_size=options.sample_size,
        )
    )

    for threshold in options.thresholds:
        created_plots.append(
            create_detection_heatmap(
                data,
                output_dir,
                threshold=threshold,
                metric_label=metric_label,
                metric_column=metric_column,
            )
        )

    print("Generated metric tables:")
    for csv_path in [
        output_dir / "detection_rates_by_dataset.csv",
        output_dir / "detection_rates_summary.csv",
        output_dir / "auc_separation_by_dataset.csv",
        output_dir / "auc_separation_summary.csv",
        output_dir / "divergence_summary.csv",
        output_dir / "metrics_summary.md",
    ]:
        print(f" - {csv_path}")

    print("\nGenerated plots:")
    for plot in created_plots:
        print(f" - {plot}")


if __name__ == "__main__":
    main()
