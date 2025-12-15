"\"\"\"Visualisations for confidence and threshold metrics.\"\"\""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DEFAULT_OUTPUT_ROOT = Path("outputs/analytics/confidence")
SUMMARY_COLUMNS = [
    "prediction_confidence",
    "median_threshold",
    "explanation_size",
    "unique_token_count",
    "semantic_density",
    "sparsity",
    "masked_confidence",
    "maskout_confidence",
]


def _iter_summary_csvs(root: Path, pattern: str) -> Iterable[Path]:
    return sorted(root.rglob(pattern))


def _plot_confidence_threshold(
    frame: pd.DataFrame,
    title: str,
    output_path: Path,
    *,
    hue_col: Optional[str] = None,
    dpi: int = 300,
) -> Path:
    data = frame[["median_threshold", "prediction_confidence"] + ([hue_col] if hue_col else [])].dropna()
    if data.empty:
        return output_path

    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(6, 4))
    kwargs = dict(x="median_threshold", y="prediction_confidence", alpha=0.65, s=36, ax=ax)
    if hue_col and hue_col in data.columns and data[hue_col].notna().any():
        sns.scatterplot(data=data, hue=data[hue_col].astype(str), **kwargs)
        ax.legend(title=hue_col, bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        sns.scatterplot(data=data, color="#4C72B0", **kwargs)

    ax.set_xlabel("Median threshold")
    ax.set_ylabel("Prediction confidence")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_correlation(frame: pd.DataFrame, columns: List[str], title: str, output_path: Path, *, dpi: int = 300) -> Path:
    available = [col for col in columns if col in frame.columns]
    if len(available) < 2:
        return output_path
    data = frame[available].apply(pd.to_numeric, errors="coerce").dropna()
    if data.empty:
        return output_path

    corr = data.corr()
    if corr.empty:
        return output_path

    sns.set_theme(style="white", context="paper")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, cmap="coolwarm", vmin=-1, vmax=1, center=0, annot=True, fmt=".2f", linewidths=0.4, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_confidence_threshold_visuals(
    summary_root: Path | str,
    output_root: Path | str | None = DEFAULT_OUTPUT_ROOT,
    *,
    pattern: str = "*summary.csv",
    dpi: int = 300,
) -> List[Path]:
    root_dir = Path(summary_root)
    out_dir = Path(output_root) if output_root else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    produced: List[Path] = []
    for csv_path in _iter_summary_csvs(root_dir, pattern):
        frame = pd.read_csv(csv_path)
        if "prediction_confidence" not in frame.columns or "median_threshold" not in frame.columns:
            continue

        target_dir = csv_path.parent if out_dir is None else (out_dir / csv_path.relative_to(root_dir).parent)
        target_dir.mkdir(parents=True, exist_ok=True)

        produced.append(
            _plot_confidence_threshold(
                frame,
                f"Confidence vs Threshold - {csv_path.stem}",
                target_dir / f"{csv_path.stem}_confidence_vs_threshold.png",
                hue_col="label" if "label" in frame.columns else None,
                dpi=dpi,
            )
        )

        groups = frame.groupby("label") if "label" in frame.columns else []
        for label, subset in groups:
            if subset.empty:
                continue
            output_path = target_dir / f"{csv_path.stem}_label_{label}_confidence_vs_threshold.png"
            produced.append(
                _plot_confidence_threshold(
                    subset,
                    f"Confidence vs Threshold - label {label}",
                    output_path,
                    hue_col=None,
                    dpi=dpi,
                )
            )

        produced.append(
            _plot_correlation(
                frame,
                SUMMARY_COLUMNS,
                f"Confidence correlations - {csv_path.stem}",
                target_dir / f"{csv_path.stem}_confidence_correlation.png",
                dpi=dpi,
            )
        )

    return [path for path in produced if path is not None]
