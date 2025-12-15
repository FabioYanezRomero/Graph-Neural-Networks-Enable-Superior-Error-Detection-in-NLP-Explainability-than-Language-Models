"\"\"\"Helpers to create heatmap visualisations from analytics artefacts.\"\"\""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import leaves_list, linkage


def _load_correlation_matrix(csv_path: Path) -> pd.DataFrame:
    """Load a correlation matrix stored on disk, normalising indices and headers."""
    frame = pd.read_csv(csv_path, index_col=0)
    if frame.empty:
        raise ValueError(f"Correlation matrix at {csv_path} is empty.")
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    return frame


_GROUP_SUFFIXES: Tuple[str, ...] = ("correct", "incorrect")


def _split_group(stem: str) -> Tuple[str, Optional[str]]:
    """Split a correlation filename stem into its base identifier and group."""
    for suffix in _GROUP_SUFFIXES:
        token = f"_{suffix}"
        if stem.endswith(token):
            return stem[: -len(token)], suffix
    return stem, None


def _group_correlation_paths(root: Path, pattern: str) -> Dict[Tuple[Path, str], Dict[str, List[Path]]]:
    """Group correlation CSVs by their logical key (class, split, etc.)."""
    grouped: Dict[Tuple[Path, str], Dict[str, List[Path]]] = defaultdict(lambda: defaultdict(list))
    for csv_path in sorted(root.rglob(pattern)):
        try:
            relative = csv_path.relative_to(root)
        except ValueError:
            # Fallback if the CSV is outside of the provided root.
            relative = csv_path
        base, group = _split_group(relative.stem)
        if group is None:
            continue
        grouped[(relative.parent, base)][group].append(csv_path)
    return grouped


def _align_matrices(matrices: Sequence[pd.DataFrame]) -> List[pd.DataFrame]:
    """Align matrices to their shared axes, preserving the first matrix order."""
    if not matrices:
        raise ValueError("No correlation matrices provided.")
    index = matrices[0].index
    columns = matrices[0].columns
    for matrix in matrices[1:]:
        index = index.intersection(matrix.index)
        columns = columns.intersection(matrix.columns)
    if len(index) == 0 or len(columns) == 0:
        raise ValueError("Correlation matrices do not share common axes.")
    ordered_index = pd.Index(index)
    ordered_columns = pd.Index(columns)
    return [matrix.loc[ordered_index, ordered_columns] for matrix in matrices]


def _mean_matrix(matrices: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Compute the element-wise mean of aligned matrices."""
    if not matrices:
        raise ValueError("Cannot compute mean of an empty matrix collection.")
    accumulator = matrices[0].copy()
    for matrix in matrices[1:]:
        accumulator += matrix
    return accumulator / len(matrices)


def compute_structural_difference_matrix(
    correct_paths: Sequence[Path | str],
    incorrect_paths: Sequence[Path | str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute mean correlation matrices and their difference (correct - incorrect)."""
    if not correct_paths or not incorrect_paths:
        raise ValueError("Both correct and incorrect correlation CSV collections are required.")

    correct_matrices = [_load_correlation_matrix(Path(path)) for path in correct_paths]
    incorrect_matrices = [_load_correlation_matrix(Path(path)) for path in incorrect_paths]

    aligned = _align_matrices([*correct_matrices, *incorrect_matrices])
    aligned_correct = aligned[: len(correct_matrices)]
    aligned_incorrect = aligned[len(correct_matrices) :]

    correct_mean = _mean_matrix(aligned_correct)
    incorrect_mean = _mean_matrix(aligned_incorrect)
    difference = correct_mean - incorrect_mean

    return difference, correct_mean, incorrect_mean


def _format_title(label: str, suffix: str) -> str:
    """Human-friendly title builder for correlation artefacts."""
    base = label.replace("_", " ").title()
    return f"{base} {suffix}".strip()


def _hierarchical_order(
    matrix: pd.DataFrame,
    *,
    method: str = "average",
    metric: str = "euclidean",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute row/column orderings using hierarchical clustering."""
    if matrix.shape[0] > 1:
        row_linkage = linkage(matrix.values, method=method, metric=metric)
        row_order = leaves_list(row_linkage)
    else:
        row_order = np.arange(matrix.shape[0])

    if matrix.shape[1] > 1:
        col_linkage = linkage(matrix.values.T, method=method, metric=metric)
        col_order = leaves_list(col_linkage)
    else:
        col_order = np.arange(matrix.shape[1])

    return row_order, col_order


def plot_structural_correlation_heatmap(
    csv_path: Path | str,
    output_path: Path | str | None = None,
    *,
    title: Optional[str] = None,
    dpi: int = 300,
    cmap: str = "coolwarm",
) -> Path:
    """Create a high-quality heatmap from a structural correlation CSV."""
    csv_file = Path(csv_path)
    matrix = _load_correlation_matrix(csv_file)

    if output_path is None:
        output_file = csv_file.with_suffix(".png")
    else:
        output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="white", context="paper")

    size = max(6.0, 0.6 * matrix.shape[0])
    fig, ax = plt.subplots(figsize=(size, size))

    mask = None
    if matrix.shape[0] == matrix.shape[1]:
        mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)

    heatmap = sns.heatmap(
        matrix,
        mask=mask,
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        annot=True,
        fmt=".2f",
        square=True,
        cbar_kws={"label": "Pearson correlation"},
        linewidths=0.5,
        linecolor="white",
        annot_kws={"size": 8},
    )

    heatmap.set_title(title or csv_file.stem.replace("_", " ").title(), pad=12)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=9)
    heatmap.figure.axes[-1].tick_params(labelsize=9)  # color bar ticks
    heatmap.tick_params(axis="both", which="both", labelsize=9)

    fig.tight_layout()
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_file


def _iter_structural_csvs(root: Path, pattern: str) -> Iterable[Path]:
    """Yield correlation CSV files under a directory that match the expected pattern."""
    return sorted(root.rglob(pattern))


def generate_structural_correlation_heatmaps(
    structural_root: Path | str,
    output_root: Path | str | None = None,
    *,
    pattern: str = "structural_correlations_*.csv",
    dpi: int = 300,
    cmap: str = "coolwarm",
) -> List[Path]:
    """Generate heatmaps for every structural correlation CSV found in a directory tree."""
    root_dir = Path(structural_root)
    if output_root is not None:
        out_dir = Path(output_root)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = None

    produced: List[Path] = []
    for csv_path in _iter_structural_csvs(root_dir, pattern):
        if out_dir is None:
            output_path = csv_path.with_suffix(".png")
        else:
            relative = csv_path.relative_to(root_dir)
            output_path = (out_dir / relative).with_suffix(".png")
            output_path.parent.mkdir(parents=True, exist_ok=True)

        heatmap_path = plot_structural_correlation_heatmap(csv_path, output_path=output_path, dpi=dpi, cmap=cmap)
        produced.append(heatmap_path)

    if not produced:
        raise FileNotFoundError(f"No structural correlation CSVs found under {root_dir} using pattern '{pattern}'.")

    return produced


def plot_structural_difference_heatmap(
    diff_matrix: pd.DataFrame,
    output_path: Path | str,
    *,
    title: Optional[str] = None,
    significance_threshold: Optional[float] = None,
    dpi: int = 300,
    cmap: str = "coolwarm",
    background_color: str = "#E5E5E5",
) -> Path:
    """Plot a heatmap showing the difference between correct and incorrect correlations."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="white", context="paper")

    size = max(6.0, 0.6 * diff_matrix.shape[0])
    fig, ax = plt.subplots(figsize=(size, size))

    mask = None
    if significance_threshold is not None:
        mask = np.abs(diff_matrix.values) < significance_threshold
        background = pd.DataFrame(
            np.zeros_like(diff_matrix.values),
            index=diff_matrix.index,
            columns=diff_matrix.columns,
        )
        sns.heatmap(
            background,
            ax=ax,
            cmap=[background_color],
            cbar=False,
            linewidths=0.5,
            linecolor="white",
        )

    heatmap = sns.heatmap(
        diff_matrix,
        mask=mask,
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        annot=True,
        fmt=".2f",
        square=True,
        cbar_kws={"label": "Correlation difference (correct - incorrect)"},
        linewidths=0.5,
        linecolor="white",
        annot_kws={"size": 8},
        ax=ax,
    )

    heatmap.set_title(title or "Correlation Difference", pad=12)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=9)
    heatmap.figure.axes[-1].tick_params(labelsize=9)
    heatmap.tick_params(axis="both", which="both", labelsize=9)

    fig.tight_layout()
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_file


def plot_clustered_correlation_heatmap(
    matrix: pd.DataFrame,
    output_path: Path | str,
    *,
    title: Optional[str] = None,
    significance_threshold: Optional[float] = None,
    dpi: int = 300,
    cmap: str = "coolwarm",
    method: str = "average",
    metric: str = "euclidean",
    background_color: str = "#E5E5E5",
) -> Path:
    """Reorder a correlation matrix through hierarchical clustering and plot the result."""
    filled = matrix.fillna(0.0)
    row_order, col_order = _hierarchical_order(filled, method=method, metric=metric)
    ordered = matrix.iloc[row_order, :]
    ordered = ordered.iloc[:, col_order]

    return plot_structural_difference_heatmap(
        ordered,
        output_path=output_path,
        title=title,
        significance_threshold=significance_threshold,
        dpi=dpi,
        cmap=cmap,
        background_color=background_color,
    )


def generate_structural_difference_heatmaps(
    structural_root: Path | str,
    output_root: Path | str | None = None,
    *,
    pattern: str = "structural_correlations_*.csv",
    significance_threshold: Optional[float] = None,
    dpi: int = 300,
    cmap: str = "coolwarm",
    background_color: str = "#E5E5E5",
    create_clustered: bool = True,
    clustered_suffix: str = "_difference_clustered",
    method: str = "average",
    metric: str = "euclidean",
) -> List[Path]:
    """Generate difference and clustered heatmaps comparing correct vs. incorrect correlations."""
    root_dir = Path(structural_root)
    if output_root is not None:
        out_dir = Path(output_root)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = None

    grouped = _group_correlation_paths(root_dir, pattern)
    produced: List[Path] = []

    for (relative_parent, base), paths in grouped.items():
        correct_paths = paths.get("correct", [])
        incorrect_paths = paths.get("incorrect", [])
        if not correct_paths or not incorrect_paths:
            continue

        difference, correct_mean, incorrect_mean = compute_structural_difference_matrix(correct_paths, incorrect_paths)

        if out_dir is None:
            target_dir = (root_dir / relative_parent).resolve()
        else:
            target_dir = (out_dir / relative_parent).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)

        diff_csv = target_dir / f"{base}_difference.csv"
        difference.to_csv(diff_csv)
        produced.append(diff_csv)

        correct_mean_csv = target_dir / f"{base}_correct_mean.csv"
        incorrect_mean_csv = target_dir / f"{base}_incorrect_mean.csv"
        correct_mean.to_csv(correct_mean_csv)
        incorrect_mean.to_csv(incorrect_mean_csv)
        produced.extend([correct_mean_csv, incorrect_mean_csv])

        diff_png = target_dir / f"{base}_difference.png"
        title = _format_title(base, "(Correct - Incorrect)")
        produced.append(
            plot_structural_difference_heatmap(
                difference,
                output_path=diff_png,
                title=title,
                significance_threshold=significance_threshold,
                dpi=dpi,
                cmap=cmap,
                background_color=background_color,
            )
        )

        if create_clustered:
            clustered_png = target_dir / f"{base}{clustered_suffix}.png"
            clustered_title = _format_title(base, "Clustered Difference")
            produced.append(
                plot_clustered_correlation_heatmap(
                    difference,
                    output_path=clustered_png,
                    title=clustered_title,
                    significance_threshold=significance_threshold,
                    dpi=dpi,
                    cmap=cmap,
                    method=method,
                    metric=metric,
                    background_color=background_color,
                )
            )

    if not produced:
        raise FileNotFoundError(
            f"No paired 'correct' and 'incorrect' structural correlation CSVs found under {root_dir}."
        )

    return produced
