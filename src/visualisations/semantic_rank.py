"\"\"\"Visualisations for token rank/position distributions.\"\"\""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DEFAULT_OUTPUT_ROOT = Path("outputs/analytics/position/rank")


def _load_tokens(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    if not {"rank", "position"}.intersection(frame.columns):
        raise ValueError(f"Expected 'rank' or 'position' column in {csv_path}")
    return frame


def _iter_token_csvs(root: Path, pattern: str) -> Iterable[Path]:
    return sorted(root.rglob(pattern))


def _resolve_output_path(
    csv_path: Path,
    root_dir: Path,
    out_dir: Optional[Path],
    suffix: str,
) -> Path:
    if out_dir is None:
        base = csv_path.parent / suffix.strip("/") if suffix.startswith("/") else csv_path.parent
        base.mkdir(parents=True, exist_ok=True)
        return (base / csv_path.stem).with_suffix(".png")

    try:
        relative = csv_path.relative_to(root_dir)
    except ValueError:
        relative = Path(csv_path.name)
    target_dir = out_dir / relative.parent
    if suffix:
        target_dir = target_dir / suffix
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"{relative.stem}.png"


def plot_token_rank_histogram(
    csv_path: Path | str,
    output_path: Path | str | None = None,
    *,
    title: Optional[str] = None,
    dpi: int = 300,
) -> Path | None:
    csv_file = Path(csv_path)
    frame = _load_tokens(csv_file)
    column = "position" if "position" in frame.columns else "rank"
    series = frame[column].dropna().astype(float)
    if series.empty:
        return None

    output_file = Path(output_path) if output_path else csv_file.with_suffix(".png")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(6, 4))

    if column == "position":
        clipped = series.clip(0.0, 1.0)
        sns.kdeplot(clipped, ax=ax, fill=True, bw_adjust=0.5, clip=(0.0, 1.0), color="#4C72B0")
        ax.set_xlabel("Token position (0 = start, 1 = end)")
        ax.set_ylabel("Density")
        ax.set_xlim(0.0, 1.0)
        ax.margins(x=0)
    else:
        counts = series.value_counts().sort_index()
        ax.plot(counts.index, counts.values, marker="o", linewidth=1.8, color="#4C72B0")
        ax.fill_between(counts.index, counts.values, alpha=0.15, color="#4C72B0")
        ax.set_xlabel("Token rank (importance order)")
        ax.set_ylabel("Count")
        ax.set_xlim(left=max(1, counts.index.min()))

    ax.set_title(title or csv_file.stem.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_file


def generate_token_rank_histograms(
    tokens_root: Path | str,
    output_root: Path | str | None = DEFAULT_OUTPUT_ROOT,
    *,
    pattern: str = "*tokens.csv",
    dpi: int = 300,
) -> List[Path]:
    root_dir = Path(tokens_root)
    out_dir = Path(output_root) if output_root else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    produced: List[Path] = []
    for csv_path in _iter_token_csvs(root_dir, pattern):
        target = _resolve_output_path(csv_path, root_dir, out_dir, "")
        result = plot_token_rank_histogram(csv_path, output_path=target, dpi=dpi)
        if result is not None:
            produced.append(result)
    return produced


def _kde(values: np.ndarray, grid: np.ndarray, bandwidth: Optional[float] = None) -> np.ndarray:
    if values.size == 0:
        return np.zeros_like(grid)
    if bandwidth is None:
        std = values.std(ddof=1) if values.size > 1 else 0.0
        bandwidth = 1.06 * std * (values.size ** (-1 / 5)) if std > 0 else 0.05
        bandwidth = max(bandwidth, 0.02)
    diff = (grid[None, :] - values[:, None]) / bandwidth
    densities = np.exp(-0.5 * diff**2).sum(axis=0)
    densities /= (values.size * bandwidth * np.sqrt(2 * np.pi))
    return densities


def plot_token_rank_difference(
    correct_csv: Path | str,
    incorrect_csv: Path | str,
    output_path: Path | str | None = None,
    *,
    title: Optional[str] = None,
    dpi: int = 300,
) -> Path | None:
    correct_frame = _load_tokens(Path(correct_csv))
    incorrect_frame = _load_tokens(Path(incorrect_csv))

    column = "position" if "position" in correct_frame.columns else "rank"
    if column not in incorrect_frame.columns:
        return None

    correct_values = correct_frame[column].dropna().astype(float).to_numpy()
    incorrect_values = incorrect_frame[column].dropna().astype(float).to_numpy()
    if correct_values.size == 0 and incorrect_values.size == 0:
        return None

    if column == "position":
        correct_values = np.clip(correct_values, 0.0, 1.0)
        incorrect_values = np.clip(incorrect_values, 0.0, 1.0)
        grid = np.linspace(0.0, 1.0, 256)
    else:
        min_rank = int(min(np.min(correct_values, initial=1), np.min(incorrect_values, initial=1)))
        max_rank = int(max(np.max(correct_values, initial=min_rank), np.max(incorrect_values, initial=min_rank)))
        grid = np.linspace(min_rank, max_rank, max(max_rank - min_rank + 1, 200))

    difference = _kde(correct_values, grid) - _kde(incorrect_values, grid)
    output_file = Path(output_path) if output_path else Path(correct_csv).with_suffix(".png")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid, difference, color="#C44E52", linewidth=1.8)
    ax.axhline(0.0, color="#4C72B0", linestyle="--", linewidth=1.0)
    ax.fill_between(grid, difference, 0.0, where=difference >= 0, color="#C44E52", alpha=0.2)
    ax.fill_between(grid, difference, 0.0, where=difference < 0, color="#55A868", alpha=0.2)
    ax.set_xlabel("Token position" if column == "position" else "Token rank")
    ax.set_ylabel("Density difference (correct - incorrect)")
    ax.set_title(title or f"{Path(correct_csv).stem} vs incorrect")
    if column == "position":
        ax.set_xlim(0.0, 1.0)
        ax.margins(x=0)
    fig.tight_layout()
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_file


def _group_correct_incorrect(paths: Iterable[Path]) -> Dict[str, Dict[str, Path]]:
    groups: Dict[str, Dict[str, Path]] = defaultdict(dict)
    for path in paths:
        stem = path.stem
        if not stem.endswith("_tokens"):
            continue
        core = stem[: -len("_tokens")]
        if core.endswith("_correct"):
            groups[core[: -len("_correct")]]["correct"] = path
        elif core.endswith("_incorrect"):
            groups[core[: -len("_incorrect")]]["incorrect"] = path
    return groups


def generate_token_rank_differences(
    tokens_root: Path | str,
    output_root: Path | str | None = DEFAULT_OUTPUT_ROOT,
    *,
    pattern: str = "*tokens.csv",
    dpi: int = 300,
) -> List[Path]:
    root_dir = Path(tokens_root)
    out_dir = Path(output_root) if output_root else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    groups = _group_correct_incorrect(_iter_token_csvs(root_dir, pattern))
    produced: List[Path] = []
    for key, subset in groups.items():
        correct_path = subset.get("correct")
        incorrect_path = subset.get("incorrect")
        if not correct_path or not incorrect_path:
            continue

        if out_dir is None:
            target_dir = correct_path.parent / "rank_differences"
            target_dir.mkdir(parents=True, exist_ok=True)
            output_path = target_dir / f"{key}_correct_vs_incorrect_rank_diff.png"
        else:
            try:
                relative = Path(correct_path).relative_to(root_dir)
            except ValueError:
                relative = Path(correct_path.name)
            target_dir = out_dir / relative.parent / "differences"
            target_dir.mkdir(parents=True, exist_ok=True)
            output_path = target_dir / f"{key}_correct_vs_incorrect_rank_diff.png"

        result = plot_token_rank_difference(
            correct_path,
            incorrect_path,
            output_path=output_path,
            dpi=dpi,
            title=f"{key.replace('_', ' ').title()} (Correct - Incorrect)",
        )
        if result is not None:
            produced.append(result)
    return produced
