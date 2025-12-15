"""Visualisations for sparsity KDEs by class and correctness."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DEFAULT_OUTPUT_ROOT = Path("outputs/analytics/sparsity")


def _iter_summary_csvs(root: Path, pattern: str) -> Iterable[Path]:
    return sorted(root.rglob(pattern))


def _label_suffix(label: Optional[str]) -> str:
    if label is None:
        return ""
    safe = str(label).strip()
    if not safe or safe.lower() in {"nan", "none"}:
        return ""
    return safe.replace("/", "_").replace(" ", "_")


def _plot_kde(series: pd.Series, title: str, output_path: Path, *, bw_adjust: float = 1.0, dpi: int = 300) -> Path:
    cleaned = series.dropna()
    if cleaned.empty:
        return output_path
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.kdeplot(cleaned, fill=True, ax=ax, color="#4C72B0", bw_adjust=bw_adjust)
    ax.set_xlabel("Sparsity")
    ax.set_ylabel("Density")
    ax.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_difference(
    correct: pd.Series,
    incorrect: pd.Series,
    title: str,
    output_path: Path,
    *,
    bw_adjust: float = 1.0,
    dpi: int = 300,
) -> Optional[Tuple[Path, Dict[str, float]]]:
    correct = correct.dropna()
    incorrect = incorrect.dropna()
    if correct.empty or incorrect.empty:
        return None
    grid = np.linspace(0.0, 1.0, 256)

    def _kde(values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return np.zeros_like(grid)
        std = values.std(ddof=1) if values.size > 1 else 0.1
        bandwidth = max(0.05, std * (values.size ** (-1 / 5)) * bw_adjust)
        diff = (grid[None, :] - values[:, None]) / bandwidth
        densities = np.exp(-0.5 * diff**2).sum(axis=0)
        densities /= (values.size * bandwidth * np.sqrt(2 * np.pi))
        return densities

    correct_density = _kde(correct.to_numpy())
    incorrect_density = _kde(incorrect.to_numpy())
    delta = correct_density - incorrect_density

    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid, delta, color="#4C72B0", linewidth=1.6)
    ax.axhline(0.0, color="#C44E52", linestyle="--", linewidth=1.0)
    ax.fill_between(grid, delta, 0.0, where=delta >= 0, color="#4C72B0", alpha=0.25)
    ax.fill_between(grid, delta, 0.0, where=delta < 0, color="#C44E52", alpha=0.25)
    ax.set_xlabel("Sparsity")
    ax.set_ylabel("Density difference (correct - incorrect)")
    ax.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    abs_delta = np.abs(delta)
    stats = {
        "area": float(np.trapz(delta, grid)),
        "positive_mass": float(np.trapz(np.clip(delta, 0.0, None), grid)),
        "negative_mass": float(np.trapz(np.clip(-delta, 0.0, None), grid)),
        "max_difference": float(delta.max() if delta.size else 0.0),
        "min_difference": float(delta.min() if delta.size else 0.0),
        "abs_area": float(np.trapz(abs_delta, grid)),
        "std_difference": float(delta.std(ddof=0)) if delta.size else 0.0,
        "mean_difference": float(delta.mean()) if delta.size else 0.0,
        "max_abs_difference": float(abs_delta.max()) if delta.size else 0.0,
        "range_difference": float((delta.max() - delta.min()) if delta.size else 0.0),
    }
    return output_path, stats


def generate_sparsity_visuals(
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
    dataset_slug = Path(root_dir).name
    difference_rows: List[Dict[str, float]] = []

    for csv_path in _iter_summary_csvs(root_dir, pattern):
        frame = pd.read_csv(csv_path)
        if "sparsity" not in frame.columns:
            continue

        base_dir = csv_path.parent if out_dir is None else (out_dir / csv_path.relative_to(root_dir).parent)
        base_dir.mkdir(parents=True, exist_ok=True)
        sparsity_series = pd.to_numeric(frame["sparsity"], errors="coerce")
        produced.append(
            _plot_kde(
                sparsity_series,
                f"Sparsity KDE - {csv_path.stem}",
                base_dir / f"{csv_path.stem}_sparsity_kde.png",
                dpi=dpi,
            )
        )

        if "is_correct" in frame.columns:
            relative = csv_path.relative_to(root_dir)
            dataset_key = relative.parts[0] if relative.parts else "root"

            for correctness_flag, subset in frame.groupby("is_correct", dropna=False):
                if pd.isna(correctness_flag):
                    continue
                status = "correct" if correctness_flag else "incorrect"
                status_dir = base_dir / status
                produced.append(
                    _plot_kde(
                        pd.to_numeric(subset["sparsity"], errors="coerce"),
                        f"Sparsity KDE ({status}) - {csv_path.stem}",
                        status_dir / f"{csv_path.stem}_{status}_sparsity_kde.png",
                        dpi=dpi,
                    )
                )

                if "label" in subset.columns:
                    for label, combo in subset.groupby("label", dropna=False):
                        if pd.isna(label):
                            continue
                        label_id = _label_suffix(label)
                        combo_dir = status_dir / f"class_{label_id}" if label_id else status_dir / "class"
                        produced.append(
                            _plot_kde(
                                pd.to_numeric(combo["sparsity"], errors="coerce"),
                                f"Sparsity KDE ({status}, label {label}) - {csv_path.stem}",
                                combo_dir / f"{csv_path.stem}_{status}_class{label_id}_sparsity_kde.png",
                                dpi=dpi,
                            )
                        )

            # overall correctness comparison (all labels)
            correct_series = pd.to_numeric(frame.loc[frame.get("is_correct") == True, "sparsity"], errors="coerce")
            incorrect_series = pd.to_numeric(frame.loc[frame.get("is_correct") == False, "sparsity"], errors="coerce")
            overall_result = _plot_difference(
                correct_series,
                incorrect_series,
                f"Sparsity KDE (correct vs incorrect) - {csv_path.stem}",
                base_dir / f"{csv_path.stem}_correct_vs_incorrect_sparsity_kde.png",
                dpi=dpi,
            )
            if overall_result is not None:
                path, stats = overall_result
                produced.append(path)
                stats.update({"dataset": dataset_slug, "scope": "overall", "metric": "sparsity"})
                difference_rows.append(stats)

            if "label" in frame.columns:
                for label, subset in frame.groupby("label", dropna=False):
                    if pd.isna(label):
                        continue
                    label_id = _label_suffix(label)
                    label_dir = base_dir / f"class_{label_id}" if label_id else base_dir / "class"
                    correct_subset = pd.to_numeric(subset.loc[subset.get("is_correct") == True, "sparsity"], errors="coerce")
                    incorrect_subset = pd.to_numeric(subset.loc[subset.get("is_correct") == False, "sparsity"], errors="coerce")
                    produced.append(
                        _plot_kde(
                            pd.to_numeric(subset["sparsity"], errors="coerce"),
                            f"Sparsity KDE (label {label}) - {csv_path.stem}",
                            label_dir / f"{csv_path.stem}_class{label_id}_sparsity_kde.png",
                            dpi=dpi,
                        )
                    )
                    label_result = _plot_difference(
                        correct_subset,
                        incorrect_subset,
                        f"Sparsity KDE (label {label}, correct vs incorrect) - {csv_path.stem}",
                        label_dir / f"{csv_path.stem}_class{label_id}_correct_vs_incorrect_sparsity_kde.png",
                        dpi=dpi,
                    )
                    if label_result is not None:
                        path, stats = label_result
                        produced.append(path)
                        stats.update({"dataset": dataset_slug, "scope": f"class_{label_id}", "metric": "sparsity"})
                        difference_rows.append(stats)
        else:
            produced.append(
                _plot_kde(
                    pd.to_numeric(frame["sparsity"], errors="coerce"),
                    f"Sparsity KDE - {csv_path.stem}",
                    base_dir / f"{csv_path.stem}_sparsity_kde.png",
                    dpi=dpi,
                )
            )

    if difference_rows:
        metrics_dir = out_dir if out_dir else root_dir
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = metrics_dir / "sparsity_difference_metrics.csv"
        pd.DataFrame(difference_rows).to_csv(metrics_path, index=False)
        produced.append(metrics_path)

    return [p for p in produced if isinstance(p, Path)]
