from __future__ import annotations

import argparse
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from tqdm import tqdm

try:
    import plotly.graph_objects as go
    import plotly.io as pio

    _PLOTLY_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    go = pio = None
    _PLOTLY_AVAILABLE = False

import matplotlib.pyplot as plt


BASE_INPUT_ROOT = Path("outputs/analytics/consistency")
DEFAULT_GRID_POINTS = 256
MIN_BANDWIDTH = 1e-3
PLOT_WIDTH = 720
PLOT_HEIGHT = 480
FONT_SIZE = 10
COLOR_FAITHFUL = "#636EFA"
COLOR_SUFF = "#EF553B"
COLOR_COMP = "#00CC96"
COLOR_CORRECT = "#1f77b4"
COLOR_INCORRECT = "#d62728"

CHROME_DEFAULT = Path("/home/appuser/.local/share/plotly_chrome/chrome-linux64/chrome")
if CHROME_DEFAULT.exists():
    os.environ.setdefault("PLOTLY_KALEIDO_CHROME_PATH", str(CHROME_DEFAULT))
    os.environ["PATH"] = f"{CHROME_DEFAULT.parent}:{os.environ.get('PATH', '')}"
    if _PLOTLY_AVAILABLE:
        try:
            pio.kaleido.scope.chromium_executable = str(CHROME_DEFAULT)
            pio.kaleido.scope.chromium_args = [
                "--headless",
                "--disable-gpu",
                "--disable-software-rasterizer",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ]
        except Exception:
            pass


def slugify(value: object) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip().lower()
    allowed = [ch if ch.isalnum() else "_" for ch in text]
    slug = "".join(allowed).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "unknown"


def parse_numeric_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric


def parse_correctness(value: object, label: object, prediction: object) -> Optional[bool]:
    if value is not None and not (isinstance(value, float) and math.isnan(value)):
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n"}:
            return False
    label_norm = normalize_label(label)
    pred_norm = normalize_label(prediction)
    if label_norm is not None and pred_norm is not None:
        return label_norm == pred_norm
    return None


def normalize_label(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return str(int(float(text)))
    except Exception:
        return text


def build_grid(values: np.ndarray, grid_points: int) -> Optional[np.ndarray]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    vmin = float(finite.min())
    vmax = float(finite.max())
    if math.isclose(vmin, vmax):
        delta = max(abs(vmin), 1.0) * 0.05
        if delta == 0:
            delta = 0.05
        vmin -= delta
        vmax += delta
    margin = 0.02 * (vmax - vmin) or 0.1
    start = vmin - margin
    end = vmax + margin
    return np.linspace(start, end, grid_points)


def compute_density(
    samples: Iterable[float],
    *,
    grid: np.ndarray,
    bandwidth: float = MIN_BANDWIDTH,
) -> np.ndarray:
    values = np.asarray(list(samples), dtype=float)
    if values.size < 2:
        return np.zeros_like(grid)

    try:
        kde = gaussian_kde(values)
        density = kde(grid)
    except Exception:
        std = float(np.std(values))
        bw = max(bandwidth, std * 0.1)
        bw = max(bw, MIN_BANDWIDTH)
        norm = 1.0 / (bw * math.sqrt(2 * math.pi))
        diffs = (grid[None, :] - values[:, None]) / bw
        kernel = np.exp(-0.5 * diffs**2) * norm
        density = kernel.mean(axis=0)

    area = np.trapezoid(density, grid)
    if area > 0:
        density = density / area
    return density


def _safe_distribution(density: np.ndarray, epsilon: float = 1e-9) -> np.ndarray:
    dist = np.clip(density, epsilon, None)
    total = dist.sum()
    if total <= 0:
        return np.full_like(dist, 1.0 / len(dist))
    return dist / total


def compute_kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p_safe = _safe_distribution(p)
    q_safe = _safe_distribution(q)
    return float(np.sum(p_safe * np.log(p_safe / q_safe)))


def compute_js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p_safe = _safe_distribution(p)
    q_safe = _safe_distribution(q)
    m = 0.5 * (p_safe + q_safe)
    return 0.5 * float(np.sum(p_safe * np.log(p_safe / m))) + 0.5 * float(np.sum(q_safe * np.log(q_safe / m)))


def compute_ks_statistic(cdf_p: np.ndarray, cdf_q: np.ndarray) -> float:
    return float(np.max(np.abs(cdf_p - cdf_q)))


def plot_density_single(
    grid: np.ndarray,
    density: np.ndarray,
    *,
    output_path: Path,
    title: str,
    color: str,
    y_title: str = "Density",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if _PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=density,
                mode="lines",
                line=dict(color=color, width=2),
                fill="tozeroy",
                fillcolor=color,
                opacity=0.45,
                name="density",
            )
        )
        fig.update_layout(
            title=title,
            font=dict(size=FONT_SIZE),
            width=PLOT_WIDTH,
            height=PLOT_HEIGHT,
            xaxis=dict(title="Metric value"),
            yaxis=dict(title=y_title),
            margin=dict(l=60, r=25, t=60, b=60),
            showlegend=False,
        )
        try:
            fig.write_image(str(output_path), format="pdf")
            print(f"[kde] {output_path}")
            return
        except Exception as exc:  # pragma: no cover - fallback to matplotlib
            print(f"[warn] Plotly export failed for {output_path}: {exc}. Falling back to Matplotlib.")

    plt.figure(figsize=(PLOT_WIDTH / 120, PLOT_HEIGHT / 120))
    plt.plot(grid, density, color=color, linewidth=1.8)
    plt.fill_between(grid, density, color=color, alpha=0.35)
    plt.title(title, fontsize=FONT_SIZE + 2)
    plt.xlabel("Metric value", fontsize=FONT_SIZE)
    plt.ylabel(y_title, fontsize=FONT_SIZE)
    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"[kde-fallback] {output_path}")


def plot_density_difference(
    grid: np.ndarray,
    density_correct: np.ndarray,
    density_incorrect: np.ndarray,
    *,
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diff = density_correct - density_incorrect
    if _PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_hline(y=0, line=dict(color="#444", width=1, dash="dash"))
        positive = np.where(diff >= 0, diff, np.nan)
        negative = np.where(diff <= 0, diff, np.nan)
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=positive,
                mode="lines",
                line=dict(color=COLOR_CORRECT, width=2),
                fill="tozeroy",
                fillcolor="rgba(31, 119, 180, 0.3)",
                name="correct - incorrect (positive)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=negative,
                mode="lines",
                line=dict(color=COLOR_INCORRECT, width=2),
                fill="tozeroy",
                fillcolor="rgba(214, 39, 40, 0.3)",
                name="correct - incorrect (negative)",
            )
        )
        fig.update_layout(
            title=title,
            font=dict(size=FONT_SIZE),
            width=PLOT_WIDTH,
            height=PLOT_HEIGHT,
            xaxis=dict(title="Metric value"),
            yaxis=dict(title="Density difference (correct - incorrect)"),
            margin=dict(l=60, r=25, t=60, b=60),
            showlegend=False,
        )
        try:
            fig.write_image(str(output_path), format="pdf")
            print(f"[kde-diff] {output_path}")
            return
        except Exception as exc:  # pragma: no cover - fallback to matplotlib
            print(f"[warn] Plotly export failed for {output_path}: {exc}. Falling back to Matplotlib.")

    plt.figure(figsize=(PLOT_WIDTH / 120, PLOT_HEIGHT / 120))
    plt.axhline(0, color="#444", linewidth=1, linestyle="--")
    plt.fill_between(grid, np.maximum(diff, 0), color=COLOR_CORRECT, alpha=0.35)
    plt.fill_between(grid, np.minimum(diff, 0), color=COLOR_INCORRECT, alpha=0.35)
    plt.plot(grid, diff, color="#333", linewidth=1.6)
    plt.title(title, fontsize=FONT_SIZE + 2)
    plt.xlabel("Metric value", fontsize=FONT_SIZE)
    plt.ylabel("Density difference (correct - incorrect)", fontsize=FONT_SIZE)
    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"[kde-diff-fallback] {output_path}")


def save_density_csv(
    grid: np.ndarray,
    density: np.ndarray,
    output_path: Path,
    *,
    extra_columns: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    data = {"value": grid, "density": density}
    if extra_columns:
        data.update(extra_columns)
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"[data] {output_path}")


@dataclass
class SampleBucket:
    values: List[float]

    def __init__(self) -> None:
        self.values = []

    def extend(self, value: float) -> None:
        self.values.append(value)

    def extend_many(self, values: Iterable[float]) -> None:
        self.values.extend(values)

    def has_samples(self) -> bool:
        return len(self.values) >= 2 and any(np.isfinite(self.values))


METRIC_CONFIGS = {
    "faithfulness": {"title": "Faithfulness", "color": COLOR_FAITHFUL},
    "sufficiency": {"title": "Sufficiency", "color": COLOR_SUFF},
    "comprehensiveness": {"title": "Comprehensiveness", "color": COLOR_COMP},
}


def discover_consistency_csvs(root: Path) -> List[Path]:
    return sorted(root.glob("**/*.csv"))


def prepare_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    baseline_src = "baseline_margin" if "baseline_margin" in df.columns else "origin_contrastivity"
    sufficiency_src = "preservation_sufficiency" if "preservation_sufficiency" in df.columns else "masked_contrastivity"
    necessity_src = "preservation_necessity" if "preservation_necessity" in df.columns else "maskout_contrastivity"

    for column in {baseline_src, sufficiency_src, necessity_src}:
        if column in df.columns:
            df[column] = parse_numeric_series(df[column])
        else:
            df[column] = np.nan

    df["baseline_margin"] = df[baseline_src]
    df["preservation_sufficiency"] = df[sufficiency_src]
    df["preservation_necessity"] = df[necessity_src]

    df["faithfulness"] = df["baseline_margin"] - df["preservation_sufficiency"]
    df["sufficiency"] = df["preservation_sufficiency"]
    diff = (df["baseline_margin"] - df["preservation_necessity"]).abs()
    df["comprehensiveness"] = 1.0 - diff
    df["comprehensiveness"] = df["comprehensiveness"].clip(lower=0.0, upper=1.0)
    return df


def process_metric(
    df: pd.DataFrame,
    csv_path: Path,
    metric_name: str,
    metric_title: str,
    color: str,
    grid_points: int,
) -> None:
    if metric_name not in df.columns:
        return

    values = parse_numeric_series(df[metric_name])
    valid_mask = values.notna()
    if valid_mask.sum() < 2:
        return

    valid_values = values[valid_mask].to_numpy(dtype=float)
    grid = build_grid(valid_values, grid_points)
    if grid is None:
        return

    graph_dir = csv_path.parent / csv_path.stem
    metric_dir = graph_dir / metric_name
    metric_dir.mkdir(parents=True, exist_ok=True)

    groups_all: Dict[Tuple[str, Optional[str]], SampleBucket] = defaultdict(SampleBucket)
    correctness_groups: Dict[Tuple[str, Optional[str]], SampleBucket] = defaultdict(SampleBucket)
    diff_groups: Dict[Tuple[str, Optional[str]], Dict[bool, SampleBucket]] = defaultdict(
        lambda: {True: SampleBucket(), False: SampleBucket()}
    )

    for row in df[valid_mask].itertuples(index=False):
        value = getattr(row, metric_name)
        if value is None or not np.isfinite(value):
            continue

        label = getattr(row, "label", None)
        prediction_class = getattr(row, "prediction_class", None)
        is_correct_raw = getattr(row, "is_correct", None)
        correctness = parse_correctness(is_correct_raw, label, prediction_class)

        groups_all[("overall", None)].extend(value)
        if label is not None and not (isinstance(label, float) and math.isnan(label)):
            groups_all[("class", str(label))].extend(value)

        if correctness is not None:
            correctness_key = "true" if correctness else "false"
            correctness_groups[("correct", correctness_key)].extend(value)
            diff_groups[("overall", None)][correctness].extend(value)
            if label is not None and not (isinstance(label, float) and math.isnan(label)):
                label_str = str(label)
                correctness_groups[("class_correct", f"{label_str}_{correctness_key}")].extend(value)
                diff_groups[("class", label_str)][correctness].extend(value)

    # Plot overall and class groups
    for (group_type, identifier), bucket in groups_all.items():
        if not bucket.has_samples():
            continue
        if group_type == "overall":
            title = f"{metric_title} | All Samples"
            file_stub = "kde_overall"
        elif group_type == "class":
            title = f"{metric_title} | Class {identifier}"
            file_stub = f"kde_class_{slugify(identifier)}"
        else:
            continue

        density = compute_density(bucket.values, grid=grid)
        plot_density_single(
            grid,
            density,
            output_path=metric_dir / f"{file_stub}.pdf",
            title=title,
            color=color,
        )
        save_density_csv(grid, density, metric_dir / f"density_{file_stub}.csv")

    # Plot correctness groups
    for (group_type, identifier), bucket in correctness_groups.items():
        if not bucket.has_samples():
            continue
        if group_type == "correct":
            title = f"{metric_title} | Correct = {identifier}"
            file_stub = f"kde_correct_{identifier}"
            plot_color = COLOR_CORRECT if identifier == "true" else COLOR_INCORRECT
        elif group_type == "class_correct":
            label_part, correctness_part = identifier.split("_", 1)
            title = f"{metric_title} | Class {label_part} ({correctness_part})"
            file_stub = f"kde_class_{slugify(label_part)}_correct_{correctness_part}"
            plot_color = COLOR_CORRECT if correctness_part == "true" else COLOR_INCORRECT
        else:
            continue

        density = compute_density(bucket.values, grid=grid)
        plot_density_single(
            grid,
            density,
            output_path=metric_dir / f"{file_stub}.pdf",
            title=title,
            color=plot_color,
        )
        save_density_csv(grid, density, metric_dir / f"density_{file_stub}.csv")

    # Differences and divergence metrics
    for (group_type, identifier), buckets in diff_groups.items():
        bucket_true = buckets[True]
        bucket_false = buckets[False]
        if not bucket_true.has_samples() or not bucket_false.has_samples():
            continue

        if group_type == "overall":
            title = f"{metric_title} | All Samples (Correct - Incorrect)"
            file_stub = "kde_correct_vs_incorrect"
        elif group_type == "class":
            title = f"{metric_title} | Class {identifier} (Correct - Incorrect)"
            file_stub = f"kde_class_{slugify(identifier)}_correct_vs_incorrect"
        else:
            continue

        density_true = compute_density(bucket_true.values, grid=grid)
        density_false = compute_density(bucket_false.values, grid=grid)

        plot_density_difference(
            grid,
            density_true,
            density_false,
            output_path=metric_dir / f"{file_stub}.pdf",
            title=title,
        )
        save_density_csv(
            grid,
            density_true - density_false,
            metric_dir / f"density_{file_stub}.csv",
            extra_columns={
                "density_correct": density_true,
                "density_incorrect": density_false,
            },
        )

        delta = grid[1] - grid[0] if len(grid) > 1 else 1.0
        cdf_true = np.cumsum(density_true) * delta
        cdf_false = np.cumsum(density_false) * delta
        cdf_true = cdf_true / max(cdf_true[-1], 1e-9)
        cdf_false = cdf_false / max(cdf_false[-1], 1e-9)

        kl_true_false = compute_kl_divergence(density_true, density_false)
        kl_false_true = compute_kl_divergence(density_false, density_true)
        js_div = compute_js_divergence(density_true, density_false)
        ks_stat = compute_ks_statistic(cdf_true, cdf_false)

        metrics_path = metric_dir / f"metrics_{file_stub}.json"
        metrics_path.write_text(
            pd.Series(
                {
                    "kl_correct_vs_incorrect": kl_true_false,
                    "kl_incorrect_vs_correct": kl_false_true,
                    "js_divergence": js_div,
                    "ks_statistic": ks_stat,
                }
            ).to_json(indent=2)
        )


def process_csv(csv_path: Path, grid_points: int) -> None:
    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"[skip] Empty consistency CSV skipped: {csv_path}")
        return

    df = prepare_metric_columns(df)

    for metric_name, config in METRIC_CONFIGS.items():
        process_metric(
            df,
            csv_path,
            metric_name=metric_name,
            metric_title=config["title"],
            color=config["color"],
            grid_points=grid_points,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate KDEs and divergence metrics for consistency analytics.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=BASE_INPUT_ROOT,
        help="Directory containing per-dataset consistency CSV files.",
    )
    parser.add_argument(
        "--grid-points",
        type=int,
        default=DEFAULT_GRID_POINTS,
        help="Number of evaluation points for the KDE grid.",
    )
    args = parser.parse_args()

    base_dir: Path = args.base_dir
    if not base_dir.exists():
        raise SystemExit(f"Base directory not found: {base_dir}")

    csv_paths = discover_consistency_csvs(base_dir)
    if not csv_paths:
        print("No consistency CSV files found.")
        return

    for csv_path in tqdm(csv_paths, desc="Generating consistency KDEs"):
        try:
            process_csv(csv_path, args.grid_points)
        except Exception as exc:  # pragma: no cover - logging only
            print(f"[warn] failed to process {csv_path}: {exc}")


if __name__ == "__main__":
    main()
