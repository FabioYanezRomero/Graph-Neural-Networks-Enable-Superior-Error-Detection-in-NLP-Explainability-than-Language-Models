from __future__ import annotations

import argparse
import ast
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


BASE_INPUT_ROOT = Path("outputs/analytics/sequence")
DEFAULT_GRID_POINTS = 256
MIN_BANDWIDTH = 0.03
PLOT_WIDTH = 720
PLOT_HEIGHT = 480
FONT_SIZE = 10
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
    slug = "".join(allowed)
    while "__" in slug:
        slug = slug.replace("__", "_")
    slug = slug.strip("_")
    return slug or "unknown"


def parse_ranked_nodes(raw: object) -> List[int]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, np.ndarray)):
        iterable: Iterable = raw
    elif isinstance(raw, str):
        tokens = raw.replace(",", " ").split()
        iterable = tokens
    else:
        return []

    nodes: List[int] = []
    for token in iterable:
        try:
            nodes.append(int(token))
        except Exception:
            continue
    return nodes


def parse_ranked_map(raw: object) -> List[float]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, np.ndarray)):
        iterable: Iterable = raw
    elif isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            value = ast.literal_eval(text)
            if isinstance(value, (list, tuple, np.ndarray)):
                iterable = value
            else:
                # Fallback: treat as whitespace separated list
                iterable = text.replace(",", " ").split()
        except Exception:
            iterable = text.replace(",", " ").split()
    else:
        return []

    weights: List[float] = []
    for token in iterable:
        try:
            weights.append(float(token))
        except Exception:
            continue
    return weights


def safe_int(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    try:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return int(float(value))
    except Exception:
        return None


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


def compute_positions_and_weights(row: pd.Series) -> Tuple[List[float], List[float]]:
    ranked_nodes = parse_ranked_nodes(row.get("ranked_nodes"))
    ranked_map = parse_ranked_map(row.get("ranked_map"))
    if not ranked_nodes or not ranked_map:
        return [], []
    total_nodes = safe_int(row.get("total_nodes")) or safe_int(row.get("num_nodes"))
    if total_nodes is None or total_nodes <= 0:
        total_nodes = max(ranked_nodes) + 1 if ranked_nodes else 0
    if total_nodes <= 0:
        return [], []

    limit = min(len(ranked_nodes), len(ranked_map))
    if limit == 0:
        return [], []

    positions: List[float] = []
    weights: List[float] = []
    denominator = max(total_nodes - 1, 1)
    for node, weight in zip(ranked_nodes[:limit], ranked_map[:limit]):
        position = max(0.0, min(1.0, float(node) / float(denominator)))
        weight_value = max(0.0, float(weight))
        positions.append(position)
        weights.append(weight_value)
    return positions, weights


@dataclass
class SampleBucket:
    positions: List[float]
    weights: List[float]

    def __init__(self) -> None:
        self.positions = []
        self.weights = []

    def extend(self, positions: Iterable[float], weights: Iterable[float]) -> None:
        self.positions.extend(positions)
        self.weights.extend(weights)

    @property
    def count(self) -> int:
        return len(self.positions)

    def has_samples(self) -> bool:
        return self.count > 0 and any(w > 0 for w in self.weights)


def compute_density(
    bucket: SampleBucket,
    *,
    grid: np.ndarray,
    bandwidth: float = MIN_BANDWIDTH,
) -> np.ndarray:
    if not bucket.has_samples():
        return np.zeros_like(grid)

    values = np.asarray(bucket.positions, dtype=float)
    weights = np.asarray(bucket.weights, dtype=float)
    weights = np.clip(weights, 0.0, None)
    if weights.sum() <= 0 or np.allclose(weights, 0):
        weights = np.full_like(values, 1.0 / len(values))
    else:
        weights = weights / weights.sum()

    try:
        kde = gaussian_kde(values, weights=weights)
        density = kde(grid)
    except Exception:
        # Fallback: sum of Gaussian bumps with fixed bandwidth
        bw = max(bandwidth, np.std(values) * 0.15)
        bw = max(bw, MIN_BANDWIDTH)
        norm_const = 1.0 / (bw * math.sqrt(2 * math.pi))
        diffs = (grid[None, :] - values[:, None]) / bw
        kernel = np.exp(-0.5 * diffs**2) * norm_const
        density = (weights[:, None] * kernel).sum(axis=0)

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
            xaxis=dict(title="Token position (0 = beginning, 1 = end)", range=[0, 1]),
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
    plt.xlabel("Token position (0 = beginning, 1 = end)", fontsize=FONT_SIZE)
    plt.ylabel(y_title, fontsize=FONT_SIZE)
    plt.xlim(0, 1)
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
            xaxis=dict(title="Token position (0 = beginning, 1 = end)", range=[0, 1]),
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
    plt.xlabel("Token position (0 = beginning, 1 = end)", fontsize=FONT_SIZE)
    plt.ylabel("Density difference (correct - incorrect)", fontsize=FONT_SIZE)
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"[kde-diff-fallback] {output_path}")


def build_output_dir(csv_path: Path) -> Path:
    return csv_path.parent / csv_path.stem


def save_density_csv(
    grid: np.ndarray,
    density: np.ndarray,
    output_path: Path,
    *,
    extra_columns: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    data = {"position": grid, "density": density}
    if extra_columns:
        data.update(extra_columns)
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"[data] {output_path}")


def process_sequence_csv(csv_path: Path, grid: np.ndarray) -> None:
    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"[skip] Empty sequence CSV skipped: {csv_path}")
        return

    output_root = build_output_dir(csv_path)
    output_root.mkdir(parents=True, exist_ok=True)

    groups_all: Dict[Tuple[str, Optional[str]], SampleBucket] = defaultdict(SampleBucket)
    diff_groups: Dict[Tuple[str, Optional[str]], Dict[bool, SampleBucket]] = defaultdict(
        lambda: {True: SampleBucket(), False: SampleBucket()}
    )
    correctness_groups: Dict[Tuple[str, Optional[str]], SampleBucket] = defaultdict(SampleBucket)

    rows_iter = tqdm(df.itertuples(index=False), total=len(df), desc=f"{csv_path.stem} rows", leave=False)
    total_rows = 0
    for row in rows_iter:
        row_series = pd.Series(row._asdict())
        positions, weights = compute_positions_and_weights(row_series)
        if not positions:
            continue
        total_rows += 1
        label = row_series.get("label")
        correctness = parse_correctness(row_series.get("is_correct"), label, row_series.get("prediction_class"))

        groups_all[("overall", None)].extend(positions, weights)
        if label is not None and not (isinstance(label, float) and math.isnan(label)):
            groups_all[("class", str(label))].extend(positions, weights)

        if correctness is not None:
            correctness_groups[("correct", "true" if correctness else "false")].extend(positions, weights)
            if label is not None and not (isinstance(label, float) and math.isnan(label)):
                key = ("class_correct", f"{label}_{'true' if correctness else 'false'}")
                correctness_groups[key].extend(positions, weights)
            diff_groups[("overall", None)][correctness].extend(positions, weights)
            if label is not None and not (isinstance(label, float) and math.isnan(label)):
                diff_groups[("class", str(label))][correctness].extend(positions, weights)

    if total_rows == 0:
        print(f"[skip] No ranked tokens found in {csv_path}")
        return

    # Generate plots for overall & class groups
    for (group_type, identifier), bucket in groups_all.items():
        if not bucket.has_samples():
            continue
        slug_id = "" if identifier is None else f"_{slugify(identifier)}"
        if group_type == "overall":
            title = "All Tokens"
            file_stub = "kde_overall"
        elif group_type == "class":
            title = f"Class {identifier}"
            file_stub = f"kde_class{slug_id}"
        else:
            continue

        density = compute_density(bucket, grid=grid)
        plot_density_single(
            grid,
            density,
            output_path=output_root / f"{file_stub}.pdf",
            title=title,
            color="#636EFA",
        )
        save_density_csv(grid, density, output_root / f"density_{file_stub}.csv")

    # Generate correctness-only plots
    for (group_type, identifier), bucket in correctness_groups.items():
        if not bucket.has_samples():
            continue
        if group_type == "correct":
            title = f"Tokens ({identifier})"
            file_stub = f"kde_correct_{identifier}"
            color = COLOR_CORRECT if identifier == "true" else COLOR_INCORRECT
        elif group_type == "class_correct":
            label, correctness = identifier.split("_", 1)
            title = f"Class {label} ({correctness})"
            file_stub = f"kde_class_{slugify(label)}_correct_{correctness}"
            color = COLOR_CORRECT if correctness == "true" else COLOR_INCORRECT
        else:
            continue

        density = compute_density(bucket, grid=grid)
        plot_density_single(
            grid,
            density,
            output_path=output_root / f"{file_stub}.pdf",
            title=title,
            color=color,
        )
        save_density_csv(grid, density, output_root / f"density_{file_stub}.csv")

    # Generate difference plots and CSVs
    for (group_type, identifier), buckets in diff_groups.items():
        bucket_true = buckets[True]
        bucket_false = buckets[False]
        if not bucket_true.has_samples() or not bucket_false.has_samples():
            continue
        if group_type == "overall":
            title = "All Tokens (Correct - Incorrect)"
            file_stub = "kde_correct_vs_incorrect"
        elif group_type == "class":
            title = f"Class {identifier} (Correct - Incorrect)"
            file_stub = f"kde_class_{slugify(identifier)}_correct_vs_incorrect"
        else:
            continue

        density_true = compute_density(bucket_true, grid=grid)
        density_false = compute_density(bucket_false, grid=grid)

        plot_density_difference(
            grid,
            density_true,
            density_false,
            output_path=output_root / f"{file_stub}.pdf",
            title=title,
        )
        density_diff = density_true - density_false
        save_density_csv(
            grid,
            density_diff,
            output_root / f"density_{file_stub}.csv",
            extra_columns={
                "density_correct": density_true,
                "density_incorrect": density_false,
            },
        )

        cdf_true = np.cumsum(density_true)
        cdf_false = np.cumsum(density_false)
        cdf_true = cdf_true / max(cdf_true[-1], 1e-9)
        cdf_false = cdf_false / max(cdf_false[-1], 1e-9)

        kl_true_false = compute_kl_divergence(density_true, density_false)
        kl_false_true = compute_kl_divergence(density_false, density_true)
        js_div = compute_js_divergence(density_true, density_false)
        ks_stat = compute_ks_statistic(cdf_true, cdf_false)

        metrics_path = output_root / f"metrics_{file_stub}.json"
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


def discover_sequence_csvs(root: Path) -> List[Path]:
    candidates: List[Path] = []
    for path in sorted(root.glob("**/*.csv")):
        try:
            relative_parts = path.relative_to(root).parts
        except ValueError:
            continue
        if len(relative_parts) != 3:
            continue
        candidates.append(path)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate token-position KDE analytics.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=BASE_INPUT_ROOT,
        help="Directory containing per-dataset sequence analytics CSV files.",
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

    grid = np.linspace(0.0, 1.0, args.grid_points)
    csv_paths = discover_sequence_csvs(base_dir)
    if not csv_paths:
        print("No sequence CSV files found.")
        return

    for csv_path in tqdm(csv_paths, desc="Generating KDEs"):
        try:
            process_sequence_csv(csv_path, grid)
        except Exception as exc:  # pragma: no cover - logging only
            print(f"[warn] failed to process {csv_path}: {exc}")


if __name__ == "__main__":
    main()
