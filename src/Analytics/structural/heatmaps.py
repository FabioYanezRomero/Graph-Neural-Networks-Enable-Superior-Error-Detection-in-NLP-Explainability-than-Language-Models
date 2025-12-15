from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:  # Plotly is optional (fallback to seaborn if unavailable or failing)
    import plotly.graph_objects as go
    import plotly.io as pio
    _PLOTLY_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    go = pio = None
    _PLOTLY_AVAILABLE = False
import matplotlib.pyplot as plt
import seaborn as sns


CORRELATION_ROOT = Path("outputs/analytics/structural/substructures")
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
FEATURE_ORDER = [
    "avg_degree",
    "max_degree",
    "degree_variance",
    "degree_skewness",
    "n_components",
    "avg_betweenness",
    "max_betweenness",
    "min_betweenness",
    "avg_closeness",
    "max_closeness",
    "num_edges",
    "num_nodes",
]
FONT_SIZE = 6


def discover_correlation_csvs(root: Path) -> list[Path]:
    return sorted(root.glob("**/correlation_*.csv"))


def prepare_matrix(corr_path: Path) -> Optional[pd.DataFrame]:
    df = pd.read_csv(corr_path, index_col=0)
    if df.empty or df.shape[0] < 2 or df.shape[1] < 2:
        return None

    columns = [col for col in FEATURE_ORDER if col in df.columns]
    if not columns:
        return None
    df = df.loc[columns, columns]
    if df.empty or df.shape[0] < 2:
        return None
    return df.fillna(0.0)


def make_heatmap(corr_path: Path, matrix: pd.DataFrame, suffix: str = "") -> None:
    sanitized = matrix
    base_name = corr_path.stem.replace("correlation", "heatmap")
    output_path = corr_path.with_name(base_name + ".pdf")
    title_suffix = suffix.replace("_", " ").strip()
    if title_suffix:
        title_suffix = f" {title_suffix}"

    if _PLOTLY_AVAILABLE:
        try:
            z_values = []
            for i in range(sanitized.shape[0]):
                row = []
                for j in range(sanitized.shape[1]):
                    if j >= i:
                        row.append(sanitized.iloc[i, j])
                    else:
                        row.append(None)
                z_values.append(row)

            fig = go.Figure(
                data=go.Heatmap(
                    z=z_values,
                    x=sanitized.columns,
                    y=sanitized.index,
                    colorscale="RdBu",
                    zmin=-1,
                    zmax=1,
                    colorbar=dict(title=dict(text="ρ"), ticks="outside"),
                    hovertemplate="X: %{x}<br>Y: %{y}<br>ρ: %{z:.2f}<extra></extra>",
                )
            )

            fig.update_layout(
                title=corr_path.stem.replace("_", " ").title() + title_suffix,
                font=dict(size=FONT_SIZE),
                xaxis=dict(side="top"),
                yaxis=dict(autorange="reversed"),
                margin=dict(l=40, r=10, t=60, b=40),
                width=max(400, 120 * sanitized.shape[1]),
                height=max(400, 120 * sanitized.shape[0]),
            )

            fig.write_image(str(output_path), format="pdf")
            print(f"[heatmap] {output_path}")
            return
        except Exception as exc:  # pragma: no cover - fallback to matplotlib
            print(f"[warn] Plotly export failed for {corr_path}: {exc}. Falling back to Matplotlib.")

    # Fallback: Matplotlib/Seaborn
    sns.set_theme(style="white")
    plt.figure(figsize=(max(4, 0.45 * sanitized.shape[1]), max(4, 0.45 * sanitized.shape[0])))
    mask = np.tril(np.ones_like(sanitized, dtype=bool), k=-1)
    ax = sns.heatmap(
        sanitized,
        mask=mask,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar=True,
        square=True,
        annot_kws={"size": FONT_SIZE},
    )
    ax.set_title(corr_path.stem.replace("_", " ").title() + title_suffix, fontsize=FONT_SIZE + 2)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=FONT_SIZE, rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=FONT_SIZE)
    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"[heatmap-fallback] {output_path}")


def main() -> None:
    correlation_files = discover_correlation_csvs(CORRELATION_ROOT)
    if not correlation_files:
        print("No correlation CSV files found. Run linear_correlations.py first.")
        return

    true_matrices: Dict[str, Tuple[pd.DataFrame, Path]] = {}

    for corr_path in correlation_files:
        try:
            matrix = prepare_matrix(corr_path)
            if matrix is None:
                continue
            make_heatmap(corr_path, matrix)

            stem = corr_path.stem
            if "correct_true" in stem:
                base_stem = stem.replace("correct_true", "correct_false")
                true_matrices[base_stem] = (matrix, corr_path)
            elif "correct_false" in stem:
                base_stem = stem
                true_entry = true_matrices.get(base_stem)
                if true_entry is None:
                    true_path = corr_path.with_name(stem.replace("correct_false", "correct_true") + ".csv")
                    true_matrix = prepare_matrix(true_path)
                    if true_matrix is None:
                        continue
                else:
                    true_matrix = true_entry[0]

                common_idx = true_matrix.index.intersection(matrix.index)
                common_cols = true_matrix.columns.intersection(matrix.columns)
                if len(common_idx) >= 2 and len(common_cols) >= 2:
                    diff = true_matrix.loc[common_idx, common_cols] - matrix.loc[common_idx, common_cols]
                    diff = diff.fillna(0.0)
                    diff_path = corr_path.with_name(stem.replace("correct_false", "correct_vs_incorrect") + ".csv")
                    diff.to_csv(diff_path)
                    make_heatmap(diff_path, diff)

        except Exception as exc:  # pragma: no cover - logging only
            print(f"[warn] failed to create heatmap for {corr_path}: {exc}")


if __name__ == "__main__":
    main()
