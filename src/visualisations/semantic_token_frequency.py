"\"\"\"Visualisations for token frequency across explanations.\"\"\""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

DEFAULT_OUTPUT_ROOT = Path("outputs/analytics/token/frequency")


def _iter_token_csvs(root: Path, pattern: str = "*tokens.csv") -> Iterable[Path]:
    return sorted(root.rglob(pattern))


def _load_tokens(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "token" not in frame.columns:
        raise ValueError(f"'token' column missing in {path}")
    return frame


def _group_by_label(frame: pd.DataFrame) -> Dict[Optional[str], pd.DataFrame]:
    if "label" not in frame.columns:
        return {None: frame}
    groups: Dict[Optional[str], pd.DataFrame] = {}
    for label, subset in frame.groupby("label", dropna=False):
        if pd.isna(label):
            groups[None] = subset
        else:
            groups[str(label)] = subset
    return groups


def _label_suffix(label: Optional[str]) -> str:
    if label is None:
        return ""
    safe = str(label).strip()
    if not safe or safe.lower() in {"nan", "none"}:
        return ""
    return f"_label_{safe.replace('/', '_').replace(' ', '_')}"


def _top_tokens(frame: pd.DataFrame, top_k: int) -> pd.DataFrame:
    counts = frame["token"].value_counts().head(top_k)
    return pd.DataFrame({"token": counts.index, "count": counts.values})


def _plot_top_tokens_bar(summary: pd.DataFrame, title: str, output_path: Path, *, dpi: int = 300) -> Path:
    if summary.empty:
        return output_path
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(6, max(3, 0.3 * len(summary))))
    data = summary.iloc[::-1]
    ax.barh(data["token"], data["count"], color="#4C72B0", alpha=0.85)
    ax.set_xlabel("Frequency (token highlighted)")
    ax.set_ylabel("Token")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_token_frequency_charts(
    tokens_root: Path | str,
    output_root: Path | str | None = DEFAULT_OUTPUT_ROOT,
    *,
    pattern: str = "*tokens.csv",
    top_k: int = 30,
    dpi: int = 300,
) -> List[Path]:
    root = Path(tokens_root)
    out_dir = Path(output_root) if output_root else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    produced: List[Path] = []
    for csv_path in _iter_token_csvs(root, pattern):
        frame = _load_tokens(csv_path)
        groups = _group_by_label(frame)
        for label, subset in groups.items():
            summary = _top_tokens(subset, top_k=top_k)
            if summary.empty:
                continue
            if out_dir:
                target_dir = out_dir / csv_path.relative_to(root).parent
                target_dir.mkdir(parents=True, exist_ok=True)
                output_path = target_dir / f"{csv_path.stem}{_label_suffix(label)}_token_frequency.png"
            else:
                output_path = csv_path.parent / f"{csv_path.stem}{_label_suffix(label)}_token_frequency.png"
            produced.append(_plot_top_tokens_bar(summary, f"Top tokens{'' if label is None else f' (label {label})'}", output_path, dpi=dpi))
    return produced
