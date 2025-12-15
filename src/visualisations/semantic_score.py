"\"\"\"Score-based semantic visualisations.\"\"\""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from Analytics import semantic_analysis as _semantic_analysis  # type: ignore
except Exception:  # pragma: no cover - optional dependency during visualisation-only usage
    _semantic_analysis = None

DEFAULT_DENSITY_ROOT = Path("outputs/analytics/score/density")
DEFAULT_DIFFERENCE_ROOT = Path("outputs/analytics/score/difference")
DEFAULT_POSITION_DIFFERENCE_ROOT = Path("outputs/analytics/position/difference")
DEFAULT_RANKING_ROOT = Path("outputs/analytics/score/ranking")


_FALLBACK_STOPWORDS: Set[str] = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "while",
    "for",
    "to",
    "of",
    "in",
    "on",
    "at",
    "by",
    "with",
    "from",
    "as",
    "that",
    "this",
    "these",
    "those",
    "is",
    "am",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "it",
    "its",
    "he",
    "she",
    "they",
    "we",
    "you",
    "i",
    "me",
    "him",
    "her",
    "them",
    "us",
    "my",
    "your",
    "our",
    "their",
    "not",
    "no",
    "nor",
    "so",
    "than",
    "then",
    "too",
    "very",
    "can",
    "could",
    "should",
    "would",
    "may",
    "might",
    "will",
    "shall",
    "do",
    "does",
    "did",
    "doing",
    "done",
    "have",
    "has",
    "had",
    "having",
    "there",
    "here",
    "also",
    "just",
    "only",
    "over",
    "under",
    "up",
    "down",
    "out",
    "into",
    "about",
    "after",
    "before",
    "between",
    "more",
    "most",
    "less",
    "least",
    "any",
    "some",
    "such",
    "each",
    "other",
    "both",
    "all",
    "many",
    "much",
    "few",
    "several",
}


def _resolve_stopwords(custom_stopwords: Optional[Iterable[str]] = None) -> Set[str]:
    """Return the stopword inventory to filter vocabulary comparisons."""
    if custom_stopwords is not None:
        return {str(item).strip().lower() for item in custom_stopwords if isinstance(item, str)}

    resolved: Set[str] = set()
    if _semantic_analysis is not None:
        try:
            resolved.update({str(item).strip().lower() for item in _semantic_analysis._default_stopwords()})  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - defensive fallback
            pass
        global_stopwords = getattr(_semantic_analysis, "_GLOBAL_STOPWORDS", None)
        if isinstance(global_stopwords, (set, list, tuple)):
            resolved.update({str(item).strip().lower() for item in global_stopwords if isinstance(item, str)})
    if not resolved:
        resolved.update(_FALLBACK_STOPWORDS)
    return resolved


def _normalise_token(token: object, stopwords: Set[str]) -> Optional[str]:
    """Normalise a token for vocabulary overlap calculations."""
    if token is None or (isinstance(token, float) and np.isnan(token)):
        return None
    text = str(token).strip()
    if not text:
        return None
    lower = text.lower()
    if lower in stopwords:
        return None
    return lower


def _coerce_bool(value: object) -> bool:
    """Robust boolean coercion for heterogeneous dataframe columns."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _extract_token_sets(frame: pd.DataFrame, stopwords: Set[str]) -> Tuple[Set[str], Set[str]]:
    """Split a token dataframe into correct/incorrect vocabularies."""
    if "token" not in frame.columns or "is_correct" not in frame.columns:
        return set(), set()

    normalised = frame["token"].map(lambda tok: _normalise_token(tok, stopwords))
    valid_mask = normalised.notna()
    if not valid_mask.any():
        return set(), set()

    subset = frame.loc[valid_mask, ["is_correct"]].copy()
    subset["token"] = normalised[valid_mask]
    subset["is_correct"] = subset["is_correct"].map(_coerce_bool)

    correct_tokens = set(subset.loc[subset["is_correct"], "token"])
    incorrect_tokens = set(subset.loc[~subset["is_correct"], "token"])
    return correct_tokens, incorrect_tokens


def _vocabulary_overlap_metrics(frame: pd.DataFrame, stopwords: Set[str]) -> Optional[Dict[str, object]]:
    """Compute overlap metrics between correct and incorrect vocabularies."""
    correct_tokens, incorrect_tokens = _extract_token_sets(frame, stopwords)
    if not correct_tokens and not incorrect_tokens:
        return None

    intersection = correct_tokens & incorrect_tokens
    union = correct_tokens | incorrect_tokens

    size_correct = len(correct_tokens)
    size_incorrect = len(incorrect_tokens)
    size_shared = len(intersection)
    size_union = len(union)

    if size_union == 0:
        return None

    jaccard = size_shared / size_union if size_union else 0.0
    denominator = size_correct + size_incorrect
    dice = (2.0 * size_shared / denominator) if denominator else 0.0
    overlap_coeff = size_shared / min(size_correct, size_incorrect) if size_correct and size_incorrect else 0.0
    correct_ratio = size_shared / size_correct if size_correct else 0.0
    incorrect_ratio = size_shared / size_incorrect if size_incorrect else 0.0
    shared_sample = ", ".join(sorted(intersection)[:15])

    return {
        "correct_vocab_size": size_correct,
        "incorrect_vocab_size": size_incorrect,
        "total_vocab_size": size_union,
        "shared_vocab_size": size_shared,
        "correct_unique_vocab_size": size_correct - size_shared,
        "incorrect_unique_vocab_size": size_incorrect - size_shared,
        "jaccard_similarity": float(jaccard),
        "dice_coefficient": float(dice),
        "overlap_coefficient": float(overlap_coeff),
        "correct_overlap_ratio": float(correct_ratio),
        "incorrect_overlap_ratio": float(incorrect_ratio),
        "shared_tokens_sample": shared_sample,
    }


def _iter_token_csvs(root: Path, pattern: str = "*tokens.csv") -> Iterable[Path]:
    return sorted(root.rglob(pattern))


def _load_tokens(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "score" not in frame.columns:
        raise ValueError(f"'score' column missing in {path}")
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


def _ensure_dir(base: Optional[Path], csv_path: Path, root: Path) -> Path:
    if base is None:
        return csv_path.parent
    target = base / csv_path.relative_to(root).parent
    target.mkdir(parents=True, exist_ok=True)
    return target


def plot_token_score_density(series_or_path, output_path: Path | None = None, *, title: Optional[str] = None, dpi: int = 300) -> Path | None:
    if isinstance(series_or_path, (str, Path)):
        csv_file = Path(series_or_path)
        frame = _load_tokens(csv_file)
        series = frame["score"].dropna().astype(float).clip(0.0, 1.0)
        if series.empty:
            return None
        output_file = Path(output_path) if output_path else csv_file.with_suffix(".png")
        out_title = title or csv_file.stem.replace("_", " ").title()
    else:
        series = pd.Series(series_or_path).dropna().astype(float).clip(0.0, 1.0)
        if series.empty:
            return None
        if output_path is None:
            raise ValueError("output_path must be provided when passing raw series")
        output_file = Path(output_path)
        out_title = title or output_file.stem.replace("_", " ").title()

    if series.empty:
        return None
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(6, 4))
    grid = np.linspace(0.0, 1.0, 256)
    density = _kde(series.to_numpy(), grid)
    ax.plot(grid, density, color="#4C72B0", linewidth=1.8)
    ax.fill_between(grid, density, alpha=0.2, color="#4C72B0")
    ax.set_xlim(0.0, 1.0)
    ax.margins(x=0)
    ax.set_xlabel("Token score (scaled 0–1)")
    ax.set_ylabel("Density")
    ax.set_title(out_title)
    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_file


def generate_token_score_densities(
    tokens_root: Path | str,
    output_root: Path | str | None = DEFAULT_DENSITY_ROOT,
    *,
    pattern: str = "*tokens.csv",
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
            target_dir = _ensure_dir(out_dir, csv_path, root)
            output_path = target_dir / f"{csv_path.stem}{_label_suffix(label)}_score_density.png"
            result = plot_token_score_density(subset["score"], output_path=output_path, title=f"Score density{'' if label is None else f' (label {label})'}", dpi=dpi)
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


def _group_correct_incorrect(root: Path, paths: Iterable[Path]) -> Dict[str, Dict[str, Path]]:
    groups: Dict[str, Dict[str, Path]] = defaultdict(dict)
    for path in paths:
        try:
            relative = path.relative_to(root)
        except ValueError:
            relative = path
        stem = relative.stem
        parent = relative.parent
        label_key: Optional[str] = None
        if stem.endswith("_correct"):
            label_key = str((parent / stem[: -len("_correct")]).as_posix())
            groups[label_key]["correct"] = path
        elif stem.endswith("_incorrect"):
            label_key = str((parent / stem[: -len("_incorrect")]).as_posix())
            groups[label_key]["incorrect"] = path
        elif stem.endswith("_tokens"):
            core = stem[: -len("_tokens")]
            if core.endswith("_correct"):
                label_key = str((parent / core[: -len("_correct")]).as_posix())
                groups[label_key]["correct"] = path
            elif core.endswith("_incorrect"):
                label_key = str((parent / core[: -len("_incorrect")]).as_posix())
                groups[label_key]["incorrect"] = path
    return groups


def _scope_labels(stem: str) -> Tuple[str, str]:
    core = stem
    if core.startswith("tokens_"):
        core = core[len("tokens_") :]
    if core.startswith("summary_"):
        core = core[len("summary_") :]
    if core in {"", "tokens"}:
        display = "All"
        key = "all"
    elif core.startswith("class") and core[len("class") :].isdigit():
        idx = int(core[len("class") :])
        display = f"Class {idx + 1}"
        key = f"class{idx + 1}"
    else:
        display = core.replace("_", " ")
        key = core
    return display.title(), key.replace("/", "_")


def _plot_metric_difference(
    correct_values: np.ndarray,
    incorrect_values: np.ndarray,
    *,
    grid: np.ndarray,
    xlabel: str,
    title: str,
    output_file: Path,
    dpi: int,
) -> Tuple[Path, Dict[str, float]]:
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(6, 4))
    difference = _kde(correct_values, grid) - _kde(incorrect_values, grid)
    ax.plot(grid, difference, color="#C44E52", linewidth=1.8)
    ax.axhline(0.0, color="#4C72B0", linestyle="--", linewidth=1.0)
    ax.fill_between(grid, difference, 0.0, where=difference >= 0, color="#C44E52", alpha=0.2)
    ax.fill_between(grid, difference, 0.0, where=difference < 0, color="#55A868", alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density difference (correct - incorrect)")
    ax.set_xlim(grid.min(), grid.max())
    ax.margins(x=0)
    ax.set_title(title)
    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    abs_difference = np.abs(difference)
    abs_area = float(np.trapz(abs_difference, grid))
    std_difference = float(difference.std(ddof=0)) if difference.size else 0.0
    mean_difference = float(difference.mean()) if difference.size else 0.0
    max_abs_difference = float(abs_difference.max()) if difference.size else 0.0

    stats = {
        "area": float(np.trapz(difference, grid)),
        "positive_mass": float(np.trapz(np.clip(difference, 0.0, None), grid)),
        "negative_mass": float(np.trapz(np.clip(-difference, 0.0, None), grid)),
        "max_difference": float(difference.max() if difference.size else 0.0),
        "min_difference": float(difference.min() if difference.size else 0.0),
        "abs_area": abs_area,
        "std_difference": std_difference,
        "mean_difference": mean_difference,
        "max_abs_difference": max_abs_difference,
        "range_difference": float((difference.max() - difference.min()) if difference.size else 0.0),
    }
    return output_file, stats


def plot_token_score_difference(
    correct_csv: Path | str,
    incorrect_csv: Path | str,
    output_path: Path | str | None = None,
    *,
    title: Optional[str] = None,
    dpi: int = 300,
) -> Optional[Tuple[Path, Dict[str, float]]]:
    correct_frame = _load_tokens(Path(correct_csv))
    incorrect_frame = _load_tokens(Path(incorrect_csv))

    correct_values = correct_frame["score"].dropna().astype(float).clip(0.0, 1.0).to_numpy()
    incorrect_values = incorrect_frame["score"].dropna().astype(float).clip(0.0, 1.0).to_numpy()
    if correct_values.size == 0 or incorrect_values.size == 0:
        return None

    grid = np.linspace(0.0, 1.0, 256)
    output_file = Path(output_path) if output_path else Path(correct_csv).with_suffix(".png")
    title_text = title or f"{Path(correct_csv).stem} score difference"
    return _plot_metric_difference(
        correct_values,
        incorrect_values,
        grid=grid,
        xlabel="Token score (scaled 0–1)",
        title=title_text,
        output_file=output_file,
        dpi=dpi,
    )


def generate_token_score_differences(
    tokens_root: Path | str,
    output_root: Path | str | None = DEFAULT_DIFFERENCE_ROOT,
    *,
    pattern: str = "*.csv",
    dpi: int = 300,
) -> List[Path]:
    root = Path(tokens_root)
    out_dir = Path(output_root) if output_root else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    dataset_slug = root.parent.name if root.parent != root else root.name
    groups = _group_correct_incorrect(root, _iter_token_csvs(root, pattern))
    produced: List[Path] = []
    difference_rows: List[Dict[str, object]] = []

    for subset in groups.values():
        correct_path = subset.get("correct")
        incorrect_path = subset.get("incorrect")
        if not correct_path or not incorrect_path:
            continue

        if out_dir:
            relative_parent = correct_path.relative_to(root).parent
            target_dir = out_dir / relative_parent
            target_dir.mkdir(parents=True, exist_ok=True)
        else:
            target_dir = correct_path.parent / "score_differences"
            target_dir.mkdir(parents=True, exist_ok=True)

        stem = correct_path.stem
        if stem.endswith("_correct"):
            stem = stem[: -len("_correct")]
        label, safe_name = _scope_labels(stem)
        target = target_dir / f"{safe_name}_correct_vs_incorrect_score_diff.png"

        result = plot_token_score_difference(
            correct_path,
            incorrect_path,
            output_path=target,
            dpi=dpi,
            title=f"{label} Score (Correct - Incorrect)",
        )
        if result is not None:
            plot_path, stats = result
            produced.append(plot_path)

            record = {
                "dataset": dataset_slug,
                "scope": safe_name,
                "metric": "score",
                **stats,
            }
            difference_rows.append(record)

            target_dir.mkdir(parents=True, exist_ok=True)
            per_scope_path = target_dir / f"{safe_name}_difference_metrics.csv"
            pd.DataFrame([record]).to_csv(per_scope_path, index=False)
            produced.append(per_scope_path)

            for csv_path in (Path(correct_path), Path(incorrect_path)):
                sibling_path = csv_path.with_name(csv_path.stem + "_difference_metrics.csv")
                pd.DataFrame([record]).to_csv(sibling_path, index=False)
                produced.append(sibling_path)

    if difference_rows:
        summary_dir = out_dir if out_dir else root.parent
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / "score_difference_metrics.csv"
        pd.DataFrame(difference_rows).to_csv(summary_path, index=False)
        produced.append(summary_path)

    return produced


def plot_token_position_difference(
    correct_csv: Path | str,
    incorrect_csv: Path | str,
    output_path: Path | str | None = None,
    *,
    title: Optional[str] = None,
    dpi: int = 300,
) -> Optional[Tuple[Path, Dict[str, float]]]:
    correct_frame = _load_tokens(Path(correct_csv))
    incorrect_frame = _load_tokens(Path(incorrect_csv))

    correct_values = correct_frame["position"].dropna().astype(float).clip(0.0, 1.0).to_numpy()
    incorrect_values = incorrect_frame["position"].dropna().astype(float).clip(0.0, 1.0).to_numpy()
    if correct_values.size == 0 or incorrect_values.size == 0:
        return None

    grid = np.linspace(0.0, 1.0, 256)
    output_file = Path(output_path) if output_path else Path(correct_csv).with_suffix(".png")
    title_text = title or f"{Path(correct_csv).stem} position difference"
    return _plot_metric_difference(
        correct_values,
        incorrect_values,
        grid=grid,
        xlabel="Token position (0=beginning, 1=end)",
        title=title_text,
        output_file=output_file,
        dpi=dpi,
    )


def generate_token_position_differences(
    tokens_root: Path | str,
    output_root: Path | str | None = DEFAULT_POSITION_DIFFERENCE_ROOT,
    *,
    pattern: str = "*.csv",
    dpi: int = 300,
) -> List[Path]:
    root = Path(tokens_root)
    out_dir = Path(output_root) if output_root else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    produced: List[Path] = []
    correct_incorrect = _group_correct_incorrect(root, _iter_token_csvs(root, pattern))
    dataset_slug = root.parent.name if root.parent != root else root.name
    difference_rows: List[Dict[str, object]] = []

    for entry in correct_incorrect.values():
        correct_path = entry.get("correct")
        incorrect_path = entry.get("incorrect")
        if not correct_path or not incorrect_path:
            continue

        if out_dir:
            relative_parent = correct_path.relative_to(root).parent
            target_dir = out_dir / relative_parent
            target_dir.mkdir(parents=True, exist_ok=True)
        else:
            target_dir = correct_path.parent / "position_differences"
            target_dir.mkdir(parents=True, exist_ok=True)

        stem = correct_path.stem
        if stem.endswith("_correct"):
            stem = stem[: -len("_correct")]
        label, safe_name = _scope_labels(stem)
        target = target_dir / f"{safe_name}_correct_vs_incorrect_position_diff.png"

        result = plot_token_position_difference(
            correct_path,
            incorrect_path,
            output_path=target,
            dpi=dpi,
            title=f"{label} Position (Correct - Incorrect)",
        )
        if result is not None:
            plot_path, stats = result
            produced.append(plot_path)

            record = {
                "dataset": dataset_slug,
                "scope": safe_name,
                "metric": "position",
                **stats,
            }
            difference_rows.append(record)

            target_dir.mkdir(parents=True, exist_ok=True)
            per_scope_path = target_dir / f"{safe_name}_difference_metrics.csv"
            pd.DataFrame([record]).to_csv(per_scope_path, index=False)
            produced.append(per_scope_path)

            for csv_path in (Path(correct_path), Path(incorrect_path)):
                sibling_path = csv_path.with_name(csv_path.stem + "_difference_metrics.csv")
                pd.DataFrame([record]).to_csv(sibling_path, index=False)
                produced.append(sibling_path)

    if difference_rows:
        summary_dir = out_dir if out_dir else root.parent
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / "position_difference_metrics.csv"
        pd.DataFrame(difference_rows).to_csv(summary_path, index=False)
        produced.append(summary_path)

    return produced


def _top_tokens_by_score(frame: pd.DataFrame, top_k: int) -> pd.DataFrame:
    summary = frame.groupby("token")["score"].mean().sort_values(ascending=False).head(top_k)
    return pd.DataFrame({"token": summary.index, "avg_score": summary.values})


def _plot_score_ranking(summary: pd.DataFrame, title: str, output_path: Path, *, dpi: int = 300) -> Path:
    if summary.empty:
        return output_path
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(6, max(3, 0.3 * len(summary))))
    data = summary.iloc[::-1]
    ax.barh(data["token"], data["avg_score"], color="#C44E52", alpha=0.85)
    ax.set_xlabel("Average importance score")
    ax.set_ylabel("Token")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_token_score_ranking(
    tokens_root: Path | str,
    output_root: Path | str | None = DEFAULT_RANKING_ROOT,
    *,
    pattern: str = "*tokens.csv",
    top_k: int = 30,
    dpi: int = 300,
    stopwords: Optional[Iterable[str]] = None,
) -> List[Path]:
    root = Path(tokens_root)
    out_dir = Path(output_root) if output_root else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    stopword_set = _resolve_stopwords(stopwords)
    overlap_summary: Dict[Path, Dict[str, Dict[str, object]]] = defaultdict(dict)
    produced: List[Path] = []
    for csv_path in _iter_token_csvs(root, pattern):
        frame = _load_tokens(csv_path)
        groups = _group_by_label(frame)
        target_dir = _ensure_dir(out_dir, csv_path, root)
        for label, subset in groups.items():
            summary = _top_tokens_by_score(subset, top_k=top_k)
            if summary.empty:
                continue
            output_path = target_dir / f"{csv_path.stem}{_label_suffix(label)}_token_score_ranking.png"
            produced.append(_plot_score_ranking(summary, f"Average token score{'' if label is None else f' (label {label})'}", output_path, dpi=dpi))
        should_collect_overlap = "_correct_" not in csv_path.stem and "_incorrect_" not in csv_path.stem and "_class_" not in csv_path.stem
        if should_collect_overlap:
            overall_metrics = _vocabulary_overlap_metrics(frame, stopword_set)
            if overall_metrics:
                overlap_summary[target_dir]["ALL"] = overall_metrics
            for label, subset in groups.items():
                metrics = _vocabulary_overlap_metrics(subset, stopword_set)
                if metrics:
                    label_key = "UNLABELED" if label is None else str(label)
                    overlap_summary[target_dir][label_key] = metrics

    for directory, label_metrics in overlap_summary.items():
        if not label_metrics:
            continue
        records: List[Dict[str, object]] = []
        for label_key, metrics in sorted(label_metrics.items(), key=lambda item: (item[0] != "ALL", str(item[0]))):
            record = {"label": label_key}
            record.update(metrics)
            records.append(record)
        metrics_frame = pd.DataFrame(records)
        for column in (
            "jaccard_similarity",
            "dice_coefficient",
            "overlap_coefficient",
            "correct_overlap_ratio",
            "incorrect_overlap_ratio",
        ):
            if column in metrics_frame.columns:
                metrics_frame[column] = metrics_frame[column].astype(float)
        metrics_path = directory / "vocabulary_overlap_metrics.csv"
        metrics_frame.to_csv(metrics_path, index=False)
        produced.append(metrics_path)
    return produced
