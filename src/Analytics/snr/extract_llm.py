import argparse
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

EPS = 1e-9


def slugify(value: str) -> str:
    safe = value.replace("/", "_").replace(" ", "_")
    safe = safe.replace("-", "_")
    return "".join(char for char in safe if char.isalnum() or char == "_").lower()


def normalize_dataset_name(dataset_raw: str) -> str:
    name = dataset_raw.replace("_", "-")
    if name.lower() == "sst2":
        return "sst-2"
    return name


def _parse_dataset_fields(dataset_backbone: Optional[str]) -> Tuple[str, str, str]:
    if not dataset_backbone:
        return "", "", ""
    parts = dataset_backbone.split("/")
    if len(parts) >= 2:
        backbone = parts[0]
        dataset_raw = parts[-1]
    else:
        backbone = ""
        dataset_raw = parts[0]
    dataset = normalize_dataset_name(dataset_raw)
    return backbone, dataset_raw, dataset


def discover_pickles(base_dir: Path) -> List[Path]:
    pattern = "**/*split_pickle/graph_*.pkl"
    return sorted(base_dir.glob(pattern))


def load_payload(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handler:
        payload = pickle.load(handler)
    if isinstance(payload, dict):
        return payload
    if hasattr(payload, "__dict"):
        return dict(payload.__dict__)
    raise ValueError(f"Unsupported pickle payload at {path}")


def _safe_iterable(values: Optional[Iterable[Any]]) -> List[Any]:
    if values is None:
        return []
    if isinstance(values, (list, tuple, range)):
        return list(values)
    if isinstance(values, np.ndarray):
        return values.tolist()
    try:
        return list(values)
    except Exception:
        return []


def compute_signal_noise(
    payload: Mapping[str, Any]
) -> Tuple[float, float, float, float, int, int, str]:
    """Extract signal/noise statistics and provenance from a TokenSHAP payload."""

    top_word_scores = _safe_iterable(payload.get("top_word_scores"))
    if not top_word_scores:
        return 0.0, 0.0, 0.0, 0.0, 0, 0, "missing_scores"

    scores = np.asarray(top_word_scores, dtype=float)
    scores = np.abs(scores)
    if scores.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0, 0, "missing_scores"

    top_tokens = _safe_iterable(payload.get("top_tokens"))
    if top_tokens:
        k = max(1, min(len(top_tokens), scores.size))
        top_k_source = "explicit_tokens"
    else:
        sparsity = payload.get("sparsity")
        try:
            sparsity = float(sparsity) if sparsity is not None else 0.2
        except Exception:
            sparsity = 0.2
        k = max(1, int(round(scores.size * max(min(sparsity, 1.0), 0.0))))
        top_k_source = "sparsity_based"

    if k >= scores.size:
        k = max(1, scores.size // 2)
        if k == 0:
            k = 1
        top_k_source = "auto_default"

    signal_vals = scores[:k]
    noise_vals = scores[k:]

    if noise_vals.size == 0:
        half = max(1, signal_vals.size // 2)
        noise_vals = signal_vals[half:].copy()
        signal_vals = signal_vals[:half].copy()
        if noise_vals.size == 0:
            noise_vals = signal_vals.copy()

    signal_mean = float(signal_vals.mean()) if signal_vals.size else 0.0
    noise_mean = float(noise_vals.mean()) if noise_vals.size else 0.0
    signal_std = float(signal_vals.std(ddof=0)) if signal_vals.size else 0.0
    noise_std = float(noise_vals.std(ddof=0)) if noise_vals.size else 0.0

    return (
        signal_mean,
        signal_std,
        noise_mean,
        noise_std,
        int(signal_vals.size),
        int(noise_vals.size),
        top_k_source,
    )


def compute_instance_snr(signal_mean: float, noise_mean: float) -> Tuple[float, float]:
    if signal_mean <= 0.0:
        return 0.0, -np.inf
    denom = max(noise_mean, EPS)
    snr_linear = float(signal_mean / denom)
    snr_db = 20.0 * float(np.log10(max(snr_linear, EPS)))
    return snr_linear, snr_db


def build_record(payload: Dict[str, Any]) -> Dict[str, Any]:
    dataset_backbone = payload.get("dataset") or ""
    backbone, dataset_raw, dataset = _parse_dataset_fields(dataset_backbone)
    graph_type = payload.get("graph_type") or "tokens"
    method = payload.get("method") or "token_shap_llm"

    (
        signal_mean,
        signal_std,
        noise_mean,
        noise_std,
        signal_count,
        noise_count,
        top_k_source,
    ) = compute_signal_noise(payload)
    snr_linear, snr_db = compute_instance_snr(signal_mean, noise_mean)

    record: Dict[str, Any] = {
        "method": method,
        "backbone": backbone,
        "dataset": dataset,
        "dataset_raw": dataset_raw,
        "dataset_backbone": dataset_backbone,
        "graph_type": graph_type,
        "run_id": payload.get("run_id"),
        "graph_index": payload.get("graph_index"),
        "global_graph_index": payload.get("global_graph_index"),
        "label": payload.get("label"),
        "prediction_class": payload.get("prediction_class"),
        "prediction_confidence": payload.get("prediction_confidence"),
        "is_correct": payload.get("is_correct"),
        "signal_mean": signal_mean,
        "signal_std": signal_std,
        "noise_mean": noise_mean,
        "noise_std": noise_std,
        "signal_count": signal_count,
        "noise_count": noise_count,
        "signal_minus_noise": float(signal_mean - noise_mean),
        "snr_linear": snr_linear,
        "snr_db": snr_db,
        "top_k_source": top_k_source,
        "signal_k": signal_count,
        "noise_k": noise_count,
        "aggregated_from_coalitions": False,
    }
    return record


def process_pickles(
    base_dir: Path,
    output_dir: Path,
    *,
    limit: Optional[int] = None,
) -> List[Path]:
    pickle_paths = discover_pickles(base_dir)
    if not pickle_paths:
        raise FileNotFoundError(f"No TokenSHAP pickles found under {base_dir}")

    if limit is not None:
        pickle_paths = pickle_paths[:limit]

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for path in tqdm(pickle_paths, desc="Scanning TokenSHAP pickles", leave=False, colour="green"):
        try:
            payload = load_payload(path)
            record = build_record(payload)
        except Exception as exc:
            print(f"! Skipping {path} ({exc})")
            continue
        key = (record["dataset_backbone"] or record["dataset"], record["graph_type"])
        grouped[key].append(record)

    written_paths: List[Path] = []
    for (dataset_backbone, graph_type), rows in tqdm(
        sorted(grouped.items()),
        desc="Writing CSVs",
        leave=False,
        colour="cyan",
    ):
        if not rows:
            continue
        method = rows[0].get("method") or "token_shap_llm"
        dataset_slug = slugify(dataset_backbone)
        graph_slug = slugify(graph_type)
        target_dir = output_dir / method / dataset_slug
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{graph_slug}.csv"

        df = pd.DataFrame(rows)
        sort_columns = [column for column in ("run_id", "graph_index") if column in df.columns]
        if sort_columns:
            df.sort_values(sort_columns, inplace=True, kind="mergesort")
        df.to_csv(target_path, index=False)
        written_paths.append(target_path)
        print(f"✓ {method} | {dataset_backbone} | {graph_type} → {target_path} ({len(df)} rows)")

    return written_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract SNR analytics from TokenSHAP explanation payloads.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("outputs/insights/news/LLM"),
        help="Root directory containing LLM explanation outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analytics/snr"),
        help="Directory where SNR CSV files will be written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of pickles processed.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    base_dir: Path = args.base_dir
    output_dir: Path = args.output_dir

    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    written = process_pickles(base_dir, output_dir, limit=args.limit)

    if written:
        total_rows = 0
        for path in written:
            try:
                df = pd.read_csv(path)
                total_rows += len(df)
            except Exception:
                continue
        print(
            f"\nCompleted TokenSHAP SNR extraction for {len(written)} CSV file(s) ({total_rows} total rows)."
        )
    else:
        print("\nNo TokenSHAP SNR CSV files were generated.")


if __name__ == "__main__":
    main()
