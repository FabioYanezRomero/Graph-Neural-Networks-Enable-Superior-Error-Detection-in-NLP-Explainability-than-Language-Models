import argparse
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm


def slugify(value: str) -> str:
    safe = value.replace("/", "_").replace(" ", "_")
    safe = safe.replace("-", "_")
    return "".join(char for char in safe if char.isalnum() or char == "_").lower()


def normalize_dataset_name(raw: str) -> str:
    name = raw.replace("_", "-")
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


def discover_pickles(base_dir: Path) -> Sequence[Path]:
    pattern = "**/*split_pickle/graph_*.pkl"
    return sorted(base_dir.glob(pattern))


def load_payload(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handler:
        payload = pickle.load(handler)
    if isinstance(payload, dict):
        return payload
    if hasattr(payload, "__dict__"):
        return dict(payload.__dict__)
    raise ValueError(f"Unsupported pickle payload at {path}")


def extract_contrastivities(payload: Mapping[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    related = payload.get("related_prediction") or {}
    if not isinstance(related, Mapping):
        related = {}

    origin = payload.get("origin_contrastivity")
    if origin is None:
        origin = related.get("origin_contrastivity")

    masked = payload.get("masked_contrastivity")
    if masked is None:
        masked = related.get("masked_contrastivity")

    maskout = payload.get("maskout_contrastivity")
    if maskout is None:
        maskout = related.get("maskout_contrastivity")

    try:
        origin = float(origin) if origin is not None else None
    except Exception:
        origin = None
    try:
        masked = float(masked) if masked is not None else None
    except Exception:
        masked = None
    try:
        maskout = float(maskout) if maskout is not None else None
    except Exception:
        maskout = None

    return origin, masked, maskout


def _safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None:
        return None
    if abs(denominator) < 1e-9:
        return None
    return numerator / denominator


def _compute_margin_coherence(
    baseline_margin: Optional[float],
    preservation_sufficiency: Optional[float],
    preservation_necessity: Optional[float],
) -> Optional[float]:
    if preservation_sufficiency is None or preservation_necessity is None:
        return None
    reference = max(abs(baseline_margin) if baseline_margin is not None else 0.0, 0.1)
    diff = abs(preservation_sufficiency - (-preservation_necessity))
    return 1.0 - diff / reference


def _compute_margin_decomposition(
    preservation_sufficiency: Optional[float],
    preservation_necessity: Optional[float],
) -> Optional[float]:
    if preservation_sufficiency is None or preservation_necessity is None:
        return None
    total = abs(preservation_sufficiency) + abs(preservation_necessity)
    if total < 1e-9:
        return None
    return abs(preservation_sufficiency) / total


def compute_decision_margin_metrics(
    baseline_margin: Optional[float],
    preservation_sufficiency: Optional[float],
    preservation_necessity: Optional[float],
) -> Dict[str, Optional[float]]:
    suff_ratio = _safe_ratio(preservation_sufficiency, baseline_margin)
    nec_ratio = _safe_ratio(preservation_necessity, baseline_margin)
    coherence = _compute_margin_coherence(baseline_margin, preservation_sufficiency, preservation_necessity)
    decomposition = _compute_margin_decomposition(preservation_sufficiency, preservation_necessity)
    consistency_flag: Optional[float] = None
    if suff_ratio is not None and nec_ratio is not None:
        consistency_flag = float(suff_ratio > 0.7 and nec_ratio < 0.0)

    return {
        "sufficiency_ratio": suff_ratio,
        "necessity_ratio": nec_ratio,
        "margin_coherence": coherence,
        "consistency_flag": consistency_flag,
        "margin_decomposition_ratio": decomposition,
    }


def build_record(payload: Dict[str, Any]) -> Dict[str, Any]:
    dataset_backbone = payload.get("dataset") or ""
    backbone, dataset_raw, dataset = _parse_dataset_fields(dataset_backbone)
    graph_type = payload.get("graph_type") or "tokens"
    method = payload.get("method") or "token_shap_llm"

    baseline_margin, preservation_sufficiency, preservation_necessity = extract_contrastivities(payload)

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
        "baseline_margin": baseline_margin,
        "preservation_sufficiency": preservation_sufficiency,
        "preservation_necessity": preservation_necessity,
    }
    record.update(
        compute_decision_margin_metrics(
            baseline_margin=baseline_margin,
            preservation_sufficiency=preservation_sufficiency,
            preservation_necessity=preservation_necessity,
        )
    )
    return record


def process_pickles(
    base_dir: Path,
    output_dir: Path,
    *,
    limit: Optional[int] = None,
) -> Sequence[Path]:
    pickle_paths = discover_pickles(base_dir)
    if not pickle_paths:
        raise FileNotFoundError(f"No TokenSHAP pickles found under {base_dir}")
    if limit is not None:
        pickle_paths = pickle_paths[:limit]

    grouped: Dict[Tuple[str, str], list[Dict[str, Any]]] = defaultdict(list)
    for path in tqdm(pickle_paths, desc="Scanning TokenSHAP pickles", leave=False, colour="green"):
        try:
            payload = load_payload(path)
            record = build_record(payload)
        except Exception as exc:
            print(f"! Skipping {path} ({exc})")
            continue
        key = (record["dataset_backbone"] or record["dataset"], record["graph_type"])
        grouped[key].append(record)

    written_paths = []
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
    parser = argparse.ArgumentParser(description="Aggregate TokenSHAP consistency metrics into CSV summaries.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("outputs/insights/news/LLM"),
        help="Root directory containing TokenSHAP outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analytics/consistency"),
        help="Directory where consistency CSV files will be written.",
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
            f"\nCompleted TokenSHAP consistency aggregation for {len(written)} CSV file(s) ({total_rows} total rows)."
    )
    else:
        print("\nNo TokenSHAP consistency CSV files were generated.")


if __name__ == "__main__":
    main()
