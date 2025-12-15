import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm


def slugify(value: str) -> str:
    safe = value.replace("/", "_").replace(" ", "_").replace("-", "_")
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
    """Locate per-explanation TokenSHAP pickles under the provided root."""
    pattern = "**/graph_*.pkl"
    return sorted(base_dir.glob(pattern))


def load_payload(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handler:
        payload = pickle.load(handler)
    if isinstance(payload, dict):
        return payload
    if hasattr(payload, "__dict__"):
        return dict(payload.__dict__)
    raise ValueError(f"Unsupported pickle payload at {path}")


def _normalise_sequence(raw: Optional[Iterable[Any]]) -> List[float]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        iterable = raw
    else:
        return []
    result: List[float] = []
    for value in iterable:
        try:
            result.append(float(value))
        except Exception:
            continue
    return result


def extract_progressions(payload: Mapping[str, Any]) -> Tuple[List[float], List[float], List[float], List[float]]:
    related = payload.get("related_prediction") or {}
    if not isinstance(related, Mapping):
        related = {}

    maskout_conf = payload.get("maskout_progression_confidence")
    if maskout_conf is None:
        maskout_conf = related.get("maskout_progression_confidence")

    maskout_drop = payload.get("maskout_progression_drop")
    if maskout_drop is None:
        maskout_drop = related.get("maskout_progression_drop")

    suff_conf = payload.get("sufficiency_progression_confidence")
    if suff_conf is None:
        suff_conf = related.get("sufficiency_progression_confidence")

    suff_drop = payload.get("sufficiency_progression_drop")
    if suff_drop is None:
        suff_drop = related.get("sufficiency_progression_drop")

    return (
        _normalise_sequence(maskout_conf),
        _normalise_sequence(maskout_drop),
        _normalise_sequence(suff_conf),
        _normalise_sequence(suff_drop),
    )


def format_sequence(values: List[float]) -> str:
    if not values:
        return ""
    return json.dumps(values)


def build_record(payload: Dict[str, Any]) -> Dict[str, Any]:
    dataset_backbone = payload.get("dataset") or ""
    backbone, dataset_raw, dataset = _parse_dataset_fields(dataset_backbone)
    graph_type = payload.get("graph_type") or "tokens"
    method = payload.get("method") or "token_shap_llm"

    maskout_conf, maskout_drop, suff_conf, suff_drop = extract_progressions(payload)

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
        "maskout_progression_confidence": format_sequence(maskout_conf),
        "maskout_progression_confidence_len": len(maskout_conf),
        "maskout_progression_drop": format_sequence(maskout_drop),
        "maskout_progression_drop_len": len(maskout_drop),
        "sufficiency_progression_confidence": format_sequence(suff_conf),
        "sufficiency_progression_confidence_len": len(suff_conf),
        "sufficiency_progression_drop": format_sequence(suff_drop),
        "sufficiency_progression_drop_len": len(suff_drop),
    }
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
    parser = argparse.ArgumentParser(description="Aggregate TokenSHAP progression metrics into CSV summaries.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("outputs/insights/news/LLM"),
        help="Root directory containing TokenSHAP outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analytics/progression"),
        help="Directory where progression CSV files will be written.",
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
            f"\nCompleted TokenSHAP progression aggregation for {len(written)} CSV file(s) ({total_rows} total rows)."
        )
    else:
        print("\nNo TokenSHAP progression CSV files were generated.")


if __name__ == "__main__":
    main()
