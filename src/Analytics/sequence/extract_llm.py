import argparse
import pickle
import sys
from collections import defaultdict
from collections.abc import Iterable as IterableABC
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def slugify(value: str) -> str:
    safe = value.replace("/", "_").replace(" ", "_")
    safe = safe.replace("-", "_")
    return "".join(char for char in safe if char.isalnum() or char == "_").lower()


def normalize_dataset_name(dataset_raw: str) -> str:
    name = dataset_raw.replace("_", "-")
    if name.lower() == "sst2":
        return "sst-2"
    return name


def infer_split_from_run_id(run_id: Optional[str]) -> Optional[str]:
    if not run_id:
        return None
    tokens = run_id.lower()
    for candidate in ("train", "test", "validation", "val", "dev"):
        if candidate in tokens:
            if candidate == "val":
                return "validation"
            if candidate == "dev":
                return "development"
            return candidate
    return None


def format_sequence(values: Sequence[Any]) -> str:
    if not values:
        return ""
    return " ".join(str(value) for value in values)


def format_float_sequence(values: Sequence[float]) -> str:
    if not values:
        return ""
    return "[" + ", ".join(f"{value:.6f}" for value in values) + "]"


def format_token_sequence(tokens: Sequence[str], indices: Sequence[int]) -> str:
    if not tokens or not indices:
        return ""
    chosen: List[str] = []
    token_count = len(tokens)
    for index in indices:
        try:
            if 0 <= index < token_count:
                chosen.append(str(tokens[index]))
        except Exception:
            continue
    return " ".join(chosen)


def _safe_iterable(values: Optional[Iterable[Any]]) -> Iterable[Any]:
    if values is None:
        return ()
    if isinstance(values, (list, tuple, range)):
        return values
    if isinstance(values, np.ndarray):
        return values.tolist()
    return list(values)


def extract_ranked_indices(payload: Mapping[str, Any]) -> List[int]:
    ranked: Iterable[Any] = payload.get("top_nodes") or payload.get("ranked_nodes") or ()

    related_prediction = payload.get("related_prediction") or {}
    if not ranked and isinstance(related_prediction, Mapping):
        ranked = related_prediction.get("ranked_nodes") or ()

    if isinstance(ranked, str):
        ranked = ranked.split()

    ranked = _safe_iterable(ranked)

    ranked_indices: List[int] = []
    seen: set[int] = set()
    for value in ranked:
        try:
            node_id = int(value)
        except Exception:
            continue
        if node_id in seen:
            continue
        seen.add(node_id)
        ranked_indices.append(node_id)

    if ranked_indices:
        return ranked_indices

    node_importance = payload.get("node_importance")
    if isinstance(node_importance, np.ndarray):
        node_scores = node_importance.tolist()
    elif isinstance(node_importance, (list, tuple)):
        node_scores = list(node_importance)
    else:
        node_scores = []

    if node_scores:
        ranked_indices = sorted(
            range(len(node_scores)),
            key=lambda idx: node_scores[idx],
            reverse=True,
        )
        return ranked_indices

    coalitions = payload.get("coalitions")
    if isinstance(coalitions, IterableABC):
        scored_nodes: Dict[int, float] = {}
        for entry in coalitions:
            if not isinstance(entry, Mapping):
                continue
            coalition = entry.get("coalition") or entry.get("node_idx") or ()
            weight = entry.get("W") or entry.get("weight") or entry.get("importance")
            try:
                weight_value = float(weight)
            except Exception:
                weight_value = 0.0
            for node in _safe_iterable(coalition):
                try:
                    node_id = int(node)
                except Exception:
                    continue
                if weight_value > scored_nodes.get(node_id, float("-inf")):
                    scored_nodes[node_id] = weight_value
        if scored_nodes:
            ranked_indices = sorted(
                scored_nodes.keys(),
                key=lambda node: scored_nodes[node],
                reverse=True,
            )
    return ranked_indices


def discover_llm_pickles(base_dir: Path) -> List[Path]:
    pattern = "**/*_records_split_pickle/graph_*.pkl"
    return sorted(base_dir.glob(pattern))


def load_pickle_record(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handler:
        payload = pickle.load(handler)
    if isinstance(payload, dict):
        return payload
    if hasattr(payload, "__dict__"):
        return dict(payload.__dict__)
    raise ValueError(f"Unsupported pickle payload at {path}")


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
    dataset_normalized = normalize_dataset_name(dataset_raw)
    return backbone, dataset_raw, dataset_normalized


def build_record(payload: Dict[str, Any]) -> Dict[str, Any]:
    dataset_backbone = payload.get("dataset") or ""
    backbone, dataset_raw, dataset_normalized = _parse_dataset_fields(dataset_backbone)
    graph_type = payload.get("graph_type") or "tokens"
    method = payload.get("method") or "token_shap_llm"

    run_id = payload.get("run_id") or ""
    split = infer_split_from_run_id(run_id) or ""
    extras = payload.get("extras") or {}

    token_text_raw = extras.get("token_text") or payload.get("token_text") or []
    if isinstance(token_text_raw, str):
        token_text = token_text_raw.split()
    elif isinstance(token_text_raw, (list, tuple)):
        token_text = [str(token) for token in token_text_raw]
    else:
        token_text = []

    ranked_indices = extract_ranked_indices(payload)

    num_tokens = extras.get("num_tokens")
    if num_tokens is None:
        num_tokens = payload.get("num_nodes")
    if num_tokens is None and token_text:
        num_tokens = len(token_text)
    if num_tokens is None and payload.get("node_importance"):
        num_tokens = len(payload["node_importance"])
    if num_tokens is None:
        num_tokens = len(ranked_indices)
    try:
        total_tokens = int(num_tokens)
    except Exception:
        total_tokens = len(token_text)

    if total_tokens <= 0 and token_text:
        total_tokens = len(token_text)

    position_ranks: List[float] = []
    if total_tokens and total_tokens > 0:
        position_ranks = [idx / float(total_tokens) for idx in ranked_indices]

    ranked_token_text = format_token_sequence(token_text, ranked_indices)
    full_token_text = format_token_sequence(token_text, list(range(len(token_text))))

    record: Dict[str, Any] = {
        "method": method,
        "backbone": backbone,
        "dataset": dataset_normalized,
        "dataset_raw": dataset_raw,
        "dataset_backbone": dataset_backbone,
        "graph_type": graph_type,
        "run_id": run_id,
        "split": split,
        "graph_index": payload.get("graph_index"),
        "global_graph_index": payload.get("global_graph_index"),
        "label": payload.get("label"),
        "prediction_class": payload.get("prediction_class"),
        "prediction_confidence": payload.get("prediction_confidence"),
        "is_correct": payload.get("is_correct"),
        "num_nodes": total_tokens,
        "num_edges": payload.get("num_edges", 0),
        "total_nodes": total_tokens,
        "num_ranked_nodes": len(ranked_indices),
        "ranked_nodes": format_sequence(ranked_indices),
        "ranked_map": format_float_sequence(position_ranks),
        "ranked_token_text": ranked_token_text,
        "token_text": full_token_text,
        "prompt": extras.get("prompt"),
        "masked_prompt": extras.get("masked_prompt"),
        "maskout_prompt": extras.get("maskout_prompt"),
        "sampling_ratio": extras.get("sampling_ratio"),
        "elapsed_time": extras.get("elapsed_time"),
        "max_tokens": extras.get("max_tokens"),
        "max_length": extras.get("max_length"),
    }

    top_tokens = payload.get("top_tokens")
    if top_tokens:
        record["top_tokens"] = format_sequence(top_tokens)
    minimal_coalition_tokens = payload.get("minimal_coalition_tokens")
    if minimal_coalition_tokens:
        record["minimal_coalition_tokens"] = format_sequence(minimal_coalition_tokens)

    return record


def process_pickles(
    base_dir: Path,
    output_dir: Path,
    *,
    limit: Optional[int] = None,
) -> List[Path]:
    pickle_paths = discover_llm_pickles(base_dir)
    if not pickle_paths:
        raise FileNotFoundError(f"No TokenSHAP pickle shards found under {base_dir}")

    if limit is not None:
        pickle_paths = pickle_paths[:limit]

    grouped_rows: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)

    for path in tqdm(pickle_paths, desc="Scanning TokenSHAP pickles", leave=False, colour="green"):
        try:
            payload = load_pickle_record(path)
            record = build_record(payload)
        except Exception as exc:
            print(f"! Skipping {path} ({exc})", file=sys.stderr)
            continue

        method_slug = slugify(record["method"])
        dataset_slug = slugify(record["dataset_backbone"] or record["dataset"])
        graph_slug = slugify(record["graph_type"] or "tokens")
        grouped_rows[(method_slug, dataset_slug, graph_slug)].append(record)

    written_paths: List[Path] = []
    for (method_slug, dataset_slug, graph_slug), rows in tqdm(
        sorted(grouped_rows.items()),
        desc="Writing CSVs",
        leave=False,
        colour="cyan",
    ):
        if not rows:
            continue
        target_dir = output_dir / method_slug / dataset_slug
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{graph_slug}.csv"

        df = pd.DataFrame(rows)
        sort_columns = [column for column in ("split", "run_id", "graph_index") if column in df.columns]
        if sort_columns:
            df.sort_values(sort_columns, inplace=True, kind="mergesort")
        df.to_csv(target_path, index=False)
        written_paths.append(target_path)
        print(f"✓ {method_slug} | {dataset_slug} | {graph_slug} → {target_path} ({len(df)} rows)")

    if not written_paths:
        print("– No CSV files were generated (no records processed).")
    return written_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate TokenSHAP LLM pickles into CSV summaries.")
    parser.add_argument(
        "--base-dir",
        default="outputs/insights/news/LLM",
        type=Path,
        help="Root directory containing TokenSHAP LLM outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/analytics/sequence",
        type=Path,
        help="Directory where CSV files will be written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of pickles processed (useful for smoke tests).",
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
    total_rows = 0
    for path in written:
        try:
            df = pd.read_csv(path)
            total_rows += len(df)
        except Exception:
            continue
    print(f"\nCompleted TokenSHAP aggregation for {len(written)} CSV file(s) ({total_rows} total rows).")


if __name__ == "__main__":
    main()
