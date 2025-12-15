import argparse
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
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


def join_commas(values: Sequence[Any]) -> str:
    if not values:
        return ""
    return ",".join(str(value) for value in values)


def join_float_commas(values: Sequence[float]) -> str:
    if not values:
        return ""
    return ",".join(f"{value:.6f}" for value in values)


def format_token_sequence(tokens: Sequence[str], indices: Sequence[int]) -> str:
    if not tokens or not indices:
        return ""
    chosen: List[str] = []
    token_count = len(tokens)
    for index in indices:
        try:
            idx = int(index)
        except Exception:
            continue
        if 0 <= idx < token_count:
            chosen.append(str(tokens[idx]))
    return " ".join(chosen)


def _coerce_float_list(values: Optional[Iterable[Any]]) -> List[float]:
    if values is None:
        return []
    if isinstance(values, np.ndarray):
        values = values.tolist()
    floats: List[float] = []
    for value in values:
        try:
            floats.append(float(value))
        except Exception:
            continue
    return floats


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


def _extract_ranked_indices(payload: Mapping[str, Any]) -> List[int]:
    ranked: Iterable[Any] = payload.get("top_nodes") or payload.get("ranked_nodes") or ()

    related_prediction = payload.get("related_prediction") or {}
    if not ranked:
        ranked = related_prediction.get("ranked_nodes") or ()
    if not ranked:
        ranked = related_prediction.get("top_nodes") or ()

    indices: List[int] = []
    seen: set[int] = set()
    for value in ranked:
        try:
            index = int(value)
        except Exception:
            continue
        if index in seen:
            continue
        seen.add(index)
        indices.append(index)
    return indices


def _load_payload(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict):
        return payload
    if hasattr(payload, "__dict__"):
        return dict(payload.__dict__)
    raise ValueError(f"Unsupported TokenSHAP payload at {path}")


def _discover_pickles(base_dir: Path) -> Sequence[Path]:
    records = sorted(base_dir.glob("**/*_records_split_pickle/graph_*.pkl"))
    summaries = sorted(
        path
        for path in base_dir.glob("**/token_shap_shard*of*_split_pickle/graph_*.pkl")
        if "_records_split_pickle" not in path.as_posix()
    )
    seen = {path.relative_to(base_dir) for path in records}
    merged: List[Path] = list(records)
    for path in summaries:
        rel = path.relative_to(base_dir)
        if rel not in seen:
            merged.append(path)
    return merged


def _token_text_from_payload(payload: Mapping[str, Any]) -> List[str]:
    extras = payload.get("extras") or {}
    token_text_raw = extras.get("token_text") or payload.get("token_text") or []
    if isinstance(token_text_raw, str):
        return token_text_raw.split()
    if isinstance(token_text_raw, (list, tuple)):
        return [str(token) for token in token_text_raw]
    return []


def _sanitize_indices(values: Optional[Iterable[Any]]) -> List[int]:
    if values is None:
        return []
    indices: List[int] = []
    seen: set[int] = set()
    for value in values:
        try:
            idx = int(value)
        except Exception:
            continue
        if idx in seen:
            continue
        seen.add(idx)
        indices.append(idx)
    return indices


def _clean_token_piece(piece: str) -> str:
    if piece.startswith("##"):
        return piece[2:]
    if piece.startswith("Ġ") or piece.startswith("▁"):
        return piece[1:]
    return piece


def _group_wordpieces(tokens: Sequence[str]) -> Tuple[List[str], List[int]]:
    words: List[str] = []
    mapping: List[int] = []
    for token in tokens:
        if token.startswith("##") and words:
            clean = _clean_token_piece(token)
            words[-1] = words[-1] + clean
            mapping.append(len(words) - 1)
            continue
        clean = _clean_token_piece(token)
        words.append(clean)
        mapping.append(len(words) - 1)
    return words, mapping


def build_record(payload: Dict[str, Any]) -> Dict[str, Any]:
    dataset_backbone = payload.get("dataset") or ""
    backbone, dataset_raw, dataset = _parse_dataset_fields(dataset_backbone)
    graph_type = payload.get("graph_type") or "tokens"
    method = payload.get("method") or "token_shap_llm"
    run_id = payload.get("run_id")
    split = infer_split_from_run_id(run_id) or ""

    extras = payload.get("extras") or {}

    token_text = _token_text_from_payload(payload)
    total_tokens = len(token_text)
    if total_tokens == 0:
        total_tokens = int(payload.get("num_nodes") or payload.get("num_tokens") or 0)

    node_importance_raw = payload.get("node_importance")
    node_importance = _coerce_float_list(node_importance_raw)
    if total_tokens and node_importance and len(node_importance) != total_tokens:
        # Trim or pad to align with token length
        if len(node_importance) > total_tokens:
            node_importance = node_importance[:total_tokens]
        else:
            node_importance.extend([0.0] * (total_tokens - len(node_importance)))

    words: List[str]
    word_importance: List[float]
    token_text_str = ""
    if token_text:
        words, token_to_word = _group_wordpieces(token_text)
        token_text_str = " ".join(token_text)
        num_words = len(words)
        word_scores_sum: List[float] = [0.0] * num_words
        word_counts: List[int] = [0] * num_words
        for token_idx, score in enumerate(node_importance):
            if token_idx >= len(token_to_word):
                continue
            word_idx = token_to_word[token_idx]
            if word_idx < 0 or word_idx >= num_words:
                continue
            word_scores_sum[word_idx] += score
            word_counts[word_idx] += 1

        word_importance = []
        for total, count in zip(word_scores_sum, word_counts):
            if count > 0:
                word_importance.append(total / count)
            else:
                word_importance.append(0.0)
    else:
        top_words = payload.get("top_words") or payload.get("top_tokens") or []
        words = [str(item) for item in top_words]
        word_importance = _coerce_float_list(payload.get("top_word_scores"))[: len(words)]
        if len(word_importance) < len(words):
            word_importance.extend([0.0] * (len(words) - len(word_importance)))

    num_words = len(words)

    ranked_indices: List[int] = []
    if word_importance:
        ranked_indices = sorted(
            range(len(word_importance)),
            key=lambda idx: (word_importance[idx], -idx),
            reverse=True,
        )
    if not ranked_indices:
        ranked_indices = _extract_ranked_indices(payload)
        ranked_indices = [idx for idx in ranked_indices if 0 <= idx < num_words]
    ranked_scores = [word_importance[idx] for idx in ranked_indices]
    ranked_tokens_list = [words[idx] for idx in ranked_indices]
    if not ranked_indices:
        ranked_indices = list(range(num_words))
        ranked_scores = [word_importance[idx] for idx in ranked_indices]
        ranked_tokens_list = [words[idx] for idx in ranked_indices]

    record: Dict[str, Any] = {
        "method": method,
        "backbone": backbone,
        "dataset": dataset,
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
        "num_tokens": total_tokens,
        "num_words": num_words,
        "ranked_indices": join_commas(ranked_indices),
        "ranked_tokens": join_commas(ranked_tokens_list),
        "ranked_scores": join_float_commas(ranked_scores),
        "token_text": token_text_str,
        "word_text": " ".join(words),
        "sampling_ratio": extras.get("sampling_ratio"),
        "elapsed_time": extras.get("elapsed_time"),
        "max_tokens": extras.get("max_tokens"),
        "max_length": extras.get("max_length"),
        "prompt": extras.get("prompt"),
        "masked_prompt": extras.get("masked_prompt"),
        "maskout_prompt": extras.get("maskout_prompt"),
    }
    return record


def process_pickles(
    base_dir: Path,
    output_dir: Path,
    *,
    limit: Optional[int] = None,
) -> List[Path]:
    pickle_paths = _discover_pickles(base_dir)
    if limit is not None:
        pickle_paths = pickle_paths[:limit]

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for path in tqdm(pickle_paths, desc="Scanning TokenSHAP pickles", leave=False, colour="green"):
        try:
            payload = _load_payload(path)
            record = build_record(payload)
        except Exception as exc:
            print(f"! Skipping {path} ({exc})")
            continue
        key = (record.get("dataset_backbone") or record.get("dataset") or "", record.get("graph_type") or "")
        grouped[key].append(record)

    written: List[Path] = []
    for (dataset_backbone, graph_type), rows in tqdm(
        sorted(grouped.items()),
        desc="Writing CSVs",
        leave=False,
        colour="cyan",
    ):
        if not rows:
            continue
        method = rows[0].get("method") or "token_shap_llm"
        dataset_slug = slugify(dataset_backbone or rows[0].get("dataset") or "unknown")
        graph_slug = slugify(graph_type or "tokens")
        target_dir = output_dir / method / dataset_slug
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{graph_slug}.csv"

        df = pd.DataFrame(rows)
        sort_columns = [column for column in ("run_id", "graph_index") if column in df.columns]
        if sort_columns:
            df.sort_values(sort_columns, inplace=True, kind="mergesort")
        df.to_csv(target_path, index=False)
        written.append(target_path)
        print(
            f"✓ {method} | {dataset_backbone or rows[0].get('dataset')} | {graph_type or 'tokens'}"
            f" → {target_path} ({len(df)} rows)"
        )
    return written


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate TokenSHAP per-record pickles into token ranking CSVs.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("outputs/insights/LLM"),
        help="Root directory containing TokenSHAP outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analytics/tokens"),
        help="Destination directory for token analytics CSV files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of pickles processed (for debugging).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    base_dir = args.base_dir
    output_dir = args.output_dir
    limit = args.limit

    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    written = process_pickles(base_dir, output_dir, limit=limit)

    if written:
        total_rows = 0
        for path in written:
            try:
                df = pd.read_csv(path)
                total_rows += len(df)
            except Exception:
                continue
        print(f"\nCompleted TokenSHAP token aggregation for {len(written)} CSV file(s) ({total_rows} total rows).")
    else:
        print("\nNo TokenSHAP token CSV files were generated.")


if __name__ == "__main__":
    main()
