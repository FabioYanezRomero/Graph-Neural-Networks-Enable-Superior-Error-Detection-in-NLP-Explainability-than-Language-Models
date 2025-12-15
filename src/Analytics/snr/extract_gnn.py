import argparse
import pickle
import sys
import types
from collections import defaultdict
from collections.abc import Iterable as IterableABC
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

EPS = 1e-9


def ensure_subgraphx_stub() -> None:
    """Inject a lightweight SubgraphXResult stub to load pickle payloads."""

    module_name = "src.explain.gnn.subgraphx.main"
    module = sys.modules.get(module_name)
    if module is None:
        module = types.ModuleType(module_name)
        sys.modules[module_name] = module

    if hasattr(module, "SubgraphXResult"):
        return

    class SubgraphXResult:  # type: ignore[too-many-instance-attributes]
        def __init__(
            self,
            graph_index: int,
            label: Optional[int],
            explanation: Any,
            related_prediction: Dict[str, Any],
            num_nodes: int,
            num_edges: int,
            hyperparams: Dict[str, Any],
        ) -> None:
            self.graph_index = graph_index
            self.label = label
            self.explanation = explanation
            self.related_prediction = related_prediction
            self.num_nodes = num_nodes
            self.num_edges = num_edges
            self.hyperparams = hyperparams

    module.SubgraphXResult = SubgraphXResult  # type: ignore[attr-defined]
    main_module = sys.modules.get("__main__")
    if main_module is None:
        main_module = types.ModuleType("__main__")
        sys.modules["__main__"] = main_module
    setattr(main_module, "SubgraphXResult", SubgraphXResult)


def slugify(value: str) -> str:
    safe = value.replace("/", "_").replace(" ", "_")
    safe = safe.replace("-", "_")
    return "".join(char for char in safe if char.isalnum() or char == "_").lower()


def normalize_dataset_name(dataset_raw: str) -> str:
    name = dataset_raw.replace("_", "-")
    if name.lower() == "sst2":
        return "sst-2"
    return name


def infer_split_from_run_id(run_id: str) -> Optional[str]:
    tokens = run_id.lower()
    for candidate in ("train", "test", "validation", "val", "dev"):
        if candidate in tokens:
            if candidate == "val":
                return "validation"
            if candidate == "dev":
                return "development"
            return candidate
    return None


def parse_metadata_from_path(pickle_path: Path) -> Dict[str, str]:
    parts = pickle_path.resolve().parts
    if "gnn_models" not in parts:
        raise ValueError(f"Unexpected path layout for {pickle_path}")
    idx = parts.index("gnn_models")
    try:
        backbone = parts[idx + 1]
        dataset = parts[idx + 2]
        graph_type = parts[idx + 3]
        method = parts[idx + 5]
        run_id = parts[idx + 6]
    except IndexError as exc:
        raise ValueError(f"Path is too short to extract metadata: {pickle_path}") from exc
    dataset_backbone = f"{backbone}/{dataset}"
    dataset_normalized = normalize_dataset_name(dataset)
    split = infer_split_from_run_id(run_id) or ""
    return {
        "backbone": backbone,
        "dataset": dataset_normalized,
        "dataset_raw": dataset,
        "dataset_backbone": dataset_backbone,
        "graph_type": graph_type,
        "method": method,
        "run_id": run_id,
        "split": split,
    }


def load_payload(path: Path, *, method: str) -> Dict[str, Any]:
    if method == "subgraphx":
        ensure_subgraphx_stub()
    with path.open("rb") as handler:
        payload = pickle.load(handler)
    if isinstance(payload, dict):
        return payload
    if hasattr(payload, "__dict__"):
        return dict(payload.__dict__)
    raise ValueError(f"Unsupported pickle payload in {path}")


def _safe_iterable(values: Optional[Iterable[Any]]) -> Iterable[Any]:
    if values is None:
        return ()
    if isinstance(values, (list, tuple, range)):
        return values
    if isinstance(values, np.ndarray):
        return values.tolist()
    if isinstance(values, IterableABC):
        return list(values)
    return ()


def _fix_indices(indices: Iterable[Any], count: int) -> List[int]:
    valid: List[int] = []
    for value in indices:
        try:
            idx = int(value)
        except Exception:
            continue
        if 0 <= idx < count:
            valid.append(idx)
    return valid


def _aggregate_subgraphx_scores(
    explanation: Sequence[Mapping[str, Any]], num_nodes: int
) -> np.ndarray:
    """Aggregate coalition weights into per-node scores for SubgraphX outputs."""

    max_idx = num_nodes - 1
    if num_nodes <= 0:
        for entry in explanation:
            coalition = entry.get("coalition") or []
            for node in coalition:
                try:
                    max_idx = max(max_idx, int(node))
                except Exception:
                    continue
        num_nodes = max_idx + 1 if max_idx >= 0 else 0
    if num_nodes <= 0:
        return np.zeros(0, dtype=float)

    scores = np.zeros(num_nodes, dtype=float)
    for entry in explanation:
        coalition = entry.get("coalition") or []
        if not coalition:
            continue
        try:
            weight = float(entry.get("W", entry.get("P", 0.0)))
        except Exception:
            continue
        weight = abs(weight)
        if weight <= 0.0:
            continue
        norm = max(len(coalition), 1)
        for node in coalition:
            try:
                idx = int(node)
            except Exception:
                continue
            if 0 <= idx < num_nodes:
                scores[idx] += weight / norm
    return scores


def compute_signal_noise(
    payload: Mapping[str, Any],
) -> Tuple[float, float, float, float, int, int, str, bool]:
    """Return signal/noise statistics plus provenance metadata for a GNN explanation."""

    explanation = payload.get("explanation")
    importance_array: np.ndarray
    top_k_source = "unknown"
    aggregated = False

    if isinstance(explanation, Mapping):
        importance = explanation.get("node_importance")
        if importance is None:
            return 0.0, 0.0, 0.0, 0.0, 0, 0, top_k_source, aggregated
        importance_array = np.asarray(list(_safe_iterable(importance)), dtype=float)
    elif isinstance(explanation, Sequence):
        num_nodes = payload.get("num_nodes")
        try:
            num_nodes = int(num_nodes) if num_nodes is not None else 0
        except Exception:
            num_nodes = 0
        importance_array = _aggregate_subgraphx_scores(explanation, num_nodes)
        aggregated = True
    else:
        return 0.0, 0.0, 0.0, 0.0, 0, 0, top_k_source, aggregated

    if importance_array.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0, 0, top_k_source, aggregated

    related = payload.get("related_prediction")
    top_nodes: List[int] = []
    if isinstance(explanation, Mapping) and isinstance(
        explanation.get("top_nodes"), Iterable
    ):
        top_nodes = _fix_indices(explanation.get("top_nodes"), importance_array.size)
        if top_nodes:
            top_k_source = "explicit_nodes"
    if not top_nodes and isinstance(related, Mapping):
        top_nodes = _fix_indices(related.get("top_nodes", []), importance_array.size)
        if top_nodes:
            top_k_source = "explicit_nodes"

    ranking: List[int] = []
    if isinstance(explanation, Mapping) and isinstance(
        explanation.get("node_ranking"), Iterable
    ):
        ranking = _fix_indices(explanation.get("node_ranking"), importance_array.size)
    if not ranking and isinstance(related, Mapping):
        ranking = _fix_indices(related.get("ranked_nodes", []), importance_array.size)
    if not ranking:
        order = np.argsort(np.abs(importance_array))[::-1]
        ranking = order.tolist()

    if not top_nodes:
        sparsity = None
        if isinstance(related, Mapping):
            sparsity = related.get("sparsity")
        if sparsity is None:
            sparsity = payload.get("sparsity")
        try:
            sparsity = float(sparsity) if sparsity is not None else 0.2
        except Exception:
            sparsity = 0.2
        k = max(1, int(round(importance_array.size * max(min(sparsity, 1.0), 0.0))))
        top_nodes = ranking[:k]
        top_k_source = "sparsity_based"

    top_nodes = list(dict.fromkeys(top_nodes))
    if not top_nodes:
        order = np.argsort(np.abs(importance_array))[::-1]
        k = max(1, min(5, importance_array.size))
        top_nodes = order[:k].tolist()
        top_k_source = "auto_default"
    elif top_k_source == "unknown":
        # A ranking branch succeeded
        top_k_source = "ranking_based"

    mask = np.zeros_like(importance_array, dtype=bool)
    mask[top_nodes] = True
    if mask.all():
        mask[top_nodes[len(top_nodes) // 2 :]] = False  # ensure noise set non-empty

    signal_values = np.abs(importance_array[mask])
    noise_values = np.abs(importance_array[~mask])

    signal_mean = float(signal_values.mean()) if signal_values.size else 0.0
    noise_mean = float(noise_values.mean()) if noise_values.size else 0.0
    signal_std = float(signal_values.std(ddof=0)) if signal_values.size else 0.0
    noise_std = float(noise_values.std(ddof=0)) if noise_values.size else 0.0

    return (
        signal_mean,
        signal_std,
        noise_mean,
        noise_std,
        int(signal_values.size),
        int(noise_values.size),
        top_k_source,
        aggregated,
    )


def compute_instance_snr(signal_mean: float, signal_std: float, noise_mean: float, noise_std: float) -> Tuple[float, float]:
    """Compute per-instance SNR (linear and dB) from signal/noise stats."""

    if signal_mean <= 0.0:
        return 0.0, -np.inf
    denom = max(noise_mean, EPS)
    snr_linear = float(signal_mean / denom)
    snr_db = 20.0 * float(np.log10(max(snr_linear, EPS)))
    return snr_linear, snr_db


def build_record(payload: Dict[str, Any], metadata: Dict[str, str]) -> Dict[str, Any]:
    (
        signal_mean,
        signal_std,
        noise_mean,
        noise_std,
        signal_count,
        noise_count,
        top_k_source,
        aggregated,
    ) = compute_signal_noise(payload)
    snr_linear, snr_db = compute_instance_snr(signal_mean, signal_std, noise_mean, noise_std)

    label = payload.get("label")
    prediction_class = payload.get("prediction_class")
    is_correct = payload.get("is_correct")
    if is_correct is None and label is not None and prediction_class is not None:
        try:
            is_correct = int(label) == int(prediction_class)
        except Exception:
            pass

    record: Dict[str, Any] = {
        "method": metadata["method"],
        "backbone": metadata["backbone"],
        "dataset": metadata["dataset"],
        "dataset_raw": metadata["dataset_raw"],
        "dataset_backbone": metadata["dataset_backbone"],
        "graph_type": metadata["graph_type"],
        "run_id": metadata["run_id"],
        "split": metadata["split"],
        "graph_index": payload.get("graph_index"),
        "global_graph_index": payload.get("global_graph_index"),
        "label": label,
        "prediction_class": prediction_class,
        "prediction_confidence": payload.get("prediction_confidence"),
        "is_correct": is_correct,
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
        "aggregated_from_coalitions": bool(aggregated),
    }
    return record


def discover_pickles(base_dir: Path, method: str) -> List[Path]:
    pattern = f"**/explanations/{method}/**/results_split_pickle/graph_*.pkl"
    return sorted(base_dir.glob(pattern))


def process_method(
    method: str,
    base_dir: Path,
    output_dir: Path,
    limit: Optional[int] = None,
) -> List[Path]:
    pickle_paths = discover_pickles(base_dir, method)
    if not pickle_paths:
        print(f"– No {method} pickles discovered under {base_dir}")
        return []
    if limit is not None:
        pickle_paths = pickle_paths[:limit]

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for path in tqdm(pickle_paths, desc=f"Scanning {method} pickles", leave=False, colour="green"):
        try:
            metadata = parse_metadata_from_path(path)
            payload = load_payload(path, method=method)
            record = build_record(payload, metadata)
        except Exception as exc:
            print(f"! Skipping {path} ({exc})")
            continue
        key = (record["dataset_backbone"], record["graph_type"])
        grouped[key].append(record)

    written_paths: List[Path] = []
    for (dataset_backbone, graph_type), rows in tqdm(
        sorted(grouped.items()),
        desc=f"{method} datasets",
        leave=False,
        colour="cyan",
    ):
        if not rows:
            continue
        dataset_slug = slugify(dataset_backbone)
        graph_slug = slugify(graph_type)
        target_dir = output_dir / method / dataset_slug
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{graph_slug}.csv"

        df = pd.DataFrame(rows)
        sort_columns = [column for column in ("split", "run_id", "graph_index") if column in df.columns]
        if sort_columns:
            df.sort_values(sort_columns, inplace=True, kind="mergesort")
        df.to_csv(target_path, index=False)
        written_paths.append(target_path)
        print(f"✓ {method} | {dataset_backbone} | {graph_type} → {target_path} ({len(df)} rows)")

    return written_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract SNR analytics from GNN explanation payloads.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("outputs/gnn_models"),
        help="Root directory containing GNN explanation outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analytics/snr"),
        help="Directory where SNR CSV files will be written.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["subgraphx", "graphsvx"],
        help="Explainability methods to process.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of pickles processed per method.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    base_dir: Path = args.base_dir
    output_dir: Path = args.output_dir
    methods: List[str] = [method.lower() for method in args.methods]

    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    all_written: List[Path] = []
    for method in tqdm(methods, desc="Processing methods", leave=False, colour="blue"):
        all_written.extend(process_method(method, base_dir, output_dir, args.limit))

    if all_written:
        total_rows = 0
        for path in all_written:
            try:
                df = pd.read_csv(path)
                total_rows += len(df)
            except Exception:
                continue
        print(
            f"\nCompleted SNR extraction for {len(all_written)} CSV file(s) ({total_rows} total rows)."
        )
    else:
        print("\nNo SNR CSV files were generated.")


if __name__ == "__main__":
    main()
