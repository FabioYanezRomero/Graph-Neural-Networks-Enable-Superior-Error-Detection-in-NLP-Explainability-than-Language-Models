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


def ensure_subgraphx_stub() -> None:
    """
    Inject a lightweight SubgraphXResult stub so pickle payloads can be loaded
    without requiring the original module graph.
    """
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


def extract_fidelity_metrics(payload: Mapping[str, Any]) -> Tuple[
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
]:
    related = payload.get("related_prediction") or {}
    if not isinstance(related, Mapping):
        related = {}

    origin = payload.get("origin_confidence")
    if origin is None:
        origin = related.get("origin")
    if origin is None:
        origin = payload.get("prediction_confidence")

    masked = payload.get("masked_confidence")
    if masked is None:
        masked = related.get("masked")

    maskout = payload.get("maskout_confidence")
    if maskout is None:
        maskout = related.get("maskout")

    fidelity_plus = payload.get("fidelity_plus")
    if fidelity_plus is None and origin is not None and masked is not None:
        try:
            fidelity_plus = float(origin) - float(masked)
        except Exception:
            fidelity_plus = None

    fidelity_minus = payload.get("fidelity_minus")
    if fidelity_minus is None and origin is not None and maskout is not None:
        try:
            fidelity_minus = float(origin) - float(maskout)
        except Exception:
            fidelity_minus = None

    sparsity = payload.get("sparsity")
    if sparsity is None:
        sparsity = related.get("sparsity")

    try:
        sparsity = float(sparsity) if sparsity is not None else None
    except Exception:
        sparsity = None

    return (
        float(origin) if origin is not None else None,
        float(masked) if masked is not None else None,
        float(maskout) if maskout is not None else None,
        float(fidelity_plus) if fidelity_plus is not None else None,
        float(fidelity_minus) if fidelity_minus is not None else None,
        sparsity,
    )


def build_record(payload: Dict[str, Any], metadata: Dict[str, str]) -> Dict[str, Any]:
    (
        origin_confidence,
        masked_confidence,
        maskout_confidence,
        fidelity_plus,
        fidelity_minus,
        sparsity,
    ) = extract_fidelity_metrics(payload)

    sparsity_percent = float(sparsity * 100.0) if sparsity is not None else None

    label = payload.get("label")
    prediction_class = payload.get("prediction_class")
    is_correct = payload.get("is_correct")
    if is_correct is None and label is not None and prediction_class is not None:
        try:
            is_correct = int(label) == int(prediction_class)
        except Exception:
            pass

    fidelity_asymmetry: Optional[float] = None
    abs_fidelity_asymmetry: Optional[float] = None
    if fidelity_plus is not None and fidelity_minus is not None:
        try:
            fidelity_asymmetry = float(fidelity_minus) - float(fidelity_plus)
            abs_fidelity_asymmetry = abs(fidelity_asymmetry)
        except Exception:
            fidelity_asymmetry = None
            abs_fidelity_asymmetry = None

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
        "fidelity_plus": fidelity_plus,
        "fidelity_minus": fidelity_minus,
        "fidelity_asymmetry": fidelity_asymmetry,
        "abs_fidelity_asymmetry": abs_fidelity_asymmetry,
        "sparsity": sparsity,
        "sparsity_percent": sparsity_percent,
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
        sort_columns = [
            column for column in ("split", "run_id", "graph_index") if column in df.columns
        ]
        if sort_columns:
            df.sort_values(sort_columns, inplace=True, kind="mergesort")
        df.to_csv(target_path, index=False)
        written_paths.append(target_path)
        print(f"✓ {method} | {dataset_backbone} | {graph_type} → {target_path} ({len(df)} rows)")

    return written_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate GNN fidelity metrics into CSV summaries.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("outputs/gnn_models"),
        help="Root directory containing GNN explanation outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analytics/fidelity"),
        help="Directory where fidelity CSV files will be written.",
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
            f"\nCompleted fidelity aggregation for {len(all_written)} CSV file(s) ({total_rows} total rows)."
        )
    else:
        print("\nNo fidelity CSV files were generated.")


if __name__ == "__main__":
    main()
