import argparse
import pickle
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.Analytics.tokens.extract_gnn import GraphTokenResolver

def ensure_subgraphx_stub() -> None:
    """
    Injects a lightweight SubgraphXResult stub so pickle payloads can be loaded
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
            self.explanation = self._sanitize_explanation(explanation)
            self.related_prediction = related_prediction
            self.num_nodes = num_nodes
            self.num_edges = num_edges
            self.hyperparams = hyperparams

        @staticmethod
        def _sanitize_explanation(raw: Any) -> List[Dict[str, Any]]:
            cleaned: List[Dict[str, Any]] = []
            if not isinstance(raw, list):
                return cleaned
            for entry in raw:
                if not isinstance(entry, dict):
                    continue
                coalition = entry.get("coalition") or entry.get("node_idx") or []
                try:
                    coalition = [int(v) for v in coalition]
                except Exception:
                    coalition = []
                info: Dict[str, Any] = {
                    "coalition": coalition,
                    "P": entry.get("P"),
                    "W": entry.get("W"),
                    "N": entry.get("N"),
                }
                cleaned.append(info)
            return cleaned

    module.SubgraphXResult = SubgraphXResult  # type: ignore[attr-defined]
    main_module = sys.modules.get("__main__")
    if main_module is None:
        main_module = types.ModuleType("__main__")
        sys.modules["__main__"] = main_module
    setattr(main_module, "SubgraphXResult", SubgraphXResult)


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


def slugify(value: str) -> str:
    safe = value.replace("/", "_").replace(" ", "_")
    safe = safe.replace("-", "_")
    return "".join(char for char in safe if char.isalnum() or char == "_").lower()


def normalize_dataset_name(dataset_raw: str) -> str:
    name = dataset_raw.replace("_", "-")
    if name.lower() == "sst2":
        return "sst-2"
    return name


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


def load_individual_payload(path: Path, *, method: str) -> Dict[str, Any]:
    if method == "subgraphx":
        ensure_subgraphx_stub()
    with path.open("rb") as handler:
        payload = pickle.load(handler)
    if isinstance(payload, dict):
        return payload
    if hasattr(payload, "__dict__"):
        return dict(payload.__dict__)
    raise ValueError(f"Unsupported pickle payload in {path}")


def extract_ranked_nodes(payload: Dict[str, Any]) -> List[int]:
    ranked: Iterable[int] = ()
    related_prediction = payload.get("related_prediction") or {}
    ranked = related_prediction.get("ranked_nodes") or ()

    explanation = payload.get("explanation")
    if not ranked and isinstance(explanation, dict):
        ranked = explanation.get("node_ranking") or explanation.get("ranked_nodes") or ()
        if not ranked:
            raw_scores = explanation.get("node_importance")
            if isinstance(raw_scores, np.ndarray):
                scores = raw_scores.tolist()
                ranked = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
            elif isinstance(raw_scores, (list, tuple)):
                scores = list(raw_scores)
                ranked = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    if not ranked and isinstance(explanation, list):
        scored_nodes: Dict[int, float] = {}
        for entry in explanation:
            if not isinstance(entry, dict):
                continue
            coalition = entry.get("coalition")
            weight = entry.get("W")
            if coalition is None or weight is None:
                continue
            try:
                weight_val = float(weight)
            except Exception:
                continue
            for node in coalition:
                try:
                    node_id = int(node)
                except Exception:
                    continue
                scored_nodes[node_id] = max(scored_nodes.get(node_id, weight_val), weight_val)
        if scored_nodes:
            ranked = sorted(scored_nodes.keys(), key=lambda node: scored_nodes[node], reverse=True)

    seen: set[int] = set()
    ordered_nodes: List[int] = []
    for value in ranked:
        try:
            node_id = int(value)
        except Exception:
            continue
        if node_id in seen:
            continue
        seen.add(node_id)
        ordered_nodes.append(node_id)
    return ordered_nodes


def extract_graph_stats(payload: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    num_nodes = payload.get("num_nodes")
    num_edges = payload.get("num_edges")
    if num_nodes is not None and num_edges is not None:
        return int(num_nodes), int(num_edges)
    explanation = payload.get("explanation")
    if isinstance(explanation, dict):
        if num_nodes is None:
            num_nodes = explanation.get("num_nodes")
        if num_edges is None:
            num_edges = explanation.get("num_edges")
    try:
        return (int(num_nodes) if num_nodes is not None else None, int(num_edges) if num_edges is not None else None)
    except Exception:
        return None, None


def format_sequence(values: Sequence[Any]) -> str:
    if not values:
        return ""
    return " ".join(str(value) for value in values)


def format_float_sequence(values: Sequence[float]) -> str:
    if not values:
        return ""
    return "[" + ", ".join(f"{value:.6f}" for value in values) + "]"


def should_filter_constituency(graph_type: Optional[str]) -> bool:
    if not graph_type:
        return False
    return "constituency" in str(graph_type).lower()


def is_special_constituency_token(token: Any) -> bool:
    if not isinstance(token, str):
        return False
    stripped = token.strip()
    return stripped.startswith("«") and stripped.endswith("»")


def build_record(
    payload: Dict[str, Any],
    metadata: Dict[str, str],
    resolver: Optional[GraphTokenResolver] = None,
) -> Dict[str, Any]:
    ranked_nodes = extract_ranked_nodes(payload)
    num_nodes, num_edges = extract_graph_stats(payload)
    total_nodes = num_nodes if num_nodes is not None else (len(ranked_nodes) if ranked_nodes else None)

    position_ranks: List[float] = []
    reindexed_nodes: List[int] = list(ranked_nodes)

    if resolver is not None and should_filter_constituency(metadata.get("graph_type")):
        graph_index = payload.get("graph_index")
        if graph_index is None:
            raise ValueError("Missing graph_index in payload required for constituency filtering.")
        bundle = resolver.resolve(
            backbone=metadata["backbone"],
            dataset_raw=metadata["dataset_raw"],
            split=metadata["split"] or "test",
            graph_type=metadata["graph_type"],
            graph_index=int(graph_index),
        )
        node_text = bundle.node_text
        special_indices = {idx for idx, token in enumerate(node_text) if is_special_constituency_token(token)}
        mapping: Dict[int, int] = {}
        for original_idx in range(len(node_text)):
            if original_idx in special_indices:
                continue
            mapping[original_idx] = len(mapping)

        filtered_nodes: List[int] = []
        for idx in ranked_nodes:
            if idx in special_indices:
                continue
            new_idx = mapping.get(idx)
            if new_idx is None:
                continue
            filtered_nodes.append(new_idx)

        reindexed_nodes = filtered_nodes
        if mapping:
            total_nodes = len(mapping)
            num_nodes = len(mapping)
        else:
            total_nodes = 0
            num_nodes = 0

    if total_nodes and total_nodes > 0:
        position_ranks = [node / float(total_nodes) for node in reindexed_nodes]

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
        "label": payload.get("label"),
        "prediction_class": payload.get("prediction_class"),
        "prediction_confidence": payload.get("prediction_confidence"),
        "is_correct": payload.get("is_correct"),
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "total_nodes": total_nodes,
        "num_ranked_nodes": len(reindexed_nodes),
        "ranked_nodes": format_sequence(reindexed_nodes),
        "ranked_map": format_float_sequence(position_ranks),
    }
    return record


def discover_individual_pickles(base_dir: Path, method: str) -> List[Path]:
    pattern = f"**/explanations/{method}/**/results_split_pickle/*.pkl"
    return sorted(base_dir.glob(pattern))


def process_method_pickles(
    method: str,
    base_dir: Path,
    output_dir: Path,
    resolver: Optional[GraphTokenResolver],
    limit: Optional[int] = None,
) -> List[Path]:
    pickle_paths = discover_individual_pickles(base_dir, method)
    if limit is not None:
        pickle_paths = pickle_paths[:limit]

    grouped_paths: Dict[Tuple[str, str], List[Tuple[Path, Dict[str, str]]]] = defaultdict(list)
    for path in tqdm(pickle_paths, desc=f"Scanning {method} pickles", leave=False, colour="green"):
        try:
            metadata = parse_metadata_from_path(path)
        except Exception as exc:
            print(f"! Skipping {path} ({exc})")
            continue
        group_key = (metadata["dataset"], metadata["graph_type"])
        grouped_paths[group_key].append((path, metadata))

    written: List[Path] = []
    for (dataset, graph_type) in tqdm(
        sorted(grouped_paths.keys()),
        desc=f"{method} datasets",
        leave=False,
        colour="cyan",
    ):
        rows: List[Dict[str, Any]] = []
        items = grouped_paths[(dataset, graph_type)]
        for path, metadata in tqdm(
            items,
            desc=f"{dataset} | {graph_type}",
            leave=False,
            colour="yellow",
        ):
            try:
                payload = load_individual_payload(path, method=method)
                record = build_record(payload, metadata, resolver=resolver)
            except Exception as exc:
                print(f"! Skipping {path} ({exc})")
                continue
            rows.append(record)

        if not rows:
            continue

        dataset_slug = slugify(dataset)
        graph_slug = slugify(graph_type)
        target_dir = output_dir / method / dataset_slug
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{graph_slug}.csv"

        df = pd.DataFrame(rows)
        sort_cols = [col for col in ("split", "run_id", "graph_index") if col in df.columns]
        if sort_cols:
            df.sort_values(sort_cols, inplace=True, kind="mergesort")
        df.to_csv(target_path, index=False)
        written.append(target_path)
        print(f"✓ {method} | {dataset} | {graph_type} → {target_path} ({len(df)} rows)")

    if not written:
        print(f"– No {method} pickles discovered under {base_dir}")
    return written


def process_all_methods(
    methods: Sequence[str],
    base_dir: Path,
    output_dir: Path,
    resolver: Optional[GraphTokenResolver],
    limit: Optional[int] = None,
) -> List[Path]:
    written: List[Path] = []
    for method in tqdm(methods, desc="Processing methods", leave=False, colour="blue"):
        written.extend(process_method_pickles(method, base_dir, output_dir, resolver, limit))
    return written


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate GNN explanation pickles into CSV summaries.")
    parser.add_argument(
        "--base-dir",
        default="outputs/gnn_models",
        type=Path,
        help="Root directory that contains the gnn_models outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/analytics/sequence",
        type=Path,
        help="Directory where CSV files will be written.",
    )
    parser.add_argument(
        "--graph-root",
        default="outputs/graphs",
        type=Path,
        help="Directory containing NetworkX graph artefacts used for token metadata.",
    )
    parser.add_argument(
        "--pyg-root",
        default="outputs/pyg_graphs",
        type=Path,
        help="Directory containing PyG graph artefacts used for token metadata.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["subgraphx", "graphsvx"],
        help="Methods to include (subgraphx, graphsvx).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of pickles processed per method (useful for smoke tests).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    base_dir: Path = args.base_dir
    output_dir: Path = args.output_dir
    graph_root: Path = args.graph_root
    pyg_root: Path = args.pyg_root
    methods: List[str] = [method.lower() for method in args.methods]

    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")
    if not graph_root.exists():
        raise FileNotFoundError(f"Graph root directory not found: {graph_root}")
    if not pyg_root.exists():
        raise FileNotFoundError(f"PyG root directory not found: {pyg_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    resolver = GraphTokenResolver(graph_root=graph_root, pyg_root=pyg_root)
    written = process_all_methods(methods, base_dir, output_dir, resolver, args.limit)
    total_rows = 0
    for path in written:
        try:
            df = pd.read_csv(path)
            total_rows += len(df)
        except Exception:
            continue
    print(f"\nCompleted aggregation for {len(written)} CSV files ({total_rows} total rows).")


if __name__ == "__main__":
    main()
