#!/usr/bin/env python3
"""Find sentences the LLM explainer misses but GNN explainers catch."""

from __future__ import annotations

import argparse
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from use_case.save_logistic_coefficients import feature_matrix, fit_logistic  # type: ignore


DATASET_DIR = Path("outputs/use_case/module_datasets")
TOKEN_MODULE = ("token_shap_llm", "tokens")
GNN_MODULES: Tuple[Tuple[str, str], ...] = (
    ("graphsvx", "skipgrams"),
    ("graphsvx", "window"),
    ("subgraphx", "constituency"),
    ("subgraphx", "syntactic"),
)

TOKEN_EXPL_ROOT = Path("outputs/insights/LLM")
GNN_EXPL_ROOT = Path("outputs/gnn_models")
GRAPH_ROOT = Path("outputs/graphs")

GRAPH_TYPE_DIRS: Mapping[str, str] = {
    "skipgrams": "skipgrams.word.k1.n2",
    "window": "window.word.k1",
    "constituency": "constituency",
    "syntactic": "syntactic",
}


@dataclass(frozen=True)
class ModulePredictions:
    dataset: str
    method: str
    graph: str
    frame: pd.DataFrame


@dataclass(frozen=True)
class ModuleLogisticResult:
    frame: pd.DataFrame
    pipeline: object | None
    feature_cols: List[str]


class TokenShapRepository:
    def __init__(self) -> None:
        self._dir_cache: Dict[str, List[Path]] = {}
        self._file_cache: Dict[Tuple[str, int], Dict[str, object]] = {}

    def _token_dirs(self, dataset_backbone: str) -> List[Path]:
        if dataset_backbone in self._dir_cache:
            return self._dir_cache[dataset_backbone]
        base = TOKEN_EXPL_ROOT / Path(dataset_backbone)
        dirs: List[Path] = []
        if base.exists():
            candidates = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("token_shap")]
            # prefer directories without "records" so we get lighter pickles first
            dirs = sorted(candidates, key=lambda p: ("records" in p.name, p.name))
        self._dir_cache[dataset_backbone] = dirs
        return dirs

    def _load_pickle(self, dataset_backbone: str, global_index: int) -> Dict[str, object] | None:
        cache_key = (dataset_backbone, global_index)
        if cache_key in self._file_cache:
            return self._file_cache[cache_key]
        for folder in self._token_dirs(dataset_backbone):
            candidate = folder / f"graph_{global_index:05d}.pkl"
            if candidate.exists():
                with candidate.open("rb") as handle:
                    data = pickle.load(handle)
                self._file_cache[cache_key] = data
                return data
        self._file_cache[cache_key] = None  # type: ignore[assignment]
        return None

    def describe(self, dataset_backbone: str, global_index: int, top_k: int) -> str:
        data = self._load_pickle(dataset_backbone, global_index)
        if not data:
            return ""
        tokens: List[str] = []
        scores: List[float | None] = []
        if "top_words" in data:
            tokens = list(data.get("top_words", []))
            raw_scores = data.get("top_word_scores", [])
            scores = [float(s) if isinstance(s, (int, float)) else None for s in raw_scores]
        elif "top_nodes" in data and "node_importance" in data:
            node_imp = data.get("node_importance", []) or []
            node_text = data.get("extras", {}).get("token_text", []) if isinstance(data.get("extras"), dict) else []
            for idx in data.get("top_nodes", []):
                token = node_text[idx] if isinstance(node_text, list) and idx < len(node_text) else f"token_{idx}"
                score = None
                if isinstance(node_imp, list) and idx < len(node_imp):
                    score = float(node_imp[idx])
                tokens.append(token)
                scores.append(score)
        else:
            return ""
        return format_feature_list(tokens, scores, top_k)


class GraphChunkStore:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.chunk_cache: Dict[int, List[nx.Graph]] = {}
        self.chunk_size: int | None = None

    def _load_chunk(self, chunk_idx: int) -> List[nx.Graph]:
        if chunk_idx in self.chunk_cache:
            return self.chunk_cache[chunk_idx]
        chunk_path = self.base_dir / f"{chunk_idx}.pkl"
        if not chunk_path.exists():
            self.chunk_cache[chunk_idx] = []
            return []
        with chunk_path.open("rb") as handle:
            loaded = pickle.load(handle)
        graphs: List[nx.Graph] = []
        if isinstance(loaded, list) and loaded:
            entry = loaded[0]
            if isinstance(entry, tuple):
                graphs = entry[0]
            else:
                graphs = entry
        elif isinstance(loaded, tuple):
            graphs = loaded[0]
        self.chunk_cache[chunk_idx] = graphs
        if self.chunk_size is None and graphs:
            self.chunk_size = len(graphs)
        return graphs

    def get_graph(self, global_index: int) -> nx.Graph | None:
        if self.chunk_size is None:
            self._load_chunk(0)
        chunk = self.chunk_size or 1
        chunk_idx = global_index // chunk
        offset = global_index % chunk
        graphs = self._load_chunk(chunk_idx)
        if not graphs:
            return None
        if offset >= len(graphs):
            return None
        return graphs[offset]


class GraphRepository:
    def __init__(self) -> None:
        self._stores: Dict[Tuple[str, str, str], GraphChunkStore] = {}

    def get_graph(self, dataset_backbone: str, split: str, graph_type: str, global_index: int) -> nx.Graph | None:
        graph_dir = GRAPH_TYPE_DIRS.get(graph_type, graph_type)
        key = (dataset_backbone, split, graph_dir)
        if key not in self._stores:
            base = GRAPH_ROOT / Path(dataset_backbone) / split / graph_dir
            self._stores[key] = GraphChunkStore(base)
        return self._stores[key].get_graph(global_index)


class GNNFeatureRepository:
    def __init__(self) -> None:
        self.graph_repo = GraphRepository()
        self._explanation_cache: Dict[Tuple[str, str, str, str, int], Dict[str, object]] = {}

    def _expl_path(
        self, dataset_backbone: str, graph_type: str, method: str, run_id: str, graph_index: int
    ) -> Path:
        rel = Path(dataset_backbone) / graph_type / "explanations" / method / run_id / "results_split_pickle"
        return GNN_EXPL_ROOT / rel / f"graph_{graph_index:05d}.pkl"

    def _load_explanation(
        self,
        dataset_backbone: str,
        graph_type: str,
        method: str,
        run_id: str,
        graph_index: int,
    ) -> Dict[str, object] | None:
        key = (dataset_backbone, graph_type, method, run_id, graph_index)
        if key in self._explanation_cache:
            return self._explanation_cache[key]
        path = self._expl_path(dataset_backbone, graph_type, method, run_id, graph_index)
        if not path.exists():
            self._explanation_cache[key] = None  # type: ignore[assignment]
            return None
        with path.open("rb") as handle:
            data = pickle.load(handle)
        self._explanation_cache[key] = data
        return data

    def describe(
        self,
        dataset_backbone: str,
        split: str,
        graph_type: str,
        method: str,
        run_id: str | None,
        graph_index: int | None,
        global_index: int,
        top_k: int,
    ) -> str:
        if not run_id:
            return ""
        if graph_index is None or (isinstance(graph_index, float) and math.isnan(graph_index)):
            return ""
        gi = int(graph_index)
        data = self._load_explanation(dataset_backbone, graph_type, method, run_id, gi)
        if not data:
            return ""
        node_sequence: List[int] = []
        scores: List[float | None] = []
        if method == "graphsvx":
            top_nodes = data.get("explanation", {}).get("top_nodes") if isinstance(data.get("explanation"), dict) else None
            if not top_nodes and isinstance(data.get("related_prediction"), dict):
                related = data["related_prediction"]
                top_nodes = related.get("ranked_nodes") or related.get("top_nodes")
            if not top_nodes:
                return ""
            node_sequence = list(top_nodes)
            node_importance = data.get("explanation", {}).get("node_importance") if isinstance(data.get("explanation"), dict) else None
            if isinstance(node_importance, list):
                scores = [float(node_importance[idx]) if idx < len(node_importance) else None for idx in node_sequence]
            else:
                scores = [None] * len(node_sequence)
        else:  # subgraphx
            related = data.get("related_prediction") or {}
            ranked = related.get("ranked_nodes") or related.get("top_nodes")
            if not ranked:
                return ""
            node_sequence = list(ranked)
            scores = [None] * len(node_sequence)

        graph = self.graph_repo.get_graph(dataset_backbone, split, graph_type, global_index)
        tokens = [resolve_node_label(graph, node_id) for node_id in node_sequence]
        return format_feature_list(tokens, scores, top_k)


def format_feature_list(tokens: List[str], scores: List[float | None], top_k: int) -> str:
    entries: List[str] = []
    for token, score in list(zip(tokens, scores))[:top_k]:
        cleaned = str(token)
        if score is not None and not np.isnan(score):
            entries.append(f"{cleaned} ({score:+.3f})")
        else:
            entries.append(cleaned)
    return "; ".join(entries)


def resolve_node_label(graph: nx.Graph | None, node_id: int) -> str:
    if graph is None:
        return f"node_{node_id}"
    if node_id in graph.nodes:
        data = graph.nodes[node_id]
        return data.get("text") or data.get("label") or str(node_id)
    for node, attrs in graph.nodes(data=True):
        if attrs.get("id") == node_id:
            return attrs.get("label") or attrs.get("text") or str(node)
    return str(node_id)


def module_path(dataset: str, method: str, graph: str) -> Path:
    path = DATASET_DIR / dataset / f"module_dataset_{dataset}_{method}_{graph}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing module dataset for {dataset} · {method} · {graph}: {path}")
    return path


def load_module(dataset: str, method: str, graph: str) -> ModulePredictions:
    df = pd.read_csv(module_path(dataset, method, graph))
    if "global_graph_index" not in df.columns:
        raise ValueError(f"Module dataset missing global_graph_index: {dataset} · {method} · {graph}")
    df = df.copy()
    df["global_graph_index"] = df["global_graph_index"].astype(int)
    if "is_correct" not in df.columns:
        raise ValueError(f"Module dataset missing is_correct column: {dataset} · {method} · {graph}")
    df["is_correct"] = df["is_correct"].astype(bool)
    return ModulePredictions(dataset, method, graph, df)


def fitted_logistic_predictions(df: pd.DataFrame) -> ModuleLogisticResult:
    df = df.copy()
    pipeline = None
    feature_cols: List[str] = []
    try:
        X, feature_cols, y = feature_matrix(df)
    except ValueError:
        df["logistic_pred"] = df["is_correct"].astype(int)
        df["logistic_prob"] = df["is_correct"].astype(float)
        df["logistic_detects_error"] = ~df["is_correct"]
        return ModuleLogisticResult(df, None, feature_cols)

    class_counts = np.bincount(y)
    class_counts = class_counts[class_counts > 0]
    if class_counts.size < 2:
        df["logistic_pred"] = y
        df["logistic_prob"] = y.astype(float)
        df["logistic_detects_error"] = (y == 0)
        return ModuleLogisticResult(df, None, feature_cols)

    pipeline = fit_logistic(X, y)
    probs = pipeline.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    df["logistic_pred"] = preds
    df["logistic_prob"] = probs
    df["logistic_detects_error"] = df["logistic_pred"] == 0
    return ModuleLogisticResult(df, pipeline, feature_cols)


def truncated_text(text: str, limit: int = 240) -> str:
    text = (text or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def collect_gnn_hits(
    dataset: str,
    module_frames: Mapping[Tuple[str, str], ModulePredictions],
    top_k_features: int,
) -> pd.DataFrame:
    token_module = module_frames.get(TOKEN_MODULE)
    if token_module is None:
        return pd.DataFrame()

    prepared: Dict[Tuple[str, str], ModuleLogisticResult] = {}
    for key, module in module_frames.items():
        prepared[key] = fitted_logistic_predictions(module.frame)

    token_info = prepared[TOKEN_MODULE]
    token_df = token_info.frame
    if "split" not in token_df.columns:
        inferred_split = None
        for info in prepared.values():
            frame = info.frame
            if "split" in frame.columns and not frame["split"].empty:
                inferred_split = frame["split"].iloc[0]
                break
        if inferred_split is None:
            inferred_split = "test"
        token_df = token_df.copy()
        token_df["split"] = inferred_split
    base_cols = [
        "dataset_slug",
        "dataset_label",
        "dataset_backbone",
        "split",
        "global_graph_index",
        "label",
        "label_text",
        "text",
        "prediction_class",
        "is_correct",
        "logistic_pred",
        "logistic_prob",
    ]
    missing = [col for col in base_cols if col not in token_df.columns]
    if missing:
        raise ValueError(f"Token module missing columns: {missing}")

    merged = token_df[base_cols].copy()
    merged = merged.rename(
        columns={
            "logistic_pred": "token_lr_pred",
            "logistic_prob": "token_lr_prob",
            "is_correct": "token_is_correct",
        }
    )

    detector_cols: List[str] = []
    for method, graph in GNN_MODULES:
        module_key = (method, graph)
        module_info = prepared.get(module_key)
        if module_info is None:
            continue
        prefix = f"{method}_{graph}"
        cols = [
            "global_graph_index",
            "graph_index",
            "logistic_pred",
            "logistic_prob",
            "logistic_detects_error",
        ]
        module_df = module_info.frame
        has_run = "run_id" in module_df.columns
        if has_run:
            cols.append("run_id")
        for col in cols:
            if col not in module_df.columns:
                raise ValueError(f"Module {module_key} missing required column '{col}'")
        renamed = module_df[cols].rename(
            columns=
            {
                "logistic_pred": f"{prefix}_lr_pred",
                "logistic_prob": f"{prefix}_lr_prob",
                "logistic_detects_error": f"{prefix}_detects_error",
                "graph_index": f"{prefix}_graph_index",
                "run_id": f"{prefix}_run_id",
            }
        )
        merged = merged.merge(renamed, on="global_graph_index", how="left")
        detector_cols.append(f"{prefix}_detects_error")

    merged["gnn_detection_count"] = merged[detector_cols].fillna(False).sum(axis=1)

    def detectors_for_row(row: pd.Series) -> str:
        active: List[str] = []
        for col in detector_cols:
            if bool(row.get(col, False)):
                active.append(col.replace("_detects_error", ""))
        return ", ".join(active)

    merged["gnn_detectors"] = merged.apply(detectors_for_row, axis=1)
    merged = merged[(merged["token_is_correct"] == False)]  # noqa: E712
    merged = merged[merged["token_lr_pred"] == 1]
    all_gnn_mask = merged[detector_cols].all(axis=1)
    merged = merged[all_gnn_mask]
    if merged.empty:
        return merged

    merged["text_preview"] = merged["text"].fillna("").map(truncated_text)
    merged = merged.drop(columns=["text"])

    merged.insert(0, "dataset", dataset)

    token_repo = TokenShapRepository()
    gnn_repo = GNNFeatureRepository()

    merged["token_top_features"] = merged.apply(
        lambda row: token_repo.describe(
            row["dataset_backbone"], int(row["global_graph_index"]), top_k_features
        ),
        axis=1,
    )

    for method, graph in GNN_MODULES:
        prefix = f"{method}_{graph}"
        run_col = f"{prefix}_run_id"
        graph_index_col = f"{prefix}_graph_index"
        merged[f"{prefix}_top_features"] = merged.apply(
            lambda row, m=method, g=graph: gnn_repo.describe(
                row["dataset_backbone"],
                row["split"],
                g,
                m,
                row.get(run_col),
                row.get(graph_index_col),
                int(row["global_graph_index"]),
                top_k_features,
            ),
            axis=1,
        )

    return merged.sort_values(["dataset", "token_lr_prob"], ascending=[True, False])


def process_dataset(dataset: str, top_k_features: int) -> pd.DataFrame:
    frames: Dict[Tuple[str, str], ModulePredictions] = {}
    required_modules = [TOKEN_MODULE] + list(GNN_MODULES)
    for method, graph in required_modules:
        try:
            frames[(method, graph)] = load_module(dataset, method, graph)
        except FileNotFoundError:
            continue
    if TOKEN_MODULE not in frames:
        raise RuntimeError(f"TokenSHAP module missing for dataset '{dataset}'.")
    missing_gnn = [mod for mod in GNN_MODULES if mod not in frames]
    if missing_gnn:
        raise RuntimeError(f"Dataset '{dataset}' missing required GNN modules: {missing_gnn}")
    subset = collect_gnn_hits(dataset, frames, top_k_features)
    if subset is None:
        return pd.DataFrame()
    subset.insert(1, "method", "token_shap_llm")
    subset.insert(2, "graph", "tokens")
    return subset


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export cases where tokenSHAP misses errors but GNN explainers detect them.")
    parser.add_argument(
        "--dataset",
        action="append",
        help="Dataset slug(s) to process (default: auto-detect from module directory).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/use_case/token_shap_detection_gaps.csv"),
        help="Path to write the merged CSV report.",
    )
    parser.add_argument(
        "--top-features",
        type=int,
        default=5,
        help="How many top bootstrap features to include per module (default: 5).",
    )
    return parser.parse_args(argv)


def discover_datasets() -> List[str]:
    datasets: List[str] = []
    if not DATASET_DIR.exists():
        return datasets
    for child in DATASET_DIR.iterdir():
        if child.is_dir() and child.name != "coefficients":
            datasets.append(child.name)
    return sorted(datasets)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    datasets = args.dataset or discover_datasets()
    if not datasets:
        raise SystemExit("No datasets found under outputs/use_case/module_datasets.")

    combined: List[pd.DataFrame] = []
    for dataset in datasets:
        subset = process_dataset(dataset, top_k_features=args.top_features)
        if subset.empty:
            continue
        combined.append(subset)

    if not combined:
        raise SystemExit("No qualifying samples found.")

    report = pd.concat(combined, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)
    print(f"Wrote {len(report)} rows -> {args.output}")


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
