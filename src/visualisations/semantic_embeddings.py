"""Embedding-based visualisations for explanation analysis."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch_geometric.data import Data
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

DEFAULT_PYG_ROOT = Path("outputs/pyg_graphs")
DEFAULT_CENTROID_OUTPUT = Path("outputs/analytics/embedding/centroids")
DEFAULT_CORRELATION_OUTPUT = Path("outputs/analytics/embedding/correlation")
DEFAULT_NEIGHBOR_OUTPUT = Path("outputs/analytics/embedding/neighborhoods")

PREFERRED_SPLITS = ("test", "validation", "train")


def _iter_token_csvs(root: Path, pattern: str = "*tokens.csv") -> Iterable[Path]:
    return sorted(Path(root).rglob(pattern))


class PyGEmbeddingProvider:
    """Efficient loader for PyG graph chunks with cached access."""

    def __init__(self, root: Path | str = DEFAULT_PYG_ROOT, *, max_cache: int = 4) -> None:
        self.root = Path(root)
        self.max_cache = max_cache
        self._chunk_sizes: Dict[Path, int] = {}
        self._chunk_cache: Dict[Path, List] = {}
        self._cache_order: List[Path] = []

    def _resolve_dir(self, dataset_label: str, graph_type: str) -> Path:
        if "/" in dataset_label:
            backbone, dataset = dataset_label.split("/", 1)
        else:
            backbone, dataset = dataset_label, ""
        base = self.root / backbone / dataset
        for split in PREFERRED_SPLITS:
            candidate = base / split / graph_type
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No PyG directory found for {dataset_label}/{graph_type}")

    def _chunk_size(self, directory: Path) -> int:
        size = self._chunk_sizes.get(directory)
        if size is not None:
            return size
        first_file = sorted(directory.glob("*.pt"))
        if not first_file:
            raise FileNotFoundError(f"No chunk files in {directory}")
        data_list = torch.load(first_file[0])
        size = len(data_list)
        self._chunk_sizes[directory] = size
        self._chunk_cache[first_file[0]] = data_list
        self._cache_order.append(first_file[0])
        return size

    def _load_chunk(self, file_path: Path) -> List:
        cached = self._chunk_cache.get(file_path)
        if cached is not None:
            return cached
        data_list = torch.load(file_path)
        self._chunk_cache[file_path] = data_list
        self._cache_order.append(file_path)
        if len(self._cache_order) > self.max_cache:
            oldest = self._cache_order.pop(0)
            self._chunk_cache.pop(oldest, None)
        return data_list

    def get_graph(self, dataset_label: str, graph_type: str, graph_index: int) -> Data:
        directory = self._resolve_dir(dataset_label, graph_type)
        chunk_size = self._chunk_size(directory)
        chunk_idx = graph_index // chunk_size
        offset = graph_index % chunk_size
        file_path = directory / f"{chunk_idx:05d}.pt"
        if not file_path.exists():
            raise FileNotFoundError(f"Chunk {file_path} not found for graph_index {graph_index}")
        data_list = self._load_chunk(file_path)
        if offset >= len(data_list):
            raise IndexError(f"Graph index {graph_index} exceeds chunk entries in {file_path}")
        data = data_list[offset]
        data_idx = getattr(data, "data_index", offset + chunk_idx * chunk_size)
        if data_idx != graph_index:
            raise ValueError(f"Graph index mismatch: expected {graph_index}, found {data_idx}")
        return data


def _prepare_label_lookup(node_labels: Sequence) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    exact: Dict[str, List[int]] = defaultdict(list)
    lower: Dict[str, List[int]] = defaultdict(list)
    for idx, label in enumerate(node_labels):
        val = str(label)
        exact[val].append(idx)
        lower[val.lower()].append(idx)
    return exact, lower


def _match_token_index(token: str, exact: Dict[str, List[int]], lower: Dict[str, List[int]], used: set[int]) -> Optional[int]:
    if not isinstance(token, str):
        token = str(token)
    for candidate in exact.get(token, []):
        if candidate not in used:
            used.add(candidate)
            return candidate
    token_lower = token.lower()
    for candidate in lower.get(token_lower, []):
        if candidate not in used:
            used.add(candidate)
            return candidate
    return None


def _is_structural_label(label: str) -> bool:
    return label.startswith("«") and label.endswith("»")


@dataclass
class EmbeddingRecord:
    embedding: np.ndarray
    score: float
    label: Optional[str]
    is_correct: Optional[bool]
    kind: str
    token: str


def _collect_embeddings_from_csv(
    tokens_csv: Path,
    provider: PyGEmbeddingProvider,
    *,
    baseline: bool = True,
    random_state: int = 0,
    max_records: Optional[int] = None,
) -> Tuple[List[EmbeddingRecord], str, str]:
    df = pd.read_csv(tokens_csv)
    if df.empty:
        return [], "", ""
    dataset_label = df["dataset"].iloc[0]
    graph_type = df["graph_type"].iloc[0]

    rng = np.random.default_rng(random_state)
    records: List[EmbeddingRecord] = []
    for graph_index, group in df.groupby("graph_index"):
        try:
            data = provider.get_graph(dataset_label, graph_type, int(graph_index))
        except Exception:
            continue
        node_labels = [str(lbl) for lbl in getattr(data, "node_labels", [])]
        exact, lower = _prepare_label_lookup(node_labels)
        used_nodes: set[int] = set()
        graph_label = group["label"].iloc[0] if "label" in group.columns else None
        graph_correct = group["is_correct"].iloc[0] if "is_correct" in group.columns else None

        for _, row in group.iterrows():
            idx = _match_token_index(row["token"], exact, lower, used_nodes)
            if idx is None:
                continue
            embedding = data.x[idx].detach().cpu().numpy()
            records.append(
                EmbeddingRecord(
                    embedding=embedding,
                    score=float(row.get("score", 0.0)),
                    label=str(row.get("label")) if "label" in group.columns else None,
                    is_correct=bool(row.get("is_correct")) if "is_correct" in group.columns else None,
                    kind="important",
                    token=row["token"],
                )
            )
            if max_records and len(records) >= max_records:
                return records, dataset_label, graph_type

        if not baseline:
            continue

        structural_mask = [not _is_structural_label(lbl) for lbl in node_labels]
        candidate_indices = [idx for idx, keep in enumerate(structural_mask) if keep and idx not in used_nodes]
        sample_size = min(len(candidate_indices), len(used_nodes))
        if sample_size == 0:
            continue
        baseline_indices = rng.choice(candidate_indices, size=sample_size, replace=False)
        for idx in baseline_indices:
            embedding = data.x[int(idx)].detach().cpu().numpy()
            records.append(
                EmbeddingRecord(
                    embedding=embedding,
                    score=0.0,
                    label=str(graph_label) if graph_label is not None else None,
                    is_correct=bool(graph_correct) if graph_correct is not None else None,
                    kind="baseline",
                    token=str(node_labels[int(idx)]),
                )
            )
            if max_records and len(records) >= max_records:
                return records, dataset_label, graph_type

    return records, dataset_label, graph_type


def _records_to_arrays(records: List[EmbeddingRecord]) -> Tuple[pd.DataFrame, np.ndarray]:
    if not records:
        return pd.DataFrame(), np.empty((0, 0))
    embeddings = np.vstack([rec.embedding for rec in records])
    metadata = pd.DataFrame(
        {
            "score": [rec.score for rec in records],
            "label": [rec.label for rec in records],
            "is_correct": [rec.is_correct for rec in records],
            "kind": [rec.kind for rec in records],
            "token": [rec.token for rec in records],
        }
    )
    return metadata, embeddings


def _pca_project(embeddings: np.ndarray, components: int = 2, random_state: int = 0) -> Tuple[np.ndarray, PCA]:
    n_components = min(components, embeddings.shape[0], embeddings.shape[1])
    if n_components < 1:
        raise ValueError("Not enough samples for PCA")
    pca = PCA(n_components=n_components, random_state=random_state)
    projected = pca.fit_transform(embeddings)
    return projected, pca


def generate_embedding_centroids(
    tokens_root: Path | str,
    pyg_root: Path | str = DEFAULT_PYG_ROOT,
    output_root: Path | str = DEFAULT_CENTROID_OUTPUT,
    *,
    pattern: str = "*tokens.csv",
    random_state: int = 0,
) -> List[Path]:
    provider = PyGEmbeddingProvider(root=pyg_root)
    tokens_root = Path(tokens_root)
    out_dir = Path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    produced: List[Path] = []
    for csv_path in _iter_token_csvs(tokens_root, pattern):
        records, dataset_label, graph_type = _collect_embeddings_from_csv(
            csv_path,
            provider,
            baseline=False,
            max_records=5000,
        )
        if not records:
            continue
        df_meta, embeddings = _records_to_arrays(records)
        mask = df_meta["kind"] == "important"
        if mask.sum() == 0 or df_meta["label"].isna().all():
            continue
        important_embeddings = embeddings[mask.to_numpy()]
        important_meta = df_meta[mask].reset_index(drop=True)
        if "is_correct" not in important_meta.columns or important_meta["is_correct"].dropna().empty:
            continue
        correctness_groups = important_meta.groupby("is_correct", dropna=False)

        relative = csv_path.relative_to(tokens_root)
        base_dir = out_dir / relative.parent
        base_dir.mkdir(parents=True, exist_ok=True)

        for correctness_flag, subset_meta in correctness_groups:
            if subset_meta.empty or pd.isna(correctness_flag):
                continue
            labels = subset_meta["label"].dropna()
            if labels.empty:
                continue
            centroid_groups = subset_meta[subset_meta["label"].notna()].groupby("label")

            centroids: Dict[str, np.ndarray] = {}
            for label, grp in centroid_groups:
                if grp.empty:
                    continue
                idx = grp.index.to_numpy()
                centroids[str(label)] = important_embeddings[idx].mean(axis=0)

            if len(centroids) < 2:
                continue

            centroid_labels = list(centroids.keys())
            centroid_matrix = np.vstack([centroids[label] for label in centroid_labels])
            try:
                projected, _ = _pca_project(
                    centroid_matrix,
                    components=min(2, centroid_matrix.shape[0]),
                    random_state=random_state,
                )
            except ValueError:
                continue

            pc1 = projected[:, 0]
            pc2 = projected[:, 1] if projected.shape[1] > 1 else np.zeros_like(pc1)
            centroids_df = pd.DataFrame({"label": centroid_labels, "pc1": pc1, "pc2": pc2}).set_index("label")

            status = "correct" if correctness_flag is True else "incorrect" if correctness_flag is False else "mixed"
            label_suffix = status if status != "mixed" else "overall"
            target_dir = base_dir / label_suffix
            target_dir.mkdir(parents=True, exist_ok=True)

            sns.set_theme(style="whitegrid", context="paper")
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(centroids_df["pc1"], centroids_df["pc2"], s=90, color="#4C72B0")
            for label, row in centroids_df.iterrows():
                ax.text(row["pc1"], row["pc2"], str(label), fontsize=10, weight="bold")
            ax.set_xlabel("Centroid PC1")
            ax.set_ylabel("Centroid PC2")
            title_status = "Correct" if correctness_flag is True else "Incorrect" if correctness_flag is False else "Combined"
            ax.set_title(f"Class centroids ({title_status}) - {dataset_label} {graph_type}")
            fig.tight_layout()
            scatter_path = target_dir / f"{csv_path.stem}_{label_suffix}_centroids_pca.png"
            fig.savefig(scatter_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            produced.append(scatter_path)

            dist_matrix = np.linalg.norm(centroid_matrix[:, None, :] - centroid_matrix[None, :, :], axis=-1)
            dists = pd.DataFrame(dist_matrix, index=centroids_df.index, columns=centroids_df.index)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(dists, cmap="Blues", annot=True, fmt=".2f", square=True, ax=ax)
            ax.set_title(f"Centroid distances ({title_status}) - {dataset_label} {graph_type}")
            fig.tight_layout()
            heatmap_path = target_dir / f"{csv_path.stem}_{label_suffix}_centroid_distances.png"
            fig.savefig(heatmap_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            produced.append(heatmap_path)

    return produced


def generate_embedding_importance_correlation(
    tokens_root: Path | str,
    pyg_root: Path | str = DEFAULT_PYG_ROOT,
    output_root: Path | str = DEFAULT_CORRELATION_OUTPUT,
    *,
    pattern: str = "*tokens.csv",
    components: int = 5,
    random_state: int = 0,
) -> List[Path]:
    provider = PyGEmbeddingProvider(root=pyg_root)
    tokens_root = Path(tokens_root)
    out_dir = Path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    produced: List[Path] = []
    for csv_path in _iter_token_csvs(tokens_root, pattern):
        records, dataset_label, graph_type = _collect_embeddings_from_csv(
            csv_path,
            provider,
            baseline=False,
            max_records=5000,
        )
        if not records:
            continue
        df_meta, embeddings = _records_to_arrays(records)
        mask = df_meta["kind"] == "important"
        if mask.sum() == 0:
            continue
        embeddings = embeddings[mask.to_numpy()]
        df = df_meta[mask].reset_index(drop=True)
        try:
            projected, pca = _pca_project(embeddings, components=min(components, embeddings.shape[1]), random_state=random_state)
        except ValueError:
            continue
        component_cols = [f"pc{i+1}" for i in range(projected.shape[1])]
        corr_df = pd.DataFrame(projected, columns=component_cols)
        corr_df["score"] = df["score"].to_numpy()
        corr_df["norm"] = np.linalg.norm(embeddings, axis=1)
        corr = corr_df.corr()[["score", "norm"]].drop(index=["score", "norm"], errors="ignore")
        corr = corr.dropna(how="all", axis=0).dropna(how="all", axis=1)
        if corr.empty:
            continue

        relative = csv_path.relative_to(tokens_root)
        target_dir = out_dir / relative.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        output_path = target_dir / f"{csv_path.stem}_importance_component_correlation.png"
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        ax.set_title(f"Importance correlation - {dataset_label} {graph_type}")
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        produced.append(output_path)

    return produced


def generate_embedding_neighborhoods(
    tokens_root: Path | str,
    pyg_root: Path | str = DEFAULT_PYG_ROOT,
    output_root: Path | str = DEFAULT_NEIGHBOR_OUTPUT,
    *,
    pattern: str = "*tokens.csv",
    k_neighbors: int = 5,
    sample_limit: int = 2000,
    random_state: int = 0,
) -> List[Path]:
    provider = PyGEmbeddingProvider(root=pyg_root)
    tokens_root = Path(tokens_root)
    out_dir = Path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    produced: List[Path] = []
    for csv_path in _iter_token_csvs(tokens_root, pattern):
        records, dataset_label, graph_type = _collect_embeddings_from_csv(
            csv_path,
            provider,
            baseline=True,
            random_state=random_state,
            max_records=sample_limit * 2 if sample_limit else None,
        )
        if not records:
            continue
        df_meta, embeddings = _records_to_arrays(records)
        if embeddings.size == 0:
            continue
        if embeddings.shape[0] > sample_limit:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(embeddings.shape[0], size=sample_limit, replace=False)
            df_meta = df_meta.iloc[idx]
            embeddings = embeddings[idx]

        important_mask = df_meta["kind"] == "important"
        if important_mask.sum() == 0:
            continue
        neighbor_model = NearestNeighbors(n_neighbors=min(k_neighbors + 1, embeddings.shape[0]))
        neighbor_model.fit(embeddings)
        distances, indices = neighbor_model.kneighbors(embeddings[important_mask])
        neighbor_labels = []
        neighbor_correct = []
        for row_idx, neigh_indices in zip(df_meta[important_mask].index, indices):
            neigh_indices = neigh_indices[1:]  # exclude self
            labels = df_meta.iloc[neigh_indices]["label"].astype(str).tolist()
            correct = df_meta.iloc[neigh_indices]["is_correct"].tolist()
            neighbor_labels.append(labels)
            neighbor_correct.append(correct)

        summary = pd.DataFrame(
            {
                "token": df_meta[important_mask]["token"].tolist(),
                "label": df_meta[important_mask]["label"].astype(str).tolist(),
                "score": df_meta[important_mask]["score"].tolist(),
                "neighbor_labels": neighbor_labels,
                "neighbor_correct": neighbor_correct,
            }
        )
        summary["matching_neighbors"] = summary.apply(
            lambda row: sum(lbl == row["label"] for lbl in row["neighbor_labels"]), axis=1
        )
        summary["correct_neighbors"] = summary.apply(
            lambda row: sum(bool(val) for val in row["neighbor_correct"]), axis=1
        )

        relative = csv_path.relative_to(tokens_root)
        target_dir = out_dir / relative.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        output_path = target_dir / f"{csv_path.stem}_neighbor_stats.csv"
        summary.to_csv(output_path, index=False)
        produced.append(output_path)

    return produced
