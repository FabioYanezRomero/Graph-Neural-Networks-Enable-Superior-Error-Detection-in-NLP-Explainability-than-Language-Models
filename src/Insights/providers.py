from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx

try:  # Torch is heavy; import lazily when available.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


@dataclass(frozen=True)
class GraphInfo:
    """Container conveying the NetworkX graph plus node-level metadata."""

    graph: nx.DiGraph | nx.Graph
    node_names: Sequence[int | str]
    node_text: Sequence[str]

    def text_for_indices(self, indices: Iterable[int]) -> List[str]:
        return [self.node_text[i] for i in indices if 0 <= i < len(self.node_text)]


class GraphArtifactProvider:
    """
    Loads graph artefacts stored under ``outputs/graphs`` (NetworkX) and
    ``outputs/pyg_graphs`` (PyG tensors) to enrich Insight summaries with
    structural metrics and human-readable node text.
    """

    def __init__(
        self,
        graph_root: Path | str = Path("outputs/graphs"),
        pyg_root: Path | str = Path("outputs/pyg_graphs"),
        *,
        strict: bool = False,
    ) -> None:
        self.graph_root = Path(graph_root)
        self.pyg_root = Path(pyg_root)
        self.strict = strict

        self._nx_chunk_size: Dict[Path, int] = {}
        self._pyg_chunk_size: Dict[Path, int] = {}
        self._nx_cache: Dict[Tuple[Path, int], List[nx.DiGraph]] = {}
        self._pyg_cache: Dict[Tuple[Path, int], List] = {}

        if torch is None:
            raise ImportError("torch is required to read PyG artefacts.")

    def __call__(self, record) -> Optional[GraphInfo]:  # type: ignore[override]
        try:
            return self._load_graph_info(record)
        except FileNotFoundError:
            if self.strict:
                raise
            return None
        except Exception:
            if self.strict:
                raise
            return None

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _load_graph_info(self, record) -> GraphInfo:
        extras = getattr(record, "extras", {}) or {}
        backbone_label = extras.get("backbone")

        backbone, dataset_parts = self._parse_dataset(record.dataset, override_backbone=backbone_label)
        split = self._resolve_split(record)

        nx_dir = self._resolve_nx_dir(backbone, dataset_parts, split, record.graph_type)
        pyg_dir = self._resolve_pyg_dir(backbone, dataset_parts, split, record.graph_type)

        graph = self._load_networkx_graph(nx_dir, record.graph_index)
        node_names = self._load_node_names(pyg_dir, record.graph_index)
        node_text = self._token_text_from_graph(graph, node_names)
        return GraphInfo(graph=graph, node_names=node_names, node_text=node_text)

    def _parse_dataset(self, dataset: str, *, override_backbone: Optional[str] = None) -> Tuple[str, Path]:
        parts = dataset.split("/")
        if override_backbone:
            backbone = override_backbone
            dataset_parts = Path(*[p for p in parts if p != override_backbone])
            return backbone, dataset_parts

        if len(parts) == 1:
            backbone = parts[0]
            dataset_parts = Path()
        else:
            backbone = parts[0]
            dataset_parts = Path(*parts[1:])
        return backbone, dataset_parts

    def _resolve_split(self, record) -> str:
        extras = getattr(record, "extras", {}) or {}
        split = extras.get("split")
        if split:
            return str(split)
        # Fall back to common defaults
        if hasattr(record, "method") and str(record.method).lower() == "graphsvx":
            return "test"
        return "validation"

    def _resolve_nx_dir(self, backbone: str, dataset_parts: Path, split: str, graph_type: Optional[str]) -> Path:
        base = self.graph_root / backbone / dataset_parts / split
        if not base.exists():
            raise FileNotFoundError(f"Graph directory not found: {base}")

        if graph_type:
            direct = base / graph_type
            if direct.exists():
                return direct
            matches = sorted(
                (p for p in base.iterdir() if p.is_dir() and p.name.startswith(graph_type)),
                key=lambda p: len(p.name),
            )
            if matches:
                return matches[0]
        raise FileNotFoundError(f"No graph variant matching '{graph_type}' under {base}")

    def _resolve_pyg_dir(self, backbone: str, dataset_parts: Path, split: str, graph_type: Optional[str]) -> Path:
        base = self.pyg_root / backbone / dataset_parts / split
        candidate = base / str(graph_type)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"No PyG directory for graph type '{graph_type}' under {base}")

    def _load_networkx_graph(self, directory: Path, index: int) -> nx.DiGraph:
        chunk_size = self._nx_chunk_size.get(directory)
        if chunk_size is None:
            chunk_size = self._infer_nx_chunk_size(directory)
            self._nx_chunk_size[directory] = chunk_size

        chunk_idx = index // chunk_size
        offset = index % chunk_size
        graphs = self._nx_chunk(chunk_idx, directory)
        return graphs[offset]

    def _infer_nx_chunk_size(self, directory: Path) -> int:
        first = self._first_file(directory, "*.pkl")
        graphs = self._extract_graphs(first)
        if not graphs:
            raise ValueError(f"No graphs stored in {first}")
        return len(graphs)

    def _nx_chunk(self, chunk_idx: int, directory: Path) -> List[nx.DiGraph]:
        key = (directory, chunk_idx)
        cached = self._nx_cache.get(key)
        if cached is not None:
            return cached

        file_path = directory / f"{chunk_idx}.pkl"
        graphs = self._extract_graphs(file_path)
        self._nx_cache[key] = graphs
        # Simple LRU: keep cache small
        if len(self._nx_cache) > 8:
            self._nx_cache.pop(next(iter(self._nx_cache)))
        return graphs

    def _extract_graphs(self, file_path: Path) -> List[nx.DiGraph]:
        with file_path.open("rb") as handle:
            payload = pickle.load(handle)

        if isinstance(payload, list):
            if payload and isinstance(payload[0], tuple):
                graphs = payload[0][0]
            else:
                graphs = []
                for item in payload:
                    if isinstance(item, tuple):
                        graphs.extend(item[0])
        elif isinstance(payload, tuple):
            graphs = payload[0]
        else:  # pragma: no cover - unexpected format
            raise TypeError(f"Unexpected graph payload in {file_path!s}: {type(payload)}")

        if not graphs:
            raise ValueError(f"No graphs found in {file_path}")
        return graphs

    def _load_node_names(self, directory: Path, index: int) -> Sequence[int | str]:
        chunk_size = self._pyg_chunk_size.get(directory)
        if chunk_size is None:
            chunk_size = self._infer_pyg_chunk_size(directory)
            self._pyg_chunk_size[directory] = chunk_size

        chunk_idx = index // chunk_size
        offset = index % chunk_size
        data_objects = self._pyg_chunk(directory, chunk_idx)
        item = data_objects[offset]
        return list(getattr(item, "nx_node_names", []))

    def _infer_pyg_chunk_size(self, directory: Path) -> int:
        first = self._first_file(directory, "*.pt")
        data_objects = self._load_pyg_file(first)
        if not data_objects:
            raise ValueError(f"No PyG entries in {first}")
        return len(data_objects)

    def _pyg_chunk(self, directory: Path, chunk_idx: int):
        key = (directory, chunk_idx)
        cached = self._pyg_cache.get(key)
        if cached is not None:
            return cached

        file_path = directory / f"{chunk_idx:05d}.pt"
        data_objects = self._load_pyg_file(file_path)
        self._pyg_cache[key] = data_objects
        if len(self._pyg_cache) > 8:
            self._pyg_cache.pop(next(iter(self._pyg_cache)))
        return data_objects

    def _load_pyg_file(self, file_path: Path):
        if torch is None:  # pragma: no cover
            raise ImportError("torch is required to load PyG artefacts.")
        return torch.load(file_path)

    def _first_file(self, directory: Path, pattern: str) -> Path:
        candidates = sorted(directory.glob(pattern))
        if not candidates:
            raise FileNotFoundError(f"No files matching {pattern} under {directory}")
        return candidates[0]

    def _token_text_from_graph(
        self,
        graph: nx.DiGraph,
        node_names: Sequence[int | str],
    ) -> List[str]:
        text: List[str] = []
        for name in node_names:
            attr = graph.nodes[name] if graph.has_node(name) else {}
            token = attr.get("text") or attr.get("label") or str(name)
            text.append(token)
        return text
