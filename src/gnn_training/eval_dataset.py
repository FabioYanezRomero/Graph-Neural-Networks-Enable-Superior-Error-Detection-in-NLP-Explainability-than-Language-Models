from __future__ import annotations

import glob
import os
import pickle
from dataclasses import dataclass
from typing import Generator, Iterable, List, Sequence

import torch
from torch_geometric.data import Batch, Data


def _load_shard(path: str, file_type: str) -> List[Data]:
    if file_type == "pt":
        data = torch.load(path, map_location="cpu", weights_only=False)
    else:
        with open(path, "rb") as handle:
            data = pickle.load(handle)
    return data if isinstance(data, list) else [data]


@dataclass
class DatasetSummary:
    num_examples: int = 0
    num_node_features: int = 0
    max_label_seen: int = -1

    @property
    def num_classes(self) -> int:
        return max(self.max_label_seen + 1, 0)

    def __len__(self) -> int:
        return self.num_examples


class GraphBatchStreamer:
    def __init__(self, data_dir: str, batch_size: int):
        self.data_dir = data_dir
        self.batch_size = batch_size

        pt_files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        pkl_files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
        if pt_files:
            self.file_paths = pt_files
            self.file_type = "pt"
        elif pkl_files:
            self.file_paths = pkl_files
            self.file_type = "pkl"
        else:
            raise ValueError(f"No .pt or .pkl files found in {data_dir}")

        first_graphs = _load_shard(self.file_paths[0], self.file_type)
        self._prefetched_graphs = first_graphs
        self.summary = DatasetSummary()
        sample_graph = first_graphs[0]
        if hasattr(sample_graph, "x") and sample_graph.x is not None:
            self.summary.num_node_features = sample_graph.x.size(1)

    def _update_summary(self, graphs: Sequence[Data]) -> None:
        self.summary.num_examples += len(graphs)
        for graph in graphs:
            if hasattr(graph, "y") and graph.y is not None:
                y = graph.y
                if hasattr(y, "view"):
                    y_tensor = y.view(-1)
                    if y_tensor.numel():
                        max_val = int(torch.max(y_tensor).item())
                        self.summary.max_label_seen = max(
                            self.summary.max_label_seen, max_val
                        )
                else:
                    self.summary.max_label_seen = max(
                        self.summary.max_label_seen, int(y)
                    )

    def __iter__(self) -> Generator[Batch, None, None]:
        for idx, file_path in enumerate(self.file_paths):
            if idx == 0 and self._prefetched_graphs is not None:
                graphs = self._prefetched_graphs
                self._prefetched_graphs = None
            else:
                graphs = _load_shard(file_path, self.file_type)

            self._update_summary(graphs)

            for start in range(0, len(graphs), self.batch_size):
                chunk = graphs[start : start + self.batch_size]
                yield Batch.from_data_list(chunk)

            del graphs

    def __len__(self) -> int:
        # We cannot know the exact number of batches until iteration finishes.
        # Returning 0 keeps tqdm fallback behavior; evaluate_dual_labels uses hasattr check.
        return 0


def load_graph_stream(data_dir: str, batch_size: int):
    streamer = GraphBatchStreamer(data_dir, batch_size)
    return streamer.summary, streamer
