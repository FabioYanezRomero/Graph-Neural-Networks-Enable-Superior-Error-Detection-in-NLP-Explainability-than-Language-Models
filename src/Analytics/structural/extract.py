from pathlib import Path
import networkx as nx
import numpy as np
import pickle
from scipy.stats import skew
import pandas as pd
import torch
from tqdm import tqdm
from typing import Optional, Dict, Any, List
from tqdm import tqdm

DATASETS = {
    'ag-news': {
        'output_dir': 'SetFit/ag_news',
        'split': 'test',
    },
    'sst-2': {
        'output_dir': 'stanfordnlp/sst2',
        'split': 'validation',
    },
}

GRAPH_ROOT = Path("/app/outputs/pyg_graphs")
EXPLANATION_ROOT = Path("/app/outputs/gnn_models")
STRUCTURAL_OUTPUT_ROOT = Path("/app/outputs/analytics/structural")


def slugify(value: str) -> str:
    safe = value.replace("/", "_").replace(" ", "_")
    safe = safe.replace("-", "_")
    return "".join(char for char in safe if char.isalnum() or char == "_").lower()


def build_graph(edge_index, num_nodes, directed=False):
    graph_cls = nx.DiGraph if directed else nx.Graph
    G = graph_cls()
    G.add_nodes_from(range(num_nodes))
    edges = edge_index.T.tolist()
    G.add_edges_from(edges)
    return G


def compute_structural_metrics(graph: nx.Graph):
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()

    degrees = [degree for _, degree in graph.degree()]
    if degrees:
        avg_degree = float(np.mean(degrees))
        max_degree = max(degrees)
        degree_variance = float(np.var(degrees))
        degree_skewness = float(skew(degrees)) if len(degrees) > 2 else 0.0
    else:
        avg_degree = 0.0
        max_degree = 0
        degree_variance = 0.0
        degree_skewness = 0.0

    if graph.is_directed():
        n_components = nx.number_strongly_connected_components(graph)
    else:
        n_components = nx.number_connected_components(graph)

    betweenness = nx.betweenness_centrality(graph) if num_nodes > 0 else {}
    avg_betweenness = float(np.mean(list(betweenness.values()))) if betweenness else 0.0
    max_betweenness = max(betweenness.values()) if betweenness else 0.0
    min_betweenness = min(betweenness.values()) if betweenness else 0.0

    closeness = nx.closeness_centrality(graph) if num_nodes > 0 else {}
    avg_closeness = float(np.mean(list(closeness.values()))) if closeness else 0.0
    max_closeness = max(closeness.values()) if closeness else 0.0
    min_closeness = min(closeness.values()) if closeness else 0.0

    try:
        eigenvector = nx.eigenvector_centrality(graph, max_iter=1000) if num_nodes > 0 else {}
        avg_eigenvector = float(np.mean(list(eigenvector.values()))) if eigenvector else 0.0
        max_eigenvector = max(eigenvector.values()) if eigenvector else 0.0
        min_eigenvector = min(eigenvector.values()) if eigenvector else 0.0
    except Exception:
        avg_eigenvector = float('nan')
        max_eigenvector = float('nan')
        min_eigenvector = float('nan')

    return {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'avg_degree': avg_degree,
        'max_degree': max_degree,
        'degree_variance': degree_variance,
        'degree_skewness': degree_skewness,
        'n_components': n_components,
        'avg_betweenness': avg_betweenness,
        'max_betweenness': max_betweenness,
        'min_betweenness': min_betweenness,
        'avg_closeness': avg_closeness,
        'max_closeness': max_closeness,
        'min_closeness': min_closeness,
        'avg_eigenvector': avg_eigenvector,
        'max_eigenvector': max_eigenvector,
        'min_eigenvector': min_eigenvector,
    }


def serialised_to_graph(serialised) -> Optional[nx.Graph]:
    if isinstance(serialised, nx.Graph):
        return serialised
    if not isinstance(serialised, dict):
        return None
    directed = bool(serialised.get("directed", False))
    graph_cls = nx.DiGraph if directed else nx.Graph
    graph = graph_cls()
    nodes = serialised.get("nodes", [])
    for node in nodes:
        if isinstance(node, dict):
            node_id = node.get("id")
            if node_id is None:
                continue
            attrs = node.get("attributes", {})
            graph.add_node(node_id, **attrs)
        else:
            graph.add_node(node)
    edges = serialised.get("edges", [])
    for edge in edges:
        if isinstance(edge, dict):
            src = edge.get("source")
            dst = edge.get("target")
            if src is None or dst is None:
                continue
            attrs = edge.get("attributes", {})
            graph.add_edge(src, dst, **attrs)
        elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
            graph.add_edge(edge[0], edge[1])
    return graph


def load_dataset_graphs(dataset_config, graph_type):
    """Load base graphs for a dataset/graph type split as NetworkX graphs."""
    graph_data_dir = GRAPH_ROOT / dataset_config['output_dir'] / dataset_config['split'] / graph_type
    if not graph_data_dir.is_dir():
        raise FileNotFoundError(f"Graph data directory not found: {graph_data_dir}")

    graph_map: Dict[int, nx.Graph] = {}
    chunk_paths = sorted(graph_data_dir.glob("*.pt"))
    if not chunk_paths:
        raise FileNotFoundError(f"No .pt graph chunks found in {graph_data_dir}")

    for chunk_path in tqdm(chunk_paths, desc=f"Loading graphs ({graph_type})", colour="green"):
        data_list = torch.load(chunk_path)
        for data in data_list:
            global_index = int(getattr(data, "data_index", getattr(data, "idx", -1)))
            if global_index < 0:
                continue
            num_nodes = int(data.num_nodes)
            edge_index = data.edge_index.cpu().numpy()
            directed = bool(getattr(data, "is_directed", False))
            graph = build_graph(edge_index, num_nodes, directed=directed)
            graph_map[global_index] = graph
    return graph_map


def iter_pickled_explanations(explanation_dir: Path):
    for shard_dir in sorted(explanation_dir.iterdir()):
        if not shard_dir.is_dir():
            continue
        split_dir = shard_dir / "results_split_pickle"
        if not split_dir.exists():
            continue
        for pickle_path in sorted(split_dir.glob("graph_*.pkl")):
            yield shard_dir.name, pickle_path


def extract_graph_from_subgraphx(record: Dict[str, Any]) -> Optional[nx.Graph]:
    explanation = record.get("explanation")
    entry = None
    if isinstance(explanation, list) and explanation:
        entry = explanation[0]
    elif isinstance(explanation, dict):
        entry = explanation
    if not isinstance(entry, dict):
        return None
    graph_obj = entry.get("ori_graph") or entry.get("graph") or entry.get("data")
    return serialised_to_graph(graph_obj)


def process_module(dataset_name, dataset_config, graph_type, module):
    explanation_dir = (
        EXPLANATION_ROOT
        / dataset_config['output_dir']
        / graph_type
        / "explanations"
        / module
    )
    if not explanation_dir.exists():
        print(f"Explanation directory missing for {dataset_name} {graph_type} {module}: {explanation_dir}")
        return

    dataset_graphs = None
    if module.lower() == "graphsvx":
        try:
            dataset_graphs = load_dataset_graphs(dataset_config, graph_type)
        except FileNotFoundError as err:
            print(err)
            return

    rows: List[Dict[str, Any]] = []
    for shard_name, pickle_path in tqdm(iter_pickled_explanations(explanation_dir), desc=f"Processing {dataset_name} {graph_type} {module}", colour="green" ):
        with pickle_path.open("rb") as handler:
            try:
                record = pickle.load(handler)
            except Exception as exc:  # pragma: no cover - robustness
                print(f"Failed to load pickle {pickle_path}: {exc}")
                continue

        global_idx = record.get("global_graph_index", record.get("graph_index"))
        if global_idx is None:
            continue

        if module.lower() == "subgraphx":
            graph = extract_graph_from_subgraphx(record)
        else:
            graph = dataset_graphs.get(int(global_idx)) if dataset_graphs else None

        if graph is None:
            continue

        metrics = compute_structural_metrics(graph)

        row = {
            'dataset': dataset_name,
            'graph_type': graph_type,
            'module': module,
            'shard': shard_name,
            'pickle_file': str(pickle_path.relative_to(EXPLANATION_ROOT)),
            'graph_index': record.get("graph_index"),
            'global_graph_index': global_idx,
            'label': record.get("label"),
            'prediction_class': record.get("prediction_class"),
            'prediction_confidence': record.get("prediction_confidence"),
            'is_correct': record.get("is_correct"),
        }
        row.update(metrics)
        rows.append(row)

    if not rows:
        print(f"No structural properties extracted for {dataset_name} {graph_type} {module}")
        return

    df = pd.DataFrame(rows)
    dataset_slug = slugify(dataset_name)
    graph_slug = slugify(graph_type)
    module_slug = slugify(module)

    output_dir = STRUCTURAL_OUTPUT_ROOT / module_slug / dataset_slug
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{graph_slug}.csv"
    df.to_csv(output_path, index=False, sep=';')
    print(f"Saved structural properties to {output_path}")


def main():
    modules = ['subgraphx', 'graphsvx']
    graph_types = ['constituency', 'syntactic', 'skipgrams', 'window']

    for dataset_name, dataset_config in DATASETS.items():
        for graph_type in tqdm(graph_types, desc=f"Processing {dataset_name}", colour="red" ):
            for module in tqdm(modules, desc=f"Processing {dataset_name} {graph_type}", colour="blue" ):
                process_module(dataset_name, dataset_config, graph_type, module)

if __name__ == "__main__":
    main()
