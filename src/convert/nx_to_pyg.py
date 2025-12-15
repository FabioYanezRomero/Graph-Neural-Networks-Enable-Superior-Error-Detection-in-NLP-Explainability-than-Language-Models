import torch
from torch_geometric.data import Data
import networkx as nx
import numpy as np

def nx_to_pyg(graph: nx.DiGraph):
    """
    Converts a NetworkX directed graph to a PyTorch Geometric Data object.
    Preserves node 'label' (and 'embedding' if present).

    Args:
        graph (nx.DiGraph): The input NetworkX graph.

    Returns:
        Data: PyTorch Geometric Data object.
    """
    # Map node names to indices
    node_list = list(graph.nodes())
    node_idx = {node: i for i, node in enumerate(node_list)}

    # Node features: use 'embedding' if present, else zeros
    embeddings = []
    labels = []
    emb_dim = None
    # Find embedding dimension from first available embedding
    for node in node_list:
        attr = graph.nodes[node]
        emb = attr.get('embedding', None)
        if emb is not None:
            emb = np.asarray(emb)
            if emb.ndim > 1:
                emb = emb.flatten()
            emb_dim = emb.shape[0]
            break
    if emb_dim is None:
        emb_dim = 32  # fallback default
        import warnings
        warnings.warn("No node in this graph has an 'embedding' attribute; using 32-dim zero vectors.")
    for node in node_list:
        attr = graph.nodes[node]
        emb = attr.get('embedding', None)
        if emb is not None:
            emb = np.asarray(emb)
            if emb.ndim > 1:
                emb = emb.flatten()
            if emb.shape[0] != emb_dim:
                raise ValueError(f"Node {node} embedding shape {emb.shape} does not match expected {emb_dim}")
        else:
            emb = np.zeros(emb_dim)
        embeddings.append(emb)
        labels.append(attr.get('label', None))
    x = torch.tensor(np.stack(embeddings), dtype=torch.float)

    # Edge index
    edge_index = [[], []]
    edge_labels = []
    for src, dst, edata in graph.edges(data=True):
        edge_index[0].append(node_idx[src])
        edge_index[1].append(node_idx[dst])
        edge_labels.append(edata.get('label', None))
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Store node labels as a list of strings in Data object
    data = Data(x=x, edge_index=edge_index)
    data.node_labels = labels
    data.edge_labels = edge_labels
    data.nx_node_names = node_list  # For reference

    # Add data index information for label mapping (assuming single graph per data sample)
    # This is a placeholder - you may need to adjust based on your graph creation process
    data.data_index = 0  # Default index, should be set by the calling code

    return data

def nx_list_to_pyg(graphs):
    """
    Converts a list of NetworkX graphs to a list of PyG Data objects.
    """
    return [nx_to_pyg(g) for g in graphs]

if __name__ == "__main__":
    # Example usage: load a graph and convert
    import networkx as nx
    G = nx.DiGraph()
    G.add_node('A', label='alpha', embedding=np.random.randn(32))
    G.add_node('B', label='beta', embedding=np.random.randn(32))
    G.add_edge('A', 'B', label='connects')
    data = nx_to_pyg(G)
    print(data)
    print("Node labels:", data.node_labels)
    print("Edge labels:", data.edge_labels)
    print("Node names:", data.nx_node_names)
