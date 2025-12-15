import pickle as pkl
import numpy as np
import os
import sys

def is_zero_embedding(embedding):
    embedding = np.asarray(embedding)
    return np.allclose(embedding, 0)

def analyze_zero_embeddings(pkl_dir):
    total_nodes = 0
    zero_nodes = 0
    for fname in os.listdir(pkl_dir):
        if not fname.endswith('.pkl'):
            continue
        with open(os.path.join(pkl_dir, fname), 'rb') as f:
            graphs = pkl.load(f)
            for graph in graphs:
                for _, data in graph.nodes(data=True):
                    emb = data.get('embedding', None)
                    if emb is not None:
                        total_nodes += 1
                        if is_zero_embedding(emb):
                            zero_nodes += 1
    if total_nodes == 0:
        print('No embeddings found.')
        return
    percent = 100 * zero_nodes / total_nodes
    print(f'Total nodes with embeddings: {total_nodes}')
    print(f'Nodes with zero embeddings: {zero_nodes}')
    print(f'Percentage of zero embeddings: {percent:.2f}%')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python test_zero_embeddings.py <path_to_graphs_with_embeddings_dir>')
        sys.exit(1)
    analyze_zero_embeddings(sys.argv[1])
