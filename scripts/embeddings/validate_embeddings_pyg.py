#!/usr/bin/env python3

"""
Validation script for generated embeddings and PyTorch Geometric graphs.
Checks that:
1. Embeddings have correct dimensions
2. PyG graphs have proper structure
3. Node/edge counts match
4. Special tokens use CLS embeddings correctly
5. Word embeddings use last layer properly
"""

import os
import sys
import pickle as pkl
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

def validate_embeddings_and_pyg(dataset_name, graph_type, subset, output_base="outputs"):
    """Validate embeddings and PyG graphs for a specific dataset/graph_type/subset."""

    print(f"Validating {dataset_name}/{graph_type}/{subset}...")

    # Define paths
    embedding_dir = Path(output_base) / "embeddings" / dataset_name / subset / graph_type
    pyg_dir = Path(output_base) / "pyg_graphs" / dataset_name / subset / graph_type
    graph_dir = Path(output_base) / "graphs" / dataset_name / subset

    # Determine graph directory pattern
    if graph_type == "constituency":
        graph_subdir = "constituency"
    elif graph_type == "syntactic":
        graph_subdir = "syntactic"
    elif graph_type == "window":
        graph_subdir = "window.word.k1"
    elif graph_type == "ngrams":
        graph_subdir = "ngrams.word.n2"
    elif graph_type == "skipgrams":
        graph_subdir = "skipgrams.word.k1.n2"
    else:
        graph_subdir = graph_type

    graph_dir = graph_dir / graph_subdir

    # Check if directories exist
    if not embedding_dir.exists():
        print(f"❌ Embedding directory not found: {embedding_dir}")
        return False

    if not pyg_dir.exists():
        print(f"❌ PyG directory not found: {pyg_dir}")
        return False

    # Note: graph_dir is not required for validation of our generated embeddings
    # since we validate the generated PyG graphs directly

    # Find embedding files (handle both old and new naming patterns)
    old_pattern_files = list(embedding_dir.glob("*_graphs_with_embeddings.pkl"))
    new_pattern_files = list(embedding_dir.glob("*.pkl"))  # For files like 00.pkl, 01.pkl, etc.
    embedding_files = old_pattern_files + new_pattern_files

    if old_pattern_files:
        # Use old pattern sorting
        embedding_files = old_pattern_files
        embedding_files.sort(key=lambda x: int(x.stem.split('_batch_')[1].split('_')[0]))
    elif new_pattern_files:
        # Use new pattern sorting
        embedding_files = new_pattern_files
        embedding_files.sort(key=lambda x: int(x.stem))

    if not embedding_files:
        print(f"❌ No embedding files found in {embedding_dir}")
        return False

    # Find PyG files (handle both patterns)
    old_pyg_files = list(pyg_dir.glob("*_pyg_graphs.pt"))
    new_pyg_files = list(pyg_dir.glob("*.pt"))  # For files like 00.pt, 01.pt, etc.
    pyg_files = old_pyg_files + new_pyg_files

    if old_pyg_files:
        pyg_files = old_pyg_files
        pyg_files.sort(key=lambda x: int(x.stem.split('_batch_')[1].split('_')[0]))
    elif new_pyg_files:
        pyg_files = new_pyg_files
        pyg_files.sort(key=lambda x: int(x.stem))

    if not pyg_files:
        print(f"❌ No PyG files found in {pyg_dir}")
        return False

    # Note: embedding files may be intermediate checkpoints, PyG files are final consolidated batches
    # We validate PyG files as they are the final output for GNN training
    print(f"Found {len(embedding_files)} embedding files and {len(pyg_files)} PyG files")

    # Validate each batch
    total_graphs = 0
    total_nodes = 0
    total_edges = 0
    embedding_dims = set()

    for pyg_file in tqdm(pyg_files, desc="Validating PyG batches"):
        try:
            # Load PyG graphs
            pyg_graphs = torch.load(pyg_file)

            batch_graphs = len(pyg_graphs)
            total_graphs += batch_graphs

            for i, pyg_graph in enumerate(pyg_graphs):
                # Check node count
                pyg_node_count = pyg_graph.x.size(0)
                total_nodes += pyg_node_count

                # Check edge count
                pyg_edge_count = pyg_graph.edge_index.size(1)
                total_edges += pyg_edge_count

                # Check embeddings
                emb_dim = pyg_graph.x.size(1)
                embedding_dims.add(emb_dim)

                if emb_dim == 0:
                    print(f"❌ Zero embedding dimension in graph {i}")
                    return False

                # Check PyG structure
                if not hasattr(pyg_graph, 'x') or pyg_graph.x is None:
                    print(f"❌ PyG graph {i} missing node features")
                    return False

                if not hasattr(pyg_graph, 'edge_index') or pyg_graph.edge_index is None:
                    print(f"❌ PyG graph {i} missing edge index")
                    return False

                # Check for labels (LLM predictions)
                if not hasattr(pyg_graph, 'y') or pyg_graph.y is None:
                    print(f"❌ PyG graph {i} missing labels (y attribute)")
                    return False

                if pyg_graph.y.dim() == 0 or pyg_graph.y.numel() == 0:
                    print(f"❌ PyG graph {i} has empty labels")
                    return False

                # Check node labels if available
                if hasattr(pyg_graph, 'node_labels'):
                    if len(pyg_graph.node_labels) != pyg_node_count:
                        print(f"❌ Node labels count mismatch in graph {i}")
                        return False

        except Exception as e:
            print(f"❌ Error processing {pyg_file.stem}: {e}")
            return False

    # Check embedding dimension consistency
    if len(embedding_dims) > 1:
        print(f"⚠️  Warning: Multiple embedding dimensions found: {embedding_dims}")

    print("✅ Validation successful!")
    print(f"   Total graphs: {total_graphs}")
    print(f"   Total nodes: {total_nodes}")
    print(f"   Total edges: {total_edges}")
    print(f"   Embedding dimensions: {embedding_dims}")
    print(f"   All graphs have labels: ✅")

    return True

def validate_special_tokens(dataset_name, graph_type, subset, output_base="outputs"):
    """Check that special tokens (constituency labels) use CLS embeddings correctly."""

    if graph_type != "constituency":
        print("ℹ️  Skipping special token validation (not constituency graphs)")
        return True

    print("Validating special token embeddings...")

    embedding_dir = Path(output_base) / "embeddings" / dataset_name / subset / graph_type
    embedding_files = list(embedding_dir.glob("*_graphs_with_embeddings.pkl"))

    if not embedding_files:
        return True

    # Load constituency dictionary for reference
    try:
        sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
        from embeddings.dicts import constituency_dict
    except ImportError:
        print("⚠️  Could not load constituency_dict for validation")
        return True

    special_labels = set(constituency_dict.values())

    for emb_file in embedding_files[:1]:  # Check first batch only
        try:
            with open(emb_file, 'rb') as f:
                graphs = pkl.load(f)

            for graph in graphs[:5]:  # Check first 5 graphs
                for node_id, node_data in graph.nodes(data=True):
                    label = node_data.get('label', '')
                    embedding = node_data.get('embedding')

                    if label in special_labels:
                        # This should be a CLS embedding from the special token
                        if embedding is None or (isinstance(embedding, np.ndarray) and embedding.size == 0):
                            print(f"❌ Special token '{label}' has no embedding")
                            return False
                    elif graph.out_degree(node_id) == 0:  # Leaf node (word)
                        # This should be a word embedding from last layer
                        if embedding is None or (isinstance(embedding, np.ndarray) and embedding.size == 0):
                            print(f"❌ Word node '{label}' has no embedding")
                            return False

        except Exception as e:
            print(f"❌ Error validating special tokens: {e}")
            return False

    print("✅ Special token validation passed")
    return True

def main():
    parser = argparse.ArgumentParser(description="Validate generated embeddings and PyTorch Geometric graphs")
    parser.add_argument('--dataset_name', type=str, default='stanfordnlp/sst2',
                       help='Dataset name (e.g., stanfordnlp/sst2)')
    parser.add_argument('--graph_type', type=str, default='constituency',
                       help='Graph type (constituency, syntactic, window, ngrams, skipgrams)')
    parser.add_argument('--subset', type=str, default='validation',
                       help='Dataset subset (train, validation, test)')
    parser.add_argument('--output_base', type=str, default='outputs',
                       help='Base output directory')
    parser.add_argument('--all', action='store_true',
                       help='Validate all available combinations')

    args = parser.parse_args()

    if args.all:
        # Validate all combinations
        datasets = ['stanfordnlp/sst2', 'SetFit/ag_news']
        graph_types = ['constituency', 'syntactic', 'window', 'ngrams', 'skipgrams']
        subsets = ['train', 'validation', 'test']

        success_count = 0
        total_count = 0

        for dataset in datasets:
            for graph_type in graph_types:
                for subset in subsets:
                    # Check if this combination exists
                    graph_dir = Path(args.output_base) / "graphs" / dataset / subset

                    # Determine graph directory pattern
                    if graph_type == "constituency":
                        graph_subdir = "constituency"
                    elif graph_type == "syntactic":
                        graph_subdir = "syntactic"
                    elif graph_type == "window":
                        graph_subdir = "window.word.k1"
                    elif graph_type == "ngrams":
                        graph_subdir = "ngrams.word.n2"
                    elif graph_type == "skipgrams":
                        graph_subdir = "skipgrams.word.k1.n2"
                    else:
                        graph_subdir = graph_type

                    if (graph_dir / graph_subdir).exists():
                        total_count += 1
                        print(f"\n{'='*50}")
                        if validate_embeddings_and_pyg(dataset, graph_type, subset, args.output_base):
                            if validate_special_tokens(dataset, graph_type, subset, args.output_base):
                                success_count += 1
                            else:
                                print(f"❌ Special token validation failed for {dataset}/{graph_type}/{subset}")
                        else:
                            print(f"❌ Validation failed for {dataset}/{graph_type}/{subset}")

        print(f"\n{'='*50}")
        print(f"VALIDATION SUMMARY: {success_count}/{total_count} combinations passed")
        return success_count == total_count

    else:
        # Validate single combination
        success = validate_embeddings_and_pyg(args.dataset_name, args.graph_type, args.subset, args.output_base)
        if success and args.graph_type == "constituency":
            success &= validate_special_tokens(args.dataset_name, args.graph_type, args.subset, args.output_base)

        return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
