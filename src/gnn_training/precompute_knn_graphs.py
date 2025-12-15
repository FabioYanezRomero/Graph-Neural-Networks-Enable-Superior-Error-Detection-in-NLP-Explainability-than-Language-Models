"""precompute_knn_graphs.py

Script to precompute k-NN sparsified graphs from fully connected graphs and save them in batches.
This eliminates the need for on-the-fly k-NN computation during training.

Usage:
    python precompute_knn_graphs.py --input_dir /path/to/fully_connected/graphs --output_dir /path/to/knn/graphs --k 8
"""

import os
import glob
import pickle
import argparse
from tqdm import tqdm
import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
import gc


def sparsify_graph_knn(data: Data, k: int = 8) -> Data:
    """Create k-nearest neighbor graph from node features with memory optimization."""
    x = data.x
    num_nodes = x.size(0)
    
    # For very large graphs, use a more memory-efficient approach
    if num_nodes > 100:
        # Process in chunks to avoid memory issues
        chunk_size = min(50, num_nodes)
        source_nodes = []
        target_nodes = []
        
        for i in range(0, num_nodes, chunk_size):
            end_i = min(i + chunk_size, num_nodes)
            chunk_x = x[i:end_i]
            
            # Compute distances only for this chunk
            with torch.no_grad():
                dist_chunk = torch.cdist(chunk_x, x, p=2)
                _, knn_indices = torch.topk(dist_chunk, k=min(k, num_nodes), dim=1, largest=False)
                k_eff = knn_indices.size(1)  # Effective k for this chunk
                
                # Add edges for this chunk
                chunk_sources = torch.arange(i, end_i).repeat_interleave(k_eff)
                chunk_targets = knn_indices.flatten()
                
                source_nodes.extend(chunk_sources.tolist())
                target_nodes.extend(chunk_targets.tolist())
                
                # Clear intermediate tensors
                del dist_chunk, knn_indices
        
        source_nodes = torch.tensor(source_nodes, dtype=torch.long)
        target_nodes = torch.tensor(target_nodes, dtype=torch.long)
    else:
        # Original approach for smaller graphs
        with torch.no_grad():
            dist_matrix = torch.cdist(x, x, p=2)
            _, knn_indices = torch.topk(dist_matrix, k=min(k, num_nodes), dim=1, largest=False)
            k_eff = knn_indices.size(1)
            
            source_nodes = torch.arange(num_nodes).repeat_interleave(k_eff)
            target_nodes = knn_indices.flatten()
            
            del dist_matrix, knn_indices
    
    # Remove out-of-bounds indices
    valid_mask = target_nodes < num_nodes
    source_nodes = source_nodes[valid_mask]
    target_nodes = target_nodes[valid_mask]
    
    edge_index = torch.stack([source_nodes, target_nodes], dim=0)
    
    # Ensure self-loops are included
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    
    return Data(x=data.x, edge_index=edge_index, y=data.y)


def process_batch_file(input_path: str, output_path: str, k: int):
    """Process a single batch file and save k-NN sparsified graphs."""
    print(f"Processing {input_path}...")
    
    # Load original batch
    with open(input_path, 'rb') as f:
        batch = pickle.load(f)
    
    # Sparsify all graphs in the batch
    sparsified_batch = []
    for i, data in enumerate(tqdm(batch, desc="Sparsifying graphs", leave=False)):
        sparse_data = sparsify_graph_knn(data, k=k)
        sparsified_batch.append(sparse_data)
        
        # Clear memory periodically
        if i % 100 == 0:
            gc.collect()
    
    # Save sparsified batch
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(sparsified_batch, f)
    
    print(f"Saved {len(sparsified_batch)} k-NN graphs to {output_path}")
    
    # Clear memory
    del batch, sparsified_batch
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description='Precompute k-NN sparsified graphs')
    parser.add_argument('--input_dir', type=str, default="outputs/embeddings/fully_connected/stanfordnlp/sst2/train/train",
                       help='Directory containing fully connected graph batch files')
    parser.add_argument('--output_dir', type=str, default="outputs/embeddings/knn/stanfordnlp/sst2/train/train",
                       help='Directory to save k-NN sparsified graph batch files')
    parser.add_argument('--k', type=int, default=4,
                       help='Number of nearest neighbors for k-NN sparsification')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of batch files to process (for testing)')
    
    args = parser.parse_args()
    
    # Find all batch files
    batch_files = sorted(glob.glob(os.path.join(args.input_dir, "*.pkl")))
    if args.max_files:
        batch_files = batch_files[:args.max_files]
    
    print(f"Found {len(batch_files)} batch files to process")
    print(f"k-NN parameter: k={args.k}")
    print(f"Output directory: {args.output_dir}")
    
    # Process each batch file
    for input_path in tqdm(batch_files, desc="Processing batch files"):
        # Create corresponding output path
        filename = os.path.basename(input_path)
        output_path = os.path.join(args.output_dir, filename)
        
        # Skip if already processed
        if os.path.exists(output_path):
            print(f"Skipping {filename} (already exists)")
            continue
        
        try:
            process_batch_file(input_path, output_path, args.k)
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            continue
    
    print(f"\nCompleted! Processed k-NN graphs saved in: {args.output_dir}")
    print(f"Use these precomputed graphs with training_knn.py for faster training.")


if __name__ == "__main__":
    main()
