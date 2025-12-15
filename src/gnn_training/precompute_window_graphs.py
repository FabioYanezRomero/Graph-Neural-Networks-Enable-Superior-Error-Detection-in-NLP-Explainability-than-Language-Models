"""precompute_window_graphs.py

Script to precompute window-connected graphs from fully connected graphs and save them in batches.
This eliminates the need for on-the-fly window computation during training.

Usage:
    python precompute_window_graphs.py --input_dir /path/to/fully_connected/graphs --output_dir /path/to/window/graphs --window_size 5
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


def create_window_graph(data: Data, window_size: int = 5) -> Data:
    """Create window-connected graph where each token connects to neighbors within window_size."""
    x = data.x
    num_nodes = x.size(0)
    
    # Create sliding window connections
    source_nodes = []
    target_nodes = []
    
    for i in range(num_nodes):
        # Connect to nodes within window (both directions)
        start_idx = max(0, i - window_size)
        end_idx = min(num_nodes, i + window_size + 1)
        
        for j in range(start_idx, end_idx):
            if i != j:  # Avoid self-loops (will be added later)
                source_nodes.append(i)
                target_nodes.append(j)
    
    # Convert to tensors
    if source_nodes:
        edge_index = torch.stack([
            torch.tensor(source_nodes, dtype=torch.long),
            torch.tensor(target_nodes, dtype=torch.long)
        ], dim=0)
    else:
        # Handle edge case of single node
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    # Add self-loops
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    
    return Data(x=data.x, edge_index=edge_index, y=data.y)


def process_batch_file(input_path: str, output_path: str, window_size: int):
    """Process a single batch file and save window-connected graphs."""
    print(f"Processing {input_path}...")
    
    # Load original batch
    with open(input_path, 'rb') as f:
        batch = pickle.load(f)
    
    # Create window graphs for all graphs in the batch
    window_batch = []
    for i, data in enumerate(tqdm(batch, desc="Creating window graphs", leave=False)):
        window_data = create_window_graph(data, window_size=window_size)
        window_batch.append(window_data)
        
        # Clear memory periodically
        if i % 100 == 0:
            gc.collect()
    
    # Save window batch
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(window_batch, f)
    
    print(f"Saved {len(window_batch)} window graphs to {output_path}")
    
    # Clear memory
    del batch, window_batch
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description='Precompute window-connected graphs')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing fully connected graph batch files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save window-connected graph batch files')
    parser.add_argument('--window_size', type=int, default=5,
                       help='Window size for sliding window connections')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of batch files to process (for testing)')
    
    args = parser.parse_args()
    
    # Find all batch files
    batch_files = sorted(glob.glob(os.path.join(args.input_dir, "*.pkl")))
    if args.max_files:
        batch_files = batch_files[:args.max_files]
    
    print(f"Found {len(batch_files)} batch files to process")
    print(f"Window size parameter: window_size={args.window_size}")
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
            process_batch_file(input_path, output_path, args.window_size)
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            continue
    
    print(f"\nCompleted! Processed window graphs saved in: {args.output_dir}")
    print(f"Use these precomputed graphs with training_window.py for faster training.")


if __name__ == "__main__":
    main()
