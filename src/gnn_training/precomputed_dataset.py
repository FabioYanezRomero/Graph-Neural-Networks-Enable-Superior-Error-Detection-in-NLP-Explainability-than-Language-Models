"""precomputed_dataset.py

Simple dataset loader for precomputed graphs (k-NN or window-connected).
This replaces on-the-fly computation with direct loading of precomputed graphs.

Usage:
    from precomputed_dataset import PrecomputedGraphDataset
    dataset = PrecomputedGraphDataset("/path/to/precomputed/graphs")
"""

import os
import glob
import pickle
from typing import Optional
import torch
from torch_geometric.data import Dataset


class PrecomputedGraphDataset(Dataset):
    """Dataset loader for precomputed graphs (k-NN or window-connected)."""
    
    def __init__(self, root_dir: str, max_files: Optional[int] = None):
        """
        Initialize dataset with precomputed graphs.
        
        Args:
            root_dir: Directory containing precomputed graph batch files (.pkl)
            max_files: Maximum number of batch files to load (for testing)
        """
        super().__init__(root=root_dir)
        self.root_dir = root_dir
        
        # Find graph files
        pkl_files = sorted(glob.glob(os.path.join(root_dir, "*.pkl")))
        if max_files:
            pkl_files = pkl_files[:max_files]
        
        self.file_paths = pkl_files
        print(f"Found {len(self.file_paths)} precomputed batch files")
        
        if not self.file_paths:
            raise RuntimeError(f"No .pkl files found in {root_dir}")
        
        # Quick initialization - assume 1000 graphs per file
        self.graphs_per_file = 1000
        self.total_graphs = len(self.file_paths) * self.graphs_per_file
        
        # Get metadata from first file
        with open(self.file_paths[0], 'rb') as f:
            first_batch = pickle.load(f)
        
        first_graph = first_batch[0]
        self.num_node_features = first_graph.num_node_features
        
        # Determine number of classes from the first batch
        unique_labels = set()
        for graph in first_batch[:100]:  # Check first 100 graphs
            unique_labels.add(graph.y.item())
        self.num_classes = len(unique_labels)
        
        print(f"PrecomputedGraphDataset: {self.total_graphs} graphs, {self.num_node_features} features, {self.num_classes} classes")
        print(f"Graphs are already sparsified/windowed - no on-the-fly computation needed!")
    
    def __len__(self):
        return self.total_graphs
    
    def __getitem__(self, idx):
        file_idx = idx // self.graphs_per_file
        local_idx = idx % self.graphs_per_file
        
        # Load file
        with open(self.file_paths[file_idx], 'rb') as f:
            batch = pickle.load(f)
        
        # Check if local_idx is within bounds
        if local_idx >= len(batch):
            local_idx = len(batch) - 1
        
        # Return precomputed graph directly - no processing needed!
        return batch[local_idx]


def load_precomputed_data(data_dir: str, batch_size: int = 32, shuffle: bool = True, 
                         num_workers: int = 4, max_files: Optional[int] = None):
    """
    Load precomputed graph data with DataLoader.
    
    Args:
        data_dir: Directory containing precomputed graph batch files
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for DataLoader
        max_files: Maximum number of batch files to load
    
    Returns:
        tuple: (dataset, dataloader)
    """
    from torch_geometric.loader import DataLoader
    
    dataset = PrecomputedGraphDataset(data_dir, max_files=max_files)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataset, dataloader


if __name__ == "__main__":
    # Test the dataset loader
    import argparse
    
    parser = argparse.ArgumentParser(description='Test precomputed dataset loader')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing precomputed graph batch files')
    parser.add_argument('--max_files', type=int, default=1,
                       help='Maximum number of batch files to test')
    
    args = parser.parse_args()
    
    print("Testing PrecomputedGraphDataset...")
    dataset, dataloader = load_precomputed_data(
        args.data_dir, 
        batch_size=4, 
        max_files=args.max_files
    )
    
    print(f"Dataset length: {len(dataset)}")
    print(f"DataLoader batches: {len(dataloader)}")
    
    # Test loading a few batches
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}: {batch.x.shape}, {batch.edge_index.shape}, {batch.y.shape}")
        if i >= 2:  # Test first 3 batches
            break
    
    print("Test completed successfully!")
