"""
Test script to verify data loading and inspect data structure.
"""
import os
import sys
import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader as PyGDataLoader
from training import load_graph_data_optimized, CachedGraphDataset

def print_graph_info(graph, index=0):
    """Print information about a graph."""
    print(f"\nGraph {index}:")
    print(f"  Type: {type(graph)}")
    print("  Attributes:", [attr for attr in dir(graph) if not attr.startswith('_')])
    
    if hasattr(graph, 'x'):
        x = graph.x
        print(f"  x shape: {x.shape if hasattr(x, 'shape') else 'N/A'}")
        print(f"  x type: {x.dtype if hasattr(x, 'dtype') else 'N/A'}")
        print(f"  x min/max: {x.min().item() if hasattr(x, 'min') else 'N/A'}/{x.max().item() if hasattr(x, 'max') else 'N/A'}")
    
    if hasattr(graph, 'edge_index'):
        edge_index = graph.edge_index
        print(f"  edge_index shape: {edge_index.shape if hasattr(edge_index, 'shape') else 'N/A'}")
        print(f"  Num edges: {edge_index.size(1) if hasattr(edge_index, 'size') and len(edge_index.size()) > 1 else 'N/A'}")
    
    if hasattr(graph, 'y'):
        y = graph.y
        if hasattr(y, 'shape'):
            print(f"  y shape: {y.shape}")
            print(f"  y unique: {torch.unique(y) if y.numel() > 0 else 'empty'}")
        else:
            print(f"  y: {y}")

def test_data_loading(data_dir, batch_size=4, num_samples=5):
    """Test data loading and inspect samples."""
    print(f"\n{'='*50}")
    print(f"Testing data loading from: {data_dir}")
    print(f"Batch size: {batch_size}")
    print(f"{'='*50}\n")
    
    # Initialize dataset directly
    print("Initializing dataset...")
    try:
        dataset = CachedGraphDataset(root_dir=data_dir, cache_size=2, use_metadata_cache=True)
        print(f"Dataset initialized with {len(dataset)} graphs")
        print(f"Num node features: {dataset.num_node_features}")
        print(f"Num classes: {dataset.num_classes}")
        
        # Inspect first few samples
        print("\nInspecting first few samples...")
        for i in range(min(num_samples, len(dataset))):
            try:
                graph = dataset[i]
                print_graph_info(graph, i)
            except Exception as e:
                print(f"Error loading sample {i}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Test DataLoader
        print("\nTesting DataLoader...")
        dataloader = PyGDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Start with 0 workers for debugging
            pin_memory=False,
            drop_last=False
        )
        
        # Try to load first batch
        try:
            print("\nLoading first batch...")
            batch = next(iter(dataloader))
            print("\nBatch loaded successfully!")
            print(f"Batch type: {type(batch)}")
            print("Batch attributes:", [attr for attr in dir(batch) if not attr.startswith('_')])
            
            if hasattr(batch, 'x'):
                print(f"Batch x shape: {batch.x.shape if hasattr(batch.x, 'shape') else 'N/A'}")
            if hasattr(batch, 'y'):
                print(f"Batch y shape: {batch.y.shape if hasattr(batch.y, 'shape') else 'N/A'}")
                print(f"Batch y unique: {torch.unique(batch.y) if hasattr(batch.y, 'numel') and batch.y.numel() > 0 else 'N/A'}")
            
            # Print batch info for each graph in the batch
            print("\nBatch graph info:")
            for i in range(min(3, batch.num_graphs if hasattr(batch, 'num_graphs') else 1)):
                print(f"  Graph {i}:")
                if hasattr(batch, 'x'):
                    print(f"    x: {batch.x[batch.batch == i].shape if hasattr(batch, 'batch') else 'N/A'}")
                if hasattr(batch, 'y'):
                    print(f"    y: {batch.y[i].item() if hasattr(batch.y, 'item') else batch.y}")
                    
        except Exception as e:
            print(f"\nError loading batch: {str(e)}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"\nError initializing dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    # Use the same data directories as in the training script
    train_dir = "outputs/embeddings/knn8/stanfordnlp/sst2/train/train"
    val_dir = "outputs/embeddings/knn8/stanfordnlp/sst2/validation/validation"
    
    print("Testing training data...")
    train_ok = test_data_loading(train_dir)
    
    print("\n" + "="*50)
    print("Testing validation data...")
    val_ok = test_data_loading(val_dir)
    
    print("\n" + "="*50)
    print("Test Summary:")
    print(f"Training data: {'OK' if train_ok else 'FAILED'}")
    print(f"Validation data: {'OK' if val_ok else 'FAILED'}")
    print("="*50)
