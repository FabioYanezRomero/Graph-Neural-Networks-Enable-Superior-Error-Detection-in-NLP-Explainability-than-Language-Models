"""datasets_window.py

Window-based GNN Training Pipeline for token sequence graphs.

Key Features:
------------
• Window-based connections: Each token connects to neighbors within a sliding window
• Configurable window size: Control the locality of connections
• Efficient edge construction: Linear complexity O(N*W) instead of O(N²)
• Maintains sequence structure: Preserves token order relationships
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GCNConv, GATConv, GATv2Conv, SAGEConv, TransformerConv, 
    global_mean_pool, MLP
)
import pickle
import os
import glob
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import json
from datetime import datetime

def create_window_graph(data: Data, window_size: int = 5) -> Data:
    """Create graph with sliding window connections.
    
    Args:
        data: Input graph data with node features
        window_size: Size of the sliding window (connects to window_size//2 neighbors on each side)
        
    Returns:
        Data object with window-based edge connections
    """
    x = data.x
    num_nodes = x.size(0)
    
    # Calculate half window size for symmetric connections
    half_window = window_size // 2
    
    source_nodes = []
    target_nodes = []
    
    # Create window-based connections
    for i in range(num_nodes):
        # Define window boundaries
        start_idx = max(0, i - half_window)
        end_idx = min(num_nodes, i + half_window + 1)
        
        # Connect to all nodes within the window
        for j in range(start_idx, end_idx):
            if i != j:  # Skip self-loops for now
                source_nodes.append(i)
                target_nodes.append(j)
    
    # Create edge index
    if source_nodes:
        edge_index = torch.stack([
            torch.tensor(source_nodes, dtype=torch.long),
            torch.tensor(target_nodes, dtype=torch.long)
        ], dim=0)
    else:
        # Fallback to self-loops if no connections
        edge_index = torch.stack([
            torch.arange(num_nodes, dtype=torch.long),
            torch.arange(num_nodes, dtype=torch.long)
        ], dim=0)
    
    # Add self-loops
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    
    return Data(x=data.x, edge_index=edge_index, y=data.y)


def analyze_window_connectivity(num_nodes: int, window_size: int):
    """Analyze the connectivity pattern for a given window size."""
    half_window = window_size // 2
    total_edges = 0
    
    for i in range(num_nodes):
        start_idx = max(0, i - half_window)
        end_idx = min(num_nodes, i + half_window + 1)
        connections = end_idx - start_idx - 1  # Exclude self
        total_edges += connections
    
    # Add self-loops
    total_edges += num_nodes
    
    # Calculate statistics
    fully_connected_edges = num_nodes * num_nodes
    reduction_percentage = (1 - total_edges / fully_connected_edges) * 100
    avg_degree = total_edges / num_nodes
    
    return {
        'total_edges': total_edges,
        'fully_connected_edges': fully_connected_edges,
        'reduction_percentage': reduction_percentage,
        'average_degree': avg_degree
    }


class WindowGraphDataset:
    """Dataset with window-based graph construction."""
    
    def __init__(self, root_dir: str, window_size: int = 5, max_files: int = None):
        self.root_dir = root_dir
        self.window_size = window_size
        
        # Find graph files
        pkl_files = sorted(glob.glob(os.path.join(root_dir, "*.pkl")))
        if max_files:
            pkl_files = pkl_files[:max_files]
        
        self.file_paths = pkl_files
        print(f"Found {len(self.file_paths)} batch files")
        
        # Quick initialization - assume 1000 graphs per file
        self.graphs_per_file = 1000
        self.total_graphs = len(self.file_paths) * self.graphs_per_file
        
        # Get metadata from first file
        with open(self.file_paths[0], 'rb') as f:
            first_batch = pickle.load(f)
        
        first_graph = first_batch[0]
        self.num_node_features = first_graph.num_node_features
        self.num_classes = 4  # AG News has 4 classes
        
        # Analyze window connectivity
        sample_nodes = first_graph.x.size(0)
        connectivity_stats = analyze_window_connectivity(sample_nodes, window_size)
        
        print(f"WindowGraphDataset: {self.total_graphs} graphs, {self.num_node_features} features, {self.num_classes} classes")
        print(f"Window size: {window_size}")
        print(f"Average edges per graph: {connectivity_stats['total_edges']:.0f}")
        print(f"Edge reduction vs fully connected: {connectivity_stats['reduction_percentage']:.1f}%")
        print(f"Average node degree: {connectivity_stats['average_degree']:.1f}")
    
    def __len__(self):
        return self.total_graphs
    
    def __getitem__(self, idx):
        file_idx = idx // self.graphs_per_file
        local_idx = idx % self.graphs_per_file
        
        # Load file
        with open(self.file_paths[file_idx], 'rb') as f:
            batch = pickle.load(f)
        
        # Get graph and apply window connections
        data = batch[local_idx]
        window_data = create_window_graph(data, window_size=self.window_size)
        
        return window_data


class ConfigurableWindowGNN(torch.nn.Module):
    """Configurable GNN for window-based graphs supporting multiple layer types."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, 
                 gnn_type='GCNConv', heads=1, dropout=0.5):
        super().__init__()
        
        self.gnn_type = gnn_type
        self.heads = heads
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(self._create_conv_layer(input_dim, hidden_dim, gnn_type, heads))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            # For GAT layers, the input dimension changes due to multi-head attention
            if gnn_type in ['GATConv', 'GATv2Conv'] and heads > 1:
                layer_input_dim = hidden_dim * heads
            else:
                layer_input_dim = hidden_dim
            self.convs.append(self._create_conv_layer(layer_input_dim, hidden_dim, gnn_type, heads))
        
        # Calculate final dimension for classifier
        if gnn_type in ['GATConv', 'GATv2Conv'] and heads > 1:
            final_dim = hidden_dim * heads
        else:
            final_dim = hidden_dim
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(final_dim, final_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(final_dim // 2, output_dim)
        )
    
    def _create_conv_layer(self, input_dim, output_dim, gnn_type, heads):
        """Create a GNN convolution layer based on the specified type."""
        if gnn_type == 'GCNConv':
            return GCNConv(input_dim, output_dim)
        elif gnn_type == 'GATConv':
            return GATConv(input_dim, output_dim, heads=heads, dropout=self.dropout)
        elif gnn_type == 'GATv2Conv':
            return GATv2Conv(input_dim, output_dim, heads=heads, dropout=self.dropout)
        elif gnn_type == 'SAGEConv':
            return SAGEConv(input_dim, output_dim)
        elif gnn_type == 'TransformerConv':
            return TransformerConv(input_dim, output_dim, heads=heads, dropout=self.dropout)
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            # For GAT layers, we need to handle the multi-head output
            if self.gnn_type in ['GATConv', 'GATv2Conv'] and self.heads > 1:
                # For the last layer, average the heads
                if i == len(self.convs) - 1:
                    x = x.mean(dim=1)  # Average over heads
            
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        
        return x


def test_window_sizes():
    """Test different window sizes to analyze connectivity patterns."""
    print("Window Size Connectivity Analysis")
    print("=" * 50)
    
    sample_nodes = 41  # Typical sequence length from AG News
    window_sizes = [3, 5, 7, 9, 11, 15, 21]
    
    print(f"Sample sequence length: {sample_nodes} tokens")
    print(f"Fully connected edges: {sample_nodes * sample_nodes}")
    print()
    
    for window_size in window_sizes:
        stats = analyze_window_connectivity(sample_nodes, window_size)
        print(f"Window {window_size:2d}: {stats['total_edges']:4.0f} edges, "
              f"{stats['reduction_percentage']:5.1f}% reduction, "
              f"avg degree {stats['average_degree']:4.1f}")


def train_window(window_size=7, max_epochs=50, patience=10, gnn_type='GCNConv', heads=1):
    """Training with window-based graphs and early stopping.
    
    Args:
        window_size: Size of the sliding window for connections
        max_epochs: Maximum number of epochs to train
        patience: Number of epochs to wait without improvement before stopping
        gnn_type: Type of GNN layer ('GCNConv', 'GATConv', 'GATv2Conv', 'SAGEConv', 'TransformerConv')
        heads: Number of attention heads for GAT and Transformer layers
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test different window sizes
    test_window_sizes()
    print()
    
    # Load datasets with window-based connections
    print(f"Training with window size: {window_size}")
    print("Loading datasets...")
    
    train_dataset = WindowGraphDataset(
        "outputs/embeddings/fully_connected/setfit/ag_news/train/train",
        window_size=window_size,
        max_files=10  # Use subset for faster training
    )
    
    val_dataset = WindowGraphDataset(
        "outputs/embeddings/fully_connected/setfit/ag_news/validation/validation",
        window_size=window_size
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # Initialize model
    model = ConfigurableWindowGNN(
        input_dim=train_dataset.num_node_features,
        hidden_dim=128,
        output_dim=train_dataset.num_classes,
        num_layers=3,
        gnn_type=gnn_type,
        heads=heads,
        dropout=0.5
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Using GNN type: {gnn_type} with {heads} heads")
    print(f"Starting window-based training with early stopping (patience={patience})...")
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None
    
    # Training loop with early stopping
    for epoch in range(max_epochs):
        # Training phase
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{max_epochs}'):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch)
            loss = F.cross_entropy(out, batch.y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            total_correct += (pred == batch.y).sum().item()
            total_samples += batch.y.size(0)
        
        train_loss = total_loss / len(train_loader)
        train_acc = total_correct / total_samples
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                loss = F.cross_entropy(out, batch.y)
                
                val_loss += loss.item()
                pred = out.argmax(dim=1)
                val_correct += (pred == batch.y).sum().item()
                val_total += batch.y.size(0)
        
        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"  → New best model! Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        else:
            patience_counter += 1
            print(f"  → No improvement ({patience_counter}/{patience})")
            
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"Best validation loss: {best_val_loss:.4f}")
                print(f"Best validation accuracy: {best_val_acc:.4f}")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded best model with validation accuracy: {best_val_acc:.4f}")
    
    return model, best_val_acc


if __name__ == "__main__":
    # Configure GNN type here - change this to test different architectures
    GNN_TYPE = 'GCNConv'  # Options: 'GCNConv', 'GATConv', 'GATv2Conv', 'SAGEConv', 'TransformerConv'
    HEADS = 1  # For attention-based models (GAT, Transformer)
    WINDOW_SIZE = 7  # Window size for connections
    
    # First analyze window connectivity patterns
    test_window_sizes()
    print("\n" + "="*50 + "\n")
    
    # Then run training
    train_window(window_size=WINDOW_SIZE, gnn_type=GNN_TYPE, heads=HEADS)
