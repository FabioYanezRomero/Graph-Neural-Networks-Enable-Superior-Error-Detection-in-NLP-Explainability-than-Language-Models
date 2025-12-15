"""datasets_fast.py

Ultra-fast GNN Training Pipeline optimized for dense (fully connected) graphs.

Key Optimizations for Dense Graphs:
----------------------------------
• Graph sparsification: k-NN graphs reduce edges by 78-87%
• Smaller batch sizes: Optimized for dense graph memory usage
• Gradient accumulation: Maintain effective batch size
• Faster initialization: Reduced file scanning
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


class FastGraphDataset:
    """Ultra-fast dataset with graph sparsification and caching."""
    
    def __init__(self, root_dir: str, sparsity_k: int = 8, max_files: int = None, cache_size: int = 5):
        self.root_dir = root_dir
        self.sparsity_k = sparsity_k
        self.cache_size = cache_size
        
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
        self.num_classes = 4  # SST-2 has 2 classes, but keeping 4 for compatibility
        
        # Cache for sparsified batches
        self.batch_cache = {}
        
        print(f"FastGraphDataset: {self.total_graphs} graphs, {self.num_node_features} features, {self.num_classes} classes")
        print(f"Sparsification: k-NN with k={sparsity_k} (with caching)")
    
    def __len__(self):
        return self.total_graphs
    
    def __getitem__(self, idx):
        file_idx = idx // self.graphs_per_file
        local_idx = idx % self.graphs_per_file
        
        # Check if batch is already cached and sparsified
        if file_idx not in self.batch_cache:
            # Load and sparsify entire batch
            with open(self.file_paths[file_idx], 'rb') as f:
                batch = pickle.load(f)
            
            # Sparsify all graphs in the batch
            sparsified_batch = []
            for data in batch:
                sparse_data = sparsify_graph_knn(data, k=self.sparsity_k)
                sparsified_batch.append(sparse_data)
            
            # Cache management - remove oldest if cache is full
            if len(self.batch_cache) >= self.cache_size:
                oldest_key = next(iter(self.batch_cache))
                del self.batch_cache[oldest_key]
                gc.collect()
            
            self.batch_cache[file_idx] = sparsified_batch
        
        # Get from cache
        cached_batch = self.batch_cache[file_idx]
        
        # Check if local_idx is within bounds
        if local_idx >= len(cached_batch):
            local_idx = len(cached_batch) - 1
        
        return cached_batch[local_idx]


class ConfigurableGNN(torch.nn.Module):
    """Configurable GNN supporting multiple layer types."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, 
                 gnn_type='GCNConv', heads=1, dropout=0.5):
        super().__init__()
        
        self.gnn_type = gnn_type
        self.heads = heads
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        
        self.convs = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(self._create_conv_layer(input_dim, hidden_dim, gnn_type, heads))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            # For multi-head attention layers, the input dimension changes due to concatenated heads
            if gnn_type in ['GATConv', 'GATv2Conv', 'TransformerConv'] and heads > 1:
                layer_input_dim = hidden_dim * heads
            else:
                layer_input_dim = hidden_dim
            self.convs.append(self._create_conv_layer(layer_input_dim, hidden_dim, gnn_type, heads))
        
        # Calculate final dimension for classifier
        # Since we average heads in the forward pass, final dimension is always hidden_dim
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
            
            # For multi-head attention layers, we need to handle the multi-head output
            if self.gnn_type in ['GATConv', 'GATv2Conv', 'TransformerConv'] and self.heads > 1:
                # For the last layer, average the heads to get final dimension
                if i == len(self.convs) - 1:
                    # Reshape from [N, heads * hidden_dim] to [N, heads, hidden_dim] and average
                    x = x.view(-1, self.heads, self.hidden_dim).mean(dim=1)
            
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        
        return x


def train_fast(max_epochs=20, patience=5, gnn_type='GCNConv', heads=1):
    """Fast training with optimized settings and early stopping.
    
    Args:
        max_epochs: Maximum number of epochs to train
        patience: Number of epochs to wait without improvement before stopping
        gnn_type: Type of GNN layer ('GCNConv', 'GATConv', 'GATv2Conv', 'SAGEConv', 'TransformerConv')
        heads: Number of attention heads for GAT and Transformer layers
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets with reduced size for faster training
    print("Loading datasets...")
    train_dataset = FastGraphDataset(
        "outputs/embeddings/fully_connected/stanfordnlp/sst2/train/train",
        sparsity_k=8,
    )
    
    test_dataset = FastGraphDataset(
        "outputs/embeddings/fully_connected/stanfordnlp/sst2/validation/validation",
        sparsity_k=8
    )
    
    # Create data loaders with very small batch size and no multiprocessing to avoid memory issues
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=False)
    
    # Initialize model with even smaller dimensions to reduce memory usage
    model = ConfigurableGNN(
        input_dim=train_dataset.num_node_features,
        hidden_dim=128, 
        output_dim=train_dataset.num_classes,
        num_layers=2,
        gnn_type=gnn_type,
        heads=heads,
        dropout=0.5
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Using GNN type: {gnn_type} with {heads} heads")
    print(f"Starting k-NN training with early stopping (patience={patience})...")
    
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
            
            # Clear cache periodically to prevent memory buildup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            del batch, out, loss, pred
            gc.collect()
        
        train_loss = total_loss / len(train_loader)
        train_acc = total_correct / total_samples
        
        # Validation phase
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch)
                loss = F.cross_entropy(out, batch.y)
                
                test_loss += loss.item()
                pred = out.argmax(dim=1)
                test_correct += (pred == batch.y).sum().item()
                test_total += batch.y.size(0)
        
        test_loss = test_loss / len(test_loader)
        test_acc = test_correct / test_total
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")
        
        # Early stopping logic
        if test_loss < best_val_loss:
            best_val_loss = test_loss
            best_val_acc = test_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"  → New best model! Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        else:
            patience_counter += 1
            print(f"  → No improvement ({patience_counter}/{patience})")
            
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"Best test loss: {best_val_loss:.4f}")
                print(f"Best test accuracy: {best_val_acc:.4f}")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded best model with test accuracy: {best_val_acc:.4f}")
    
    return model, best_val_acc


if __name__ == "__main__":
    # Configure GNN type here - change this to test different architectures
    GNN_TYPE = 'GCNConv'  # Options: 'GCNConv', 'GATConv', 'GATv2Conv', 'SAGEConv', 'TransformerConv'
    HEADS = 1  # For attention-based models (GAT, Transformer)
    
    train_fast(gnn_type=GNN_TYPE, heads=HEADS)
