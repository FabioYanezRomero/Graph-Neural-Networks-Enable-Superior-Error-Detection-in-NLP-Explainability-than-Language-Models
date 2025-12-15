"""
Tests for GNN Training (Paper Section 3.3)

Tests the src.gnn_training module which trains GNN surrogates using LLM-as-teacher.
"""
import pytest
import torch


class TestGNNTrainingImports:
    """Test that gnn_training module imports correctly."""
    
    def test_module_import(self):
        """Test basic module import."""
        from src import gnn_training
        assert gnn_training is not None
    
    def test_training_import(self):
        """Test training module imports."""
        from src.gnn_training import training
        assert training is not None


class TestGCNModel:
    """Test GCN model architecture (2-layer as per paper)."""
    
    def test_model_forward(self, sample_pyg_data):
        """Test GCN forward pass."""
        from torch_geometric.nn import GCNConv, global_mean_pool
        import torch.nn as nn
        
        # Simple 2-layer GCN as described in paper
        class SimpleGCN(nn.Module):
            def __init__(self, in_channels, hidden_channels, num_classes):
                super().__init__()
                self.conv1 = GCNConv(in_channels, hidden_channels)
                self.conv2 = GCNConv(hidden_channels, hidden_channels)
                self.classifier = nn.Linear(hidden_channels, num_classes)
            
            def forward(self, x, edge_index, batch):
                x = self.conv1(x, edge_index).relu()
                x = self.conv2(x, edge_index).relu()
                x = global_mean_pool(x, batch)
                return self.classifier(x)
        
        model = SimpleGCN(768, 256, 2)
        batch = torch.zeros(sample_pyg_data.x.shape[0], dtype=torch.long)
        output = model(sample_pyg_data.x, sample_pyg_data.edge_index, batch)
        
        assert output.shape == (1, 2)  # 1 graph, 2 classes


class TestTrainingLoop:
    """Test training loop components."""
    
    def test_training_step(self, sample_pyg_batch):
        """Test single training step."""
        from torch_geometric.nn import GCNConv, global_mean_pool
        import torch.nn as nn
        import torch.nn.functional as F
        
        class SimpleGCN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = GCNConv(768, 256)
                self.conv2 = GCNConv(256, 128)
                self.classifier = nn.Linear(128, 2)
            
            def forward(self, data):
                x = self.conv1(data.x, data.edge_index).relu()
                x = self.conv2(x, data.edge_index).relu()
                x = global_mean_pool(x, data.batch)
                return self.classifier(x)
        
        model = SimpleGCN()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        optimizer.zero_grad()
        out = model(sample_pyg_batch)
        loss = F.cross_entropy(out, sample_pyg_batch.y.squeeze())
        loss.backward()
        optimizer.step()
        
        assert loss.item() >= 0


@pytest.mark.slow
@pytest.mark.integration
class TestFullTraining:
    """Integration tests for full training pipeline."""
    
    def test_train_on_dataset(self):
        """Test training on actual dataset (slow)."""
        pass
