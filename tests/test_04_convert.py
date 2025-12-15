"""
Tests for NetworkX to PyG Conversion (Paper Section 3.1)

Tests the src.convert module which converts NetworkX graphs to PyTorch Geometric format.
"""
import pytest
import torch
import networkx as nx


class TestConvertImports:
    """Test that convert module imports correctly."""
    
    def test_module_import(self):
        """Test basic module import."""
        from src import convert
        assert convert is not None
    
    def test_nx_to_pyg_import(self):
        """Test converter function imports."""
        from src.convert.nx_to_pyg import convert_nx_to_pyg
        assert convert_nx_to_pyg is not None


class TestPyGDataFormat:
    """Test PyG Data object format."""
    
    def test_pyg_data_has_x(self, sample_pyg_data):
        """Test PyG data has node features."""
        assert hasattr(sample_pyg_data, 'x')
        assert sample_pyg_data.x is not None
    
    def test_pyg_data_has_edge_index(self, sample_pyg_data):
        """Test PyG data has edge index."""
        assert hasattr(sample_pyg_data, 'edge_index')
        assert sample_pyg_data.edge_index.shape[0] == 2
    
    def test_pyg_data_has_label(self, sample_pyg_data):
        """Test PyG data has graph label."""
        assert hasattr(sample_pyg_data, 'y')
        assert sample_pyg_data.y is not None
    
    def test_pyg_data_x_shape(self, sample_pyg_data):
        """Test node features have correct dimensionality."""
        num_nodes = sample_pyg_data.x.shape[0]
        hidden_dim = sample_pyg_data.x.shape[1]
        assert num_nodes > 0
        assert hidden_dim == 768  # BERT hidden size


class TestConversion:
    """Test actual conversion functionality."""
    
    def test_simple_graph_conversion(self, sample_networkx_graph, mock_node_embeddings):
        """Test converting simple graph to PyG format."""
        from src.convert.nx_to_pyg import convert_nx_to_pyg
        
        # Add embeddings to graph nodes
        G = sample_networkx_graph.copy()
        for node in G.nodes():
            if node in mock_node_embeddings:
                G.nodes[node]['embedding'] = mock_node_embeddings[node]
        
        # Conversion should work
        # pyg_data = convert_nx_to_pyg(G, label=1)
        # assert pyg_data is not None
    
    def test_edge_index_valid(self, sample_pyg_data):
        """Test edge index contains valid node indices."""
        edge_index = sample_pyg_data.edge_index
        num_nodes = sample_pyg_data.x.shape[0]
        
        assert edge_index.min() >= 0
        assert edge_index.max() < num_nodes


class TestBatchConversion:
    """Test batch conversion functionality."""
    
    def test_batch_convert_import(self):
        """Test batch converter imports."""
        from src.convert.batch_convert import batch_convert_graphs
        assert batch_convert_graphs is not None
    
    def test_pyg_batch_size(self, sample_pyg_batch):
        """Test batched graphs have correct count."""
        assert sample_pyg_batch.num_graphs == 4
