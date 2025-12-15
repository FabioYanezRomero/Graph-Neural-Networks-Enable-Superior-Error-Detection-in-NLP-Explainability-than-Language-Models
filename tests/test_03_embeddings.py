"""
Tests for Embedding Generation (Paper Section 3.2)

Tests the src.embeddings module which extracts node embeddings from fine-tuned LLMs.
"""
import pytest
import torch


class TestEmbeddingsImports:
    """Test that embeddings module imports correctly."""
    
    def test_module_import(self):
        """Test basic module import."""
        from src import embeddings
        assert embeddings is not None
    
    def test_generate_import(self):
        """Test generate module imports."""
        from src.embeddings import generate
        assert generate is not None


class TestEmbeddingFormat:
    """Test embedding output formats."""
    
    def test_mock_embeddings_shape(self, mock_embeddings):
        """Test mock embeddings have expected shape."""
        assert mock_embeddings.dim() == 3  # (batch, seq, hidden)
        assert mock_embeddings.shape[2] == 768  # BERT hidden size
    
    def test_node_embeddings_are_tensors(self, mock_node_embeddings):
        """Test node embeddings are tensors."""
        for idx, emb in mock_node_embeddings.items():
            assert isinstance(emb, torch.Tensor)
            assert emb.shape[0] == 768


class TestEmbeddingDictionary:
    """Test embedding dictionary utilities."""
    
    def test_dicts_module_import(self):
        """Test dicts module imports."""
        from src.embeddings.dicts import EmbeddingDict
        assert EmbeddingDict is not None


@pytest.mark.slow
@pytest.mark.integration
class TestEmbeddingGeneration:
    """Integration tests for actual embedding generation."""
    
    def test_generate_embeddings_for_graph(self, sample_networkx_graph):
        """Test embedding generation for graph nodes."""
        # Would require model loading
        pass
