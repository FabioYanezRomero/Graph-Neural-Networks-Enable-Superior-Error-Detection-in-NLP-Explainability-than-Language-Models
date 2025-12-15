"""
Pytest configuration and shared fixtures for pipeline testing.

These fixtures provide test data and utilities for validating each step
of the GNN Explainability pipeline as described in paper Sections 3.1-3.6.
"""
import pytest
import torch
import networkx as nx
from typing import List, Dict, Any


# =============================================================================
# Sample Text Data
# =============================================================================

@pytest.fixture
def sample_sentences() -> List[str]:
    """Minimal test sentences for graph building."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models require training data.",
        "Natural language processing enables text understanding.",
    ]


@pytest.fixture
def sample_labeled_data() -> List[Dict[str, Any]]:
    """Labeled data for training/testing."""
    return [
        {"text": "This movie is excellent!", "label": 1},  # positive
        {"text": "Terrible waste of time.", "label": 0},   # negative
        {"text": "Great acting and storyline.", "label": 1},
        {"text": "Boring and predictable plot.", "label": 0},
    ]


# =============================================================================
# Graph Fixtures
# =============================================================================

@pytest.fixture
def sample_networkx_graph() -> nx.DiGraph:
    """Sample dependency-style graph for testing."""
    G = nx.DiGraph()
    # Nodes with word attributes
    G.add_node(0, word="The", pos="DET", idx=0)
    G.add_node(1, word="cat", pos="NOUN", idx=1)
    G.add_node(2, word="sat", pos="VERB", idx=2)
    G.add_node(3, word="on", pos="ADP", idx=3)
    G.add_node(4, word="mat", pos="NOUN", idx=4)
    
    # Edges (head -> dependent)
    G.add_edge(2, 1)  # sat -> cat (nsubj)
    G.add_edge(2, 3)  # sat -> on (prep)
    G.add_edge(1, 0)  # cat -> The (det)
    G.add_edge(3, 4)  # on -> mat (pobj)
    
    return G


@pytest.fixture
def sample_window_graph() -> nx.Graph:
    """Sample window-based co-occurrence graph."""
    G = nx.Graph()
    words = ["machine", "learning", "models", "require", "data"]
    for i, word in enumerate(words):
        G.add_node(i, word=word, idx=i)
    
    # Window size 2 connections
    for i in range(len(words)):
        for j in range(i + 1, min(i + 3, len(words))):
            G.add_edge(i, j)
    
    return G


# =============================================================================
# PyTorch Geometric Fixtures
# =============================================================================

@pytest.fixture
def sample_pyg_data():
    """Sample PyG Data object for GNN testing."""
    from torch_geometric.data import Data
    
    # 5 nodes, 768-dim embeddings (like BERT)
    x = torch.randn(5, 768)
    
    # Edge index (COO format)
    edge_index = torch.tensor([
        [0, 1, 2, 2, 3],  # source
        [1, 2, 1, 3, 4],  # target
    ], dtype=torch.long)
    
    # Graph-level label
    y = torch.tensor([1], dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, y=y)


@pytest.fixture
def sample_pyg_batch():
    """Batch of PyG graphs for training tests."""
    from torch_geometric.data import Data, Batch
    
    graphs = []
    for i in range(4):
        num_nodes = 5 + i
        x = torch.randn(num_nodes, 768)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        y = torch.tensor([i % 2], dtype=torch.long)
        graphs.append(Data(x=x, edge_index=edge_index, y=y))
    
    return Batch.from_data_list(graphs)


# =============================================================================
# Mock Model Fixtures
# =============================================================================

@pytest.fixture
def mock_embeddings() -> torch.Tensor:
    """Mock BERT-like embeddings (batch=1, seq_len=10, hidden=768)."""
    return torch.randn(1, 10, 768)


@pytest.fixture
def mock_node_embeddings() -> Dict[int, torch.Tensor]:
    """Node index to embedding mapping."""
    return {i: torch.randn(768) for i in range(5)}


# =============================================================================
# Explainability Fixtures
# =============================================================================

@pytest.fixture
def sample_importance_scores() -> List[float]:
    """Feature importance scores for explainability tests."""
    return [0.35, 0.25, 0.15, 0.12, 0.08, 0.05]


@pytest.fixture
def sample_explanation_result() -> Dict[str, Any]:
    """Structure matching explainer output format."""
    return {
        "graph_idx": 0,
        "prediction": 1,
        "confidence": 0.92,
        "node_importance": [0.4, 0.3, 0.2, 0.1],
        "selected_nodes": [0, 1],
        "fidelity_plus": 0.85,
        "fidelity_minus": 0.78,
    }


# =============================================================================
# Analytics Fixtures
# =============================================================================

@pytest.fixture
def sample_auc_metrics() -> Dict[str, float]:
    """Sample AUC metrics for analytics tests."""
    return {
        "deletion_auc": 0.82,
        "insertion_auc": 0.75,
        "origin_confidence": 0.95,
    }


@pytest.fixture
def sample_progression_data() -> Dict[str, List[float]]:
    """Progression data for dimension 4 tests."""
    return {
        "maskout_progression_confidence": [0.95, 0.85, 0.70, 0.55, 0.40],
        "maskout_progression_drop": [0.10, 0.15, 0.15, 0.15],
        "sufficiency_progression_confidence": [0.20, 0.45, 0.65, 0.80, 0.92],
        "sufficiency_progression_drop": [0.25, 0.20, 0.15, 0.12],
    }


# =============================================================================
# Utility Functions
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")
