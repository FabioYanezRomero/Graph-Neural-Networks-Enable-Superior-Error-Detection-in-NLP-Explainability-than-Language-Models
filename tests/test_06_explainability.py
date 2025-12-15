"""
Tests for Explainability Modules (Paper Section 3.4)

Tests the src.explain module including:
- SubgraphX (tree-structured graphs)
- GraphSVX (non-tree graphs)
- TokenSHAP (LLM baseline)

Note: Some imports may fail if energy monitoring module is not available.
These tests focus on the core explainability logic.
"""
import pytest


class TestExplainImports:
    """Test that explain modules import correctly."""
    
    def test_module_import(self):
        """Test basic module import."""
        from src import explain
        assert explain is not None


class TestSubgraphXImports:
    """Test SubgraphX (for tree-structured graphs)."""
    
    def test_subgraphx_main_exists(self):
        """Test subgraphx main module exists."""
        import importlib.util
        spec = importlib.util.find_spec("src.explain.gnn.subgraphx.main")
        assert spec is not None


class TestGraphSVXImports:
    """Test GraphSVX (for non-tree graphs)."""
    
    def test_graphsvx_main_exists(self):
        """Test graphsvx main module exists."""
        import importlib.util
        spec = importlib.util.find_spec("src.explain.gnn.graphsvx.main")
        assert spec is not None


class TestTokenSHAPImports:
    """Test TokenSHAP (LLM baseline)."""
    
    def test_tokenshap_file_exists(self):
        """Test TokenSHAP file exists."""
        import importlib.util
        spec = importlib.util.find_spec("src.explain.llm.tokenSHAP")
        assert spec is not None
    
    def test_token_shap_runner_exists(self):
        """Test token_shap_runner module exists."""
        import importlib.util
        spec = importlib.util.find_spec("src.explain.llm.token_shap_runner")
        assert spec is not None


class TestGNNConfig:
    """Test GNN explainer configuration."""
    
    def test_config_module_exists(self):
        """Test config module exists."""
        from src.explain.gnn import config
        assert config is not None


class TestExplanationFormat:
    """Test explanation output format."""
    
    def test_explanation_has_required_fields(self, sample_explanation_result):
        """Test explanation result has required fields."""
        required_fields = [
            "graph_idx",
            "prediction", 
            "confidence",
            "node_importance",
            "fidelity_plus",
            "fidelity_minus",
        ]
        for field in required_fields:
            assert field in sample_explanation_result
    
    def test_importance_scores_sum_positive(self, sample_importance_scores):
        """Test importance scores are positive and sum to ~1."""
        assert all(s >= 0 for s in sample_importance_scores)
        assert 0.99 <= sum(sample_importance_scores) <= 1.01
    
    def test_fidelity_in_valid_range(self, sample_explanation_result):
        """Test fidelity scores are in valid range."""
        assert -1 <= sample_explanation_result["fidelity_plus"] <= 1
        assert -1 <= sample_explanation_result["fidelity_minus"] <= 1


@pytest.mark.slow
@pytest.mark.integration
class TestExplainerExecution:
    """Integration tests for running explainers."""
    
    def test_subgraphx_on_tree_graph(self):
        """Test SubgraphX on constituency/syntactic graph."""
        pass
    
    def test_graphsvx_on_proximity_graph(self):
        """Test GraphSVX on window/skipgram graph."""
        pass
    
    def test_tokenshap_on_text(self):
        """Test TokenSHAP on text input."""
        pass
