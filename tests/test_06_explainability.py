"""
Tests for Explainability Modules (Paper Section 3.4)

Tests the src.explain module including:
- SubgraphX (tree-structured graphs)
- GraphSVX (non-tree graphs)
- TokenSHAP (LLM baseline)
"""
import pytest


class TestExplainImports:
    """Test that explain modules import correctly."""
    
    def test_module_import(self):
        """Test basic module import."""
        from src import explain
        assert explain is not None
    
    def test_gnn_explainers_import(self):
        """Test GNN explainer modules import."""
        from src.explain import gnn
        assert gnn is not None
    
    def test_llm_explainers_import(self):
        """Test LLM explainer modules import."""
        from src.explain import llm
        assert llm is not None


class TestSubgraphXImports:
    """Test SubgraphX (for tree-structured graphs)."""
    
    def test_subgraphx_module_exists(self):
        """Test subgraphx module exists."""
        from src.explain.gnn import subgraphx
        assert subgraphx is not None


class TestGraphSVXImports:
    """Test GraphSVX (for non-tree graphs)."""
    
    def test_graphsvx_module_exists(self):
        """Test graphsvx module exists."""
        from src.explain.gnn import graphsvx
        assert graphsvx is not None


class TestTokenSHAPImports:
    """Test TokenSHAP (LLM baseline)."""
    
    def test_tokenshap_module_exists(self):
        """Test TokenSHAP module exists."""
        from src.explain.llm import tokenSHAP
        assert tokenSHAP is not None


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


class TestFairExplainability:
    """Test fair comparison methodology (Section 3.4.2)."""
    
    def test_config_has_fair_params(self):
        """Test explainer config includes fairness parameters."""
        from src.explain.gnn.config import ExplainerConfig
        config = ExplainerConfig()
        # Fair comparison: 2000 forward passes budget
        assert hasattr(config, 'max_evaluations') or hasattr(config, 'budget')


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
