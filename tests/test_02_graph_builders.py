"""
Tests for Graph Building (Paper Section 3.1)

Tests the src.graph_builders module which converts text to graph representations:
- Constituency trees
- Dependency (syntactic) trees  
- Skip-gram graphs
- Window graphs
"""
import pytest
import networkx as nx


class TestGraphBuildersImports:
    """Test that graph_builders modules import correctly."""
    
    def test_module_import(self):
        """Test basic module import."""
        from src import graph_builders
        assert graph_builders is not None
    
    def test_constituency_import(self):
        """Test constituency builder imports."""
        from src.graph_builders import constituency
        assert constituency is not None
    
    def test_syntactic_import(self):
        """Test syntactic (dependency) builder imports."""
        from src.graph_builders import syntactic
        assert syntactic is not None
    
    def test_skipgrams_import(self):
        """Test skipgrams builder imports."""
        from src.graph_builders import skipgrams
        assert skipgrams is not None
    
    def test_window_import(self):
        """Test window builder imports."""
        from src.graph_builders import window
        assert window is not None


class TestRegistryPattern:
    """Test the graph builder registry pattern."""
    
    def test_registry_exists(self):
        """Test GENERATORS registry exists."""
        from src.graph_builders.registry import GENERATORS
        assert GENERATORS is not None
    
    def test_registry_has_names(self):
        """Test registry can list available builders."""
        from src.graph_builders.registry import GENERATORS
        names = GENERATORS.names()
        assert isinstance(names, list)


class TestGraphOutputFormat:
    """Test graph builders produce valid NetworkX graphs."""
    
    def test_sample_graph_is_networkx(self, sample_networkx_graph):
        """Test fixture produces NetworkX graph."""
        assert isinstance(sample_networkx_graph, nx.DiGraph)
    
    def test_graph_has_nodes(self, sample_networkx_graph):
        """Test graph has nodes with attributes."""
        G = sample_networkx_graph
        assert G.number_of_nodes() > 0
        for node in G.nodes():
            assert "word" in G.nodes[node]
    
    def test_graph_has_edges(self, sample_networkx_graph):
        """Test graph has edges."""
        G = sample_networkx_graph
        assert G.number_of_edges() > 0
    
    def test_window_graph_is_undirected(self, sample_window_graph):
        """Test window graphs are undirected."""
        assert isinstance(sample_window_graph, nx.Graph)
        assert not isinstance(sample_window_graph, nx.DiGraph)


class TestBaseGenerator:
    """Test base generator class."""
    
    def test_base_generator_import(self):
        """Test BaseTreeGenerator can be imported."""
        from src.graph_builders.base_generator import BaseTreeGenerator
        assert BaseTreeGenerator is not None


class TestWindowGenerator:
    """Test window graph generator."""
    
    def test_word_window_generator_exists(self):
        """Test WordWindowGenerator class exists."""
        from src.graph_builders.window import WordWindowGenerator
        assert WordWindowGenerator is not None
    
    def test_token_window_generator_exists(self):
        """Test TokenWindowGenerator class exists."""
        from src.graph_builders.window import TokenWindowGenerator
        assert TokenWindowGenerator is not None


@pytest.mark.slow
@pytest.mark.integration
class TestGraphGeneration:
    """Integration tests for actual graph generation."""
    
    def test_constituency_generation(self, sample_sentences):
        """Test constituency tree generation."""
        # Would require stanza model download
        pass
    
    def test_syntactic_generation(self, sample_sentences):
        """Test syntactic (dependency) tree generation."""
        # Would require stanza model download
        pass
