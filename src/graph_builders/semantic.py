"""
Semantic Graph Generator Module

This module provides functionality to create semantic graphs from sentences.
It leverages the supar Parser to generate semantic dependency structures and converts them
into directed graphs where nodes represent words and edges represent semantic relations.
"""

from typing import List, Dict, Union, Any, Optional
import networkx as nx
import torch
from supar import Parser
from .base_generator import BaseGraphGenerator

# Available Supar semantic dependency parsing models
SUPAR_SDP_MODELS = [
    'sdp-biaffine-en',  # Basic biaffine semantic dependency parser
    'sdp-vi-en',  # Variational inference semantic dependency parser
    'sdp-vi-roberta-en'  # RoBERTa-enhanced variational inference parser
]

class SemanticGraphGenerator(BaseGraphGenerator):
    """
    Creates semantic graphs from sentences.

    This class processes sentences using a semantic parser and converts
    the parsed structures into directed graphs. Each node in the graph represents
    a word from the sentence, and edges represent semantic dependencies.
    """

    def __init__(self, model: str, device: str = 'cuda:0'):
        """
        Initialize the semantic parser with a given model.

        Args:
            model (str): Name of the semantic parser model to load.
            device (str, optional): Device to run the parser on. Defaults to 'cuda:0'.

        Raises:
            RuntimeError: If the specified device is not available.
        """
        super().__init__(model, device)
        
        if model not in SUPAR_SDP_MODELS:
            raise ValueError(f"Unknown model: {model}. Available models: {SUPAR_SDP_MODELS}")
        self.property = 'semantic'
        
        # Verify device availability for better error handling
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please use 'cpu' instead or check CUDA installation.")
        
        self.device = device
        if device.startswith('cuda'):
            torch.cuda.set_device(device)
        
        try:
            self.sem = Parser.load(model)
        except Exception as e:
            raise RuntimeError(f"Failed to load semantic parser model '{model}': {e}")

    def _parse(self, sentences: List[str]):
        """
        Parse sentences using the semantic parser.

        Args:
            sentences (List[str]): List of sentences to parse.

        Returns:
            The parsed semantic structures.
        """
        return self.sem.predict(sentences, verbose=False, lang='en')
    
    def _build_graph(self, semantic_trees, ids: Optional[List[str]] = None) -> List[nx.DiGraph]:
        """
        Build graphs from the parsed semantic structures.

        Args:
            semantic_trees: The parsed semantic structures.
            ids (Optional[List[str]], optional): Optional list of IDs to assign to graphs. Defaults to None.

        Returns:
            List[nx.DiGraph]: List of directed graphs representing the semantic structure.
        """
        graph_list = []
        
        for i, semantic_tree in enumerate(semantic_trees):
            graph = nx.DiGraph()
            
            # Add nodes (words) to the graph
            for j in range(len(semantic_tree.values[1])):
                graph.add_node(j+1, label=semantic_tree.values[1][j])

            # Handle potential inconsistencies in the semantic tree structure
            if len(semantic_tree.values[8]) == len(semantic_tree.values[1])-1:
                semantic_tree.values[8].append('punct')

            # Add edges (semantic relations) to the graph
            for j in range(len(semantic_tree.values[8])):
                parents = semantic_tree.values[8][j]
                
                # Skip punctuation and special tokens
                if '_' in parents or 'punct' in parents:
                    continue
                
                # Handle multiple parents (a word can have multiple semantic relations)
                if '|' in parents:
                    parents = parents.split('|')
                    for parent in parents:
                        if parent[0] == '0':
                            continue
                        par, relation = int(parent.split(':')[0]), parent.split(':')[1]
                        child = int(semantic_tree.values[0][j])
                        graph.add_edge(par, child, label=relation)
                else:
                    # Handle single parent
                    if parents[0] == '0':
                        continue    
                    parent, relation = int(parents.split(':')[0]), parents.split(':')[1]
                    child = int(semantic_tree.values[0][j])
                    graph.add_edge(parent, child, label=relation)

            # Add graph metadata
            graph.graph['model'] = self.model
            graph.graph['property'] = self.property
            if ids and i < len(ids):
                graph.graph['id'] = ids[i]
                
            graph_list.append(graph)
            
        return graph_list

    def get_graph(self, sentences: List[str], ids: Optional[List[str]] = None) -> List[nx.DiGraph]:
        """
        Get the semantic graph for each sentence in the list.

        Args:
            sentences (List[str]): List of sentences to process.
            ids (Optional[List[str]], optional): List of IDs to assign to each graph. Defaults to None.

        Returns:
            List[nx.DiGraph]: List of semantic graphs.
            
        Raises:
            ValueError: If ids are provided but don't match the number of sentences.
        """
        if ids is not None and len(ids) != len(sentences):
            raise ValueError(f"Number of ids ({len(ids)}) must match number of sentences ({len(sentences)})")
        
        try:
            semantic_trees = self._parse(sentences)
            graphs = self._build_graph(semantic_trees, ids)
            return graphs
        except Exception as e:
            raise RuntimeError(f"Error generating semantic graphs: {e}")
