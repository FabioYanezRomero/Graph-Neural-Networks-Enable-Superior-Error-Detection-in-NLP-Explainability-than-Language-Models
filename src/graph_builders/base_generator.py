"""
Base Tree Generator Module
This module provides a base class for tree generators to implement.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Any
import networkx as nx
import pickle as pkl
import matplotlib.pyplot as plt

class BaseTreeGenerator(ABC):
    """
    Abstract base class for all tree generators.
    
    This class defines the common interface that all tree generators must implement,
    ensuring consistency across different tree generation approaches.
    
    Attributes:
        model (str): Name of the model used for tree generation.
        property (str): Type of tree property (e.g., 'constituency').
        device (str): Device to run computation on ('cpu', 'cuda:0', etc.)
    """
    def __init__(self, property: str = None, device: str = 'cuda:0'): # type: ignore
        """
        Initialize the tree generator with a model and device.
        
        Args:
            property (str): Type of tree property to generate.
            device (str, optional): Device to run the model on. Defaults to 'cuda:0'.
        """
        self.property = property  # To be set by child classes
        self.device = device
    
    @abstractmethod
    def _parse(self, sentences: List[str]) -> Any:
        """
        Parse the input sentences using the appropriate parser.
        
        Args:
            sentences (List[str]): List of sentences to parse.
        
        Returns:
            Any: The parsed representation of the sentences.
        """
        pass
    
    @abstractmethod
    def _build_graph(self, *args, **kwargs) -> List[nx.DiGraph]:
        """
        Build graphs from the parsed data.
        
        Args:
            args (Any): The parsed representation of the sentences.
        
        Returns:
            List[nx.DiGraph]: List of directed graphs.
        """
        pass
    
    @abstractmethod
    def get_graph(self, sentences: List[str], ids: Optional[List[str]] = None) -> nx.DiGraph:
        """
        Generate graphs for a list of sentences.
        
        Args:
            sentences (List[str]): List of sentences to process.
            ids (Optional[List[str]]): Optional list of IDs to assign to the graphs.
        
        Returns:
            List[nx.DiGraph]: List of generated graphs.
        """
        pass
    
    def draw_graph(self, graph: nx.DiGraph, figsize: tuple = (12, 8)) -> None:
        """
        Draw the graph for visualization.
        
        Args:
            graph (nx.DiGraph): The graph to draw.
            figsize (tuple, optional): Size of the figure. Defaults to (12, 8).
        """
        plt.figure(figsize=figsize)
        labels = nx.get_node_attributes(graph, 'label')
        pos = nx.kamada_kawai_layout(graph)
        
        nx.draw(graph, pos, with_labels=True, labels=labels,
                node_color='lightblue', node_size=1500,
                font_weight='bold', font_size=8,
                edge_color='gray')
                
        edge_labels = nx.get_edge_attributes(graph, 'label')
        if edge_labels:
            nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
            
        plt.title(f"{self.property.capitalize()} Tree")
        plt.show()
    
    def save_graph(self, graph: Union[nx.DiGraph, List[nx.DiGraph]], folder: str, filename: str) -> None:
        """
        Save the graph(s) to a file in pickle format.
        
        Args:
            graph (Union[nx.DiGraph, List[nx.DiGraph]]): The graph or list of graphs to save.
            folder (str): Destination folder path.
            filename (str): Name of the file without extension.
            
        Raises:
            IOError: If there's an error saving the file.
        """
        try:
            with open(f'{folder}/{filename}.pkl', 'wb') as f:
                pkl.dump(graph, f)
            print(f"Tree(s) successfully saved to {folder}/{self.property}/{filename}.pkl")
        except IOError as e:
            raise IOError(f"Failed to save tree to {folder}/{self.property}/{filename}.pkl: {e}")
