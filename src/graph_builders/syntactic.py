"""
Syntactic Tree Generator Module

This module provides functionality to create syntactic trees from sentences.
It leverages the Stanza NLP library to generate syntactic trees and converts them
into directed graphs where nodes represent words and edges represent syntactic relations.
"""

from typing import List, Dict, Union, Any, Optional, TYPE_CHECKING
import os
import networkx as nx
import torch
import torch.cuda
from .base_generator import BaseTreeGenerator
from .registry import GENERATORS
import concurrent.futures

if TYPE_CHECKING:
    # Only for type checking; avoids hard dependency at import time
    import stanza  # pragma: no cover



@GENERATORS.register("syntactic")
class SyntacticTreeGenerator(BaseTreeGenerator):
    """
    Creates syntactic trees from sentences.

    This class processes sentences using a Stanza syntactic parser and converts
    the parsed trees into directed graphs. Each node in the graph represents
    either a syntactic phrase or a word from the sentence.

    Attributes:
        model (str): Name or configuration of the syntactic parser model.
        property (str): Property type, always set to 'syntactic'.
        nlp: The loaded Stanza pipeline.
        device (str): The device to run the parser on (CPU or CUDA).
    """

    def __init__(self, device: str = 'cuda:0'):
        """
        Initialize the syntactic parser with Stanza.

        Args:
            device (str, optional): Device to run the parser on. Defaults to 'cuda:0'.

        Raises:
            RuntimeError: If the specified device is not available.
        """
        super().__init__(property='syntactic', device=device)
        
        # Verify device availability for better error handling
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please use 'cpu' instead or check CUDA installation.")
        
        self.device = device
        if device.startswith('cuda'):
            torch.cuda.set_device(device)

        # Lazy import stanza so module import doesn't fail if it's missing
        try:
            import stanza  # type: ignore
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "The 'stanza' package is required for syntactic graphs. "
                "Install with 'pip install stanza' and download English models via "
                "'python -c \"import stanza; stanza.download(\'en\')\"'."
            ) from e

        # Set up Stanza configuration
        stanza_device = 'gpu' if device.startswith('cuda') else 'cpu'

        # Ensure required models are available; if not, provide a clear hint
        try:
            self.nlp = stanza.Pipeline(
                lang='en',
                processors='tokenize,pos,lemma,depparse',
                package={'depparse': 'combined_charlm'},
                use_gpu=stanza_device == 'gpu',
                download_method=stanza.DownloadMethod.NONE,  # require pre-downloaded models
                tokenize_pretokenized=False,
                tokenize_no_ssplit=False,
                pos_batch_size=1000,
                depparse_batch_size=1000,
                depparse_pretagged=True,
            )
        except Exception as e:
            raise RuntimeError(
                "Stanza English models not found. Download them with: "
                "python -c \"import stanza; stanza.download('en', package='combined_charlm')\""
            ) from e
        print("Stanza pipeline with combined_charlm model initialized successfully.")
        

    def _parse(self, sentences: List[str]):
        """
        Parse each input instance separately to preserve 1:1 alignment.

        Returns a list of Stanza documents (one per input string).
        """
        docs: List[Any] = []
        for s in sentences:
            docs.append(self.nlp(str(s)))
        return docs

    def _build_single_graph(self, doc: Any) -> nx.DiGraph:
        """Build a single dependency graph for the entire doc, merging all Stanza
        sentences into one graph with disjoint components (no cross-sentence edges).
        Node ids are assigned sequentially to avoid collisions across sentences.
        """
        graph = nx.DiGraph()
        next_id = 1
        # Map per-sentence (local_id) -> global node id
        for sent in doc.sentences:
            local_to_global = {}
            # First pass: create nodes
            for tok in sent.tokens:
                w = tok.words[0]
                gid = next_id
                next_id += 1
                local_to_global[w.id] = gid
                graph.add_node(gid, text=w.text, pos=w.upos)
            # Second pass: add edges within the sentence
            for tok in sent.tokens:
                w = tok.words[0]
                if w.head != 0 and w.head in local_to_global:
                    src = local_to_global[w.head]
                    dst = local_to_global[w.id]
                    graph.add_edge(src, dst, deprel=w.deprel)
        graph.graph['property'] = self.property
        return graph

    def _build_graph(self, docs: List[Any]) -> List[nx.DiGraph]:
        """Build one graph per input document to preserve 1:1 alignment."""
        graphs: List[nx.DiGraph] = []
        for doc in docs:
            graphs.append(self._build_single_graph(doc))
        return graphs

    def get_graph(self, sentences: List[str], ids: Optional[List[str]] = None) -> List[nx.DiGraph]:
        """
        Generate syntactic dependency graphs for a list of sentences.

        Args:
            sentences (List[str]): List of sentences to process.
            ids (List[str], optional): Not used, for interface compatibility.
    
        Returns:
            List[nx.DiGraph]: List of syntactic dependency graphs.
        """
        docs = self._parse(sentences)
        return self._build_graph(docs)
