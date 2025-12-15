"""
Constituency Tree Generator Module

This module provides functionality to create constituency trees from sentences.
It leverages the Stanza NLP library to generate constituency trees and converts them
into directed graphs where nodes represent constituents and words.
"""

from typing import List, Dict, Union, Any, Optional
import os
import networkx as nx
import torch
import torch.cuda
import stanza
from .base_generator import BaseTreeGenerator
from .registry import GENERATORS

# Dictionary mapping constituency labels to more descriptive phrases
POS_MAPPER = {
    'ROOT': '«ROOT»',  # Added root node designation [1]
    'SENTENCE': '«SENTENCE»',  # Confirmed sentence label [1][2]
    'CC': '«COORDINATING CONJUNCTION»',
    'CD': '«CARDINAL NUMBER»',
    'DT': '«DETERMINER»',
    'EX': '«EXISTENTIAL THERE»',
    'FW': '«FOREIGN WORD»',
    'IN': '«PREPOSITION OR SUBORDINATING CONJUNCTION»',
    'JJ': '«ADJECTIVE»',
    'JJR': '«ADJECTIVE, COMPARATIVE»',
    'JJS': '«ADJECTIVE, SUPERLATIVE»',
    'LS': '«LIST MARKER»',
    'MD': '«MODAL VERB»',
    'NN': '«NOUN, SINGULAR OR MASS»',
    'NNS': '«NOUN, PLURAL»',
    'NNP': '«PROPER NOUN, SINGULAR»',
    'NNPS': '«PROPER NOUN, PLURAL»',
    'PDT': '«PREDETERMINER»',
    'POS': '«POSSESSIVE ENDING»',
    'PRP': '«PERSONAL PRONOUN»',
    'PRP$': '«POSSESSIVE PRONOUN»',
    'RB': '«ADVERB»',
    'RBR': '«ADVERB, COMPARATIVE»',
    'RBS': '«ADVERB, SUPERLATIVE»',
    'RP': '«PARTICLE»',
    'SYM': '«SYMBOL»',
    'TO': '«TO»',
    'UH': '«INTERJECTION»',
    'VB': '«VERB, BASE FORM»',
    'VBD': '«VERB, PAST TENSE»',
    'VBG': '«VERB, GERUND OR present participle»',
    'VBN': '«VERB, past participle»',
    'VBP': '«VERB, non-3rd person singular present»',
    'VBZ': '«VERB, 3rd person singular present»',
    'WDT': '«WH-DETERMINER»',
    'WP': '«WH-PRONOUN»',
    'WP$': '«WH-POSSESSIVE PRONOUN»',
    'WRB': '«WH-ADVERB»',
}



CONSTITUENCY_MAPPER = {
    'NP': '«NOUN PHRASE»',
    'VP': '«VERB PHRASE»',
    'PP': '«PREPOSITIONAL PHRASE»',
    'ADJP': '«ADJECTIVE PHRASE»',
    'ADVP': '«ADVERB PHRASE»',
    'SBAR': '«SUBORDINATE CLAUSE»',
    'PRT': '«PARTICLE»',
    'INTJ': '«INTERJECTION»',
    'CONJP': '«CONJUCTION PHRASE»',
    'LST': '«LIST MARKER»',
    'UCP': '«UNLIKE COORDINATED PHRASE»',
    'PRN': '«PARENTETICAL»',
    'FRAG': '«FRAGMENT»',
    'SINV': '«INVERTED SENTENCE»',
    'SBARQ': '«SUBORDINATE CLAUSE QUESTION»',
    'SQ': '«QUESTION»',
    'WHADJP': '«WH-ADJECTIVE PHRASE»',
    'WHAVP': '«WH-ADVERB PHRASE»',
    'WHNP': '«WH-NOUN PHRASE»',
    'WHPP': '«WH-PREPOSITIONAL PHRASE»',
    'RRC': '«REDUCED RELATIVE CLAUSE»',
    'NX': '«NOUN PHRASE (NO HEAD)»',
    'WHADVP': '«WH-ADVERB PHRASE»',
    'QP': '«QUANTIFIER PHRASE»',
    'NAC': '«NOT A CONSTITUENT»',
    'X': '«UNKNOWN»',
    'HYPH': '«HYPHEN»',
    'HVS': '«HYPHENATED VERB SUBSTITUTION»',
    'NML': '«NOMINALIZATION»',
    'LRB': '«LEFT PARENTHESIS»',
    'RRB': '«RIGHT PARENTHESIS»', 
}


PHRASE_MAPPER = {
    # POS TAGS
    'ROOT': '«ROOT»',  # Added root node designation [1]
    'SENTENCE': '«SENTENCE»',  # Confirmed sentence label [1][2]
    'CC': '«COORDINATING CONJUNCTION»',
    'CD': '«CARDINAL NUMBER»',
    'DT': '«DETERMINER»',
    'EX': '«EXISTENTIAL THERE»',
    'FW': '«FOREIGN WORD»',
    'IN': '«PREPOSITION OR SUBORDINATING CONJUNCTION»',
    'JJ': '«ADJECTIVE»',
    'JJR': '«ADJECTIVE, COMPARATIVE»',
    'JJS': '«ADJECTIVE, SUPERLATIVE»',
    'LS': '«LIST MARKER»',
    'MD': '«MODAL VERB»',
    'NN': '«NOUN, SINGULAR OR MASS»',
    'NNS': '«NOUN, PLURAL»',
    'NNP': '«PROPER NOUN, SINGULAR»',
    'NNPS': '«PROPER NOUN, PLURAL»',
    'PDT': '«PREDETERMINER»',
    'POS': '«POSSESSIVE ENDING»',
    'PRP': '«PERSONAL PRONOUN»',
    'PRP$': '«POSSESSIVE PRONOUN»',
    'RB': '«ADVERB»',
    'RBR': '«ADVERB, COMPARATIVE»',
    'RBS': '«ADVERB, SUPERLATIVE»',
    'RP': '«PARTICLE»',
    'SYM': '«SYMBOL»',
    'TO': '«TO»',
    'UH': '«INTERJECTION»',
    'VB': '«VERB, BASE FORM»',
    'VBD': '«VERB, PAST TENSE»',
    'VBG': '«VERB, GERUND OR present participle»',
    'VBN': '«VERB, past participle»',
    'VBP': '«VERB, non-3rd person singular present»',
    'VBZ': '«VERB, 3rd person singular present»',
    'WDT': '«WH-DETERMINER»',
    'WP': '«WH-PRONOUN»',
    'WP$': '«WH-POSSESSIVE PRONOUN»',
    'WRB': '«WH-ADVERB»',
    # CONSTITUENCY TAGS
    'NP': '«NOUN PHRASE»',
    'VP': '«VERB PHRASE»',
    'PP': '«PREPOSITIONAL PHRASE»',
    'ADJP': '«ADJECTIVE PHRASE»',
    'ADVP': '«ADVERB PHRASE»',
    'SBAR': '«SUBORDINATE CLAUSE»',
    'PRT': '«PARTICLE»',
    'INTJ': '«INTERJECTION»',
    'CONJP': '«CONJUCTION PHRASE»',
    'LST': '«LIST MARKER»',
    'UCP': '«UNLIKE COORDINATED PHRASE»',
    'PRN': '«PARENTETICAL»',
    'FRAG': '«FRAGMENT»',
    'SINV': '«INVERTED SENTENCE»',
    'SBARQ': '«SUBORDINATE CLAUSE QUESTION»',
    'SQ': '«QUESTION»',
    'WHADJP': '«WH-ADJECTIVE PHRASE»',
    'WHAVP': '«WH-ADVERB PHRASE»',
    'WHNP': '«WH-NOUN PHRASE»',
    'WHPP': '«WH-PREPOSITIONAL PHRASE»',
    'RRC': '«REDUCED RELATIVE CLAUSE»',
    'NX': '«NOUN PHRASE (NO HEAD)»',
    'WHADVP': '«WH-ADVERB PHRASE»',
    'QP': '«QUANTIFIER PHRASE»',
    'NAC': '«NOT A CONSTITUENT»',
    'X': '«UNKNOWN»',
    'HYPH': '«HYPHEN»',
    'HVS': '«HYPHENATED VERB SUBSTITUTION»',
    'NML': '«NOMINALIZATION»',
    'LRB': '«LEFT PARENTHESIS»',
    'RRB': '«RIGHT PARENTHESIS»',    # Add more constituency tags as needed
}


@GENERATORS.register("constituency")
class ConstituencyTreeGenerator(BaseTreeGenerator):
    """
    Creates constituency trees from sentences.

    This class processes sentences using a Stanza constituency parser and converts
    the parsed trees into directed graphs. Each node in the graph represents
    either a constituent phrase or a word from the sentence.

    Attributes:
        model (str): Name or configuration of the constituency parser model.
        property (str): Property type, always set to 'constituency'.
        nlp: The loaded Stanza pipeline.
        device (str): The device to run the parser on (CPU or CUDA).
    """

    def __init__(self, device: str = 'cuda:0'):
        """
        Initialize the constituency parser with Stanza.

        Args:
            device (str, optional): Device to run the parser on. Defaults to 'cuda:0'.

        Raises:
            RuntimeError: If the specified device is not available.
        """
        super().__init__(property='constituency', device=device)
        
        # Verify device availability for better error handling
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please use 'cpu' instead or check CUDA installation.")
        
        self.device = device
        if device.startswith('cuda'):
            torch.cuda.set_device(device)
        
        # Set up Stanza configuration
        stanza_device = 'gpu' if device.startswith('cuda') else 'cpu'
        
        # Initialize the Stanza pipeline with available constituency parser
        print("Initializing Stanza pipeline with constituency parser...")
        try:
            self.nlp = stanza.Pipeline(
                lang='en',
                processors='tokenize,pos,lemma,depparse,constituency',
                package={
                    'pos': 'combined_charlm',
                    'lemma': 'combined_nocharlm',
                    'depparse': 'combined_charlm',
                    'constituency': 'ptb3-revised_charlm'
                },
                use_gpu=(stanza_device == 'gpu'),
                download_method=stanza.DownloadMethod.NONE,  # We already downloaded the models
                tokenize_pretokenized=False,
                tokenize_no_ssplit=False,
                pos_batch_size=1000,
                depparse_batch_size=1000,
                constituency_batch_size=1000,
                constituency_pretagged=True  # Use POS tags from the POS tagger
            )
        except Exception as e:
            print(f"Error initializing pipeline: {e}")
            print("Falling back to simpler pipeline...")
            self.nlp = stanza.Pipeline(
                lang='en',
                processors='tokenize,pos,constituency',
                package='default',
                use_gpu=(stanza_device == 'gpu'),
                download_method=stanza.DownloadMethod.NONE,
                tokenize_pretokenized=False,
                tokenize_no_ssplit=False,
                pos_batch_size=1000,
                constituency_batch_size=1000
            )
        print("Stanza pipeline with constituency parser initialized successfully.")
        

    def _parse(self, sentences: List[str]):
        """
        Parse sentences using the Stanza constituency parser.
        If a sentence is split into multiple sentences, they are combined into a single parse.

        Args:
            sentences (List[str]): List of sentences to parse.

        Returns:
            List: List of parsed constituency trees, one per input sentence.
        """
        trees = []
        print(f"Starting to parse {len(sentences)} sentences...")
        for i, sentence in enumerate(sentences):
            if i % 10 == 0:  # Print progress every 10 sentences
                print(f"Processing sentence {i+1}/{len(sentences)}")
            try:
                # Process the current sentence with timeout protection
                import signal
                from contextlib import contextmanager

                @contextmanager
                def timeout_context(seconds):
                    def timeout_handler(signum, frame):
                        raise TimeoutError(f"Parsing timeout after {seconds} seconds")

                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(seconds)
                    try:
                        yield
                    finally:
                        signal.alarm(0)

                try:
                    with timeout_context(30):  # 30 second timeout per sentence
                        doc = self.nlp(sentence)
                except TimeoutError:
                    print(f"Timeout parsing sentence {i+1}, skipping...")
                    # Create a simple fallback tree
                    words = sentence.split()[:50]  # Limit to 50 words
                    flat_tree = ['ROOT', ['S'] + [['WORD', word] for word in words]]
                    trees.append(flat_tree)
                    continue

                if not doc.sentences:
                    raise ValueError("No sentences returned by parser")

                # If we get multiple sentences, create a combined parse
                if len(doc.sentences) > 1:
                    # Create a new root node for the combined parse
                    combined_parse = ['ROOT', ['S']]  # Start with ROOT -> S structure
                    
                    for sent in doc.sentences:
                        if hasattr(sent, 'constituency') and sent.constituency:
                            # Get the parse tree as a string
                            parse_str = str(sent.constituency)
                            # Parse it into a nested list structure
                            sent_parse = self._tree_to_list(parse_str)
                            # If the parse starts with ['ROOT', ...], take the children of ROOT
                            if (isinstance(sent_parse, list) and len(sent_parse) > 1 and 
                                isinstance(sent_parse[0], str) and sent_parse[0] == 'ROOT'):
                                # Add all children of ROOT (skipping ROOT itself)
                                combined_parse[1].extend(sent_parse[1:])
                            else:
                                # Otherwise add the entire parse
                                combined_parse[1].append(sent_parse)
                                
                    trees.append(combined_parse)
                else:
                    # Single sentence case
                    sent = doc.sentences[0]
                    if hasattr(sent, 'constituency') and sent.constituency:
                        trees.append(sent.constituency)
                    else:
                        raise ValueError("No constituency parse available")
                        
            except Exception as e:
                print(f"Error processing sentence {i}: {e}")
                print(f"Sentence content: {sentence}")
                # Fallback: create a simple flat structure
                words = sentence.split()[:100]  # Limit to 100 words
                flat_tree = ['ROOT', ['S'] + [['WORD', word] for word in words]]
                trees.append(flat_tree)
                continue
                
        if len(trees) != len(sentences):
            raise RuntimeError(f"Parse count mismatch: expected {len(sentences)}, got {len(trees)}")
            
        return trees

    def _tree_to_list(self, tree) -> Union[str, List]:
        """
        Convert the parsed constituency tree into a nested list structure.

        Args:
            tree: A Stanza constituency tree, subtree, or list.

        Returns:
            Union[str, List]: A string for leaf nodes or a list for non-leaf nodes.
        """
        # If tree is already a list, return it as is
        if isinstance(tree, list):
            return tree
            
        # If tree is a string, it's already in string format
        if isinstance(tree, str):
            try:
                # Use a stack-based iterative parser
                stack = []
                current = []
                i = 0
                n = len(tree)
                
                while i < n:
                    if tree[i] == ' ':
                        i += 1
                        continue
                        
                    if tree[i] == '(':
                        # Push current context to stack and start new one
                        stack.append(current)
                        current = []
                        i += 1
                    elif tree[i] == ')':
                        # Pop the last item from stack as parent
                        if current:
                            if stack:
                                parent = stack.pop()
                                parent.append(current)
                                current = parent
                            else:
                                # This is the root
                                if len(current) == 1:
                                    return current[0]
                                return current
                        i += 1
                    else:
                        # Read token
                        j = i
                        while j < n and tree[j] not in '() ':
                            j += 1
                        token = tree[i:j].strip()
                        if token:
                            current.append(token)
                        i = j
                        
                # Handle any remaining context
                if stack:
                    while stack:
                        parent = stack.pop()
                        if current:
                            parent.append(current)
                        current = parent
                        
                return current[0] if len(current) == 1 else current
                
            except Exception as e:
                print(f"Error parsing tree string: {e}")
                # Fall back to a simple flat structure
                words = [w for w in tree.split() if w not in '()']
                return ['ROOT', ['S'] + [['WORD', word] for word in words]]

        # Original logic for Stanza Tree objects
        if tree.is_leaf():
            return tree.label
        
        # Handle non-leaf nodes (constituents)
        return [tree.label] + [self._tree_to_list(child) for child in tree.children]

        # Fallback for unknown types
        return str(tree)

    def _build_graph(self, graph: nx.DiGraph, node_list: List, sentence: str, parent_id: str = '', graph_id: str = None, node_id_counter=None, parent_nid=None) -> nx.DiGraph:
        """
        Add edges and nodes to the graph from the node list.

        Args:
            graph (nx.DiGraph): The graph to build.
            node_list (List): The nested list representing the constituency tree.
            sentence (str): The original sentence.
            parent_id (str, optional): ID of the parent node. Defaults to ''.
            graph_id (str, optional): ID for the graph. Defaults to None.

        Returns:
            nx.DiGraph: The constructed graph.
        """
        # Setup id counter if not provided
        if node_id_counter is None:
            node_id_counter = {'val': 0}

        import string
        parent_label = node_list[0]
        children = node_list[1:]
        # Check if parent is a punctuation constituent (single character and in string.punctuation)
        if isinstance(parent_label, str) and len(parent_label) == 1 and parent_label in string.punctuation:
            # Do not add the punctuation constituent node, connect children directly to parent's parent
            for i, child in enumerate(children):
                if isinstance(child, list):
                    # Recursively process children, connecting to parent's parent
                    self._build_graph(graph, child, sentence, parent_id, graph_id, node_id_counter, parent_nid=parent_nid)
                else:
                    # Add leaf node (punctuation word)
                    try:
                        leaf_nid = node_id_counter['val']
                        node_id_counter['val'] += 1
                        mapped_leaf_label = PHRASE_MAPPER.get(child, child)
                        if parent_nid is not None:
                            # Find the parent node key by id
                            parent_node_key = None
                            for k, v in graph.nodes(data=True):
                                if v.get('id') == parent_nid:
                                    parent_node_key = k
                                    break
                            if parent_node_key is not None:
                                graph.add_node(leaf_nid, id=leaf_nid, label=mapped_leaf_label)
                                graph.add_edge(parent_node_key, leaf_nid, label="constituency relation")
                        else:
                            # If no parent_nid, treat as root
                            graph.add_node(leaf_nid, id=leaf_nid, label=mapped_leaf_label)
                    except Exception:
                        continue
            return graph
        # Normal case: not a punctuation constituent
        parent = parent_label + parent_id
        if parent not in graph:
            nid = node_id_counter['val']
            node_id_counter['val'] += 1
            label_key = parent_label
            mapped_label = PHRASE_MAPPER.get(label_key, label_key)
            graph.add_node(parent, id=nid, label=mapped_label)
        for i, child in enumerate(children):
            if isinstance(child, list):
                child_id = parent_id + str(i)
                child_label = child[0]
                node_key = str(child_label) + str(child_id)
                # Only skip adding node if it's a punctuation constituent
                if not (isinstance(child_label, str) and len(child_label) == 1 and child_label in string.punctuation):
                    if node_key not in graph:
                        child_nid = node_id_counter['val']
                        node_id_counter['val'] += 1
                        mapped_child_label = PHRASE_MAPPER.get(child_label, child_label)
                        graph.add_node(node_key, id=child_nid, label=mapped_child_label)
                    graph.add_edge(parent, node_key, label="constituency relation")
                    self._build_graph(graph, child, sentence, child_id, graph_id, node_id_counter, parent_nid=graph.nodes[parent]['id'])
                else:
                    # If child is a punctuation constituent, skip adding node and process its children directly
                    self._build_graph(graph, child, sentence, child_id, graph_id, node_id_counter, parent_nid=graph.nodes[parent]['id'])
            else:
                try:
                    leaf_nid = node_id_counter['val']
                    node_id_counter['val'] += 1
                    mapped_leaf_label = PHRASE_MAPPER.get(child, child)
                    graph.add_node(leaf_nid, id=leaf_nid, label=mapped_leaf_label)
                    graph.add_edge(parent, leaf_nid, label="constituency relation")
                except Exception:
                    continue
        return graph
        # Add graph metadata
        graph.graph['property'] = self.property
        if graph_id:
            graph.graph['id'] = graph_id
        return graph
    
    def _remove_nodes_and_reconnect(self, graph: nx.DiGraph) -> None:
        """
        Remove nodes that start with '_' and reconnect their parents to their children.

        Args:
            graph (nx.DiGraph): The graph to modify.
        """
        nodes_to_remove = [node for node in graph.nodes() if str(node).startswith('_')]
        for node in nodes_to_remove:
            # Get the parents and children of the node
            parents = list(graph.predecessors(node))
            children = list(graph.successors(node))
            
            # Connect each parent node to each child node
            for parent in parents:
                for child in children:
                    # Add an edge from parent to child
                    graph.add_edge(parent, child, label="constituency relation")
            
            # Remove the node
            graph.remove_node(node)

    def get_graph(self, sentences: List[str], ids: List[str] = None) -> List[nx.DiGraph]:
        """
        Generate constituency trees for a list of sentences.

        Args:
            sentences (List[str]): List of sentences to process.
            ids (List[str], optional): List of IDs to assign to each graph. Defaults to None.

        Returns:
            List[nx.DiGraph]: List of constituency trees.
            
        Raises:
            ValueError: If ids are provided but don't match the number of sentences.
        """
        if ids is not None and len(ids) != len(sentences):
            raise ValueError(f"Number of ids ({len(ids)}) must match number of sentences ({len(sentences)})")
        
        try:
            # Parse the sentences
            constituency_trees = self._parse(sentences)
            
            # Create a graph for each sentence
            graphs = []
            for i, tree in enumerate(constituency_trees):
                # Convert tree to nested list
                tree_list = self._tree_to_list(tree)
                
                # Create a new graph
                graph = nx.DiGraph()
                
                # Build the graph from the tree list
                graph = self._build_graph(
                    graph=graph,
                    node_list=tree_list,
                    sentence=sentences[i],
                    graph_id=ids[i] if ids else None,
                    node_id_counter={'val': 0}  # reset id counter for each graph
                )
                
                # Remove nodes that start with '_' and reconnect their parents to their children
                self._remove_nodes_and_reconnect(graph)
                
                graphs.append(graph)
            
            return graphs
            
        except Exception as e:
            raise RuntimeError(f"Error generating constituency trees: {e}")
