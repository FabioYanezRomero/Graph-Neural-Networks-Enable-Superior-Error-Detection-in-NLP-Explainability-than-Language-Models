from __future__ import annotations
from typing import List
import networkx as nx
from .base_generator import BaseTreeGenerator
from .registry import GENERATORS
import stanza


def _stanza_words(sentence: str, nlp) -> List[str]:
    doc = nlp(sentence)
    words: List[str] = []
    for sent in doc.sentences:
        for tok in sent.tokens:
            words.append(tok.text)
    return words


@GENERATORS.register("fully.word")
@GENERATORS.register("fully")
@GENERATORS.register("fully_connected")
class WordFullyConnectedGenerator(BaseTreeGenerator):
    def __init__(self, device: str = 'cuda:0'):
        super().__init__(property='fully.word', device=device)
        self._nlp = stanza.Pipeline(lang='en', processors='tokenize', tokenize_no_ssplit=True, use_gpu=device.startswith('cuda'))

    def _parse(self, sentences: List[str]):
        return sentences

    def _build_graph(self, sentences: List[str]) -> List[nx.DiGraph]:
        graphs: List[nx.DiGraph] = []
        for sentence in sentences:
            words = _stanza_words(sentence, self._nlp)
            g = nx.DiGraph()
            for i, w in enumerate(words, start=1):
                g.add_node(i, text=w, type='word')
            # fully connect directed except self
            for i in range(1, len(words) + 1):
                for j in range(1, len(words) + 1):
                    if i == j:
                        continue
                    g.add_edge(i, j, label='fully')
            g.graph['property'] = self.property
            graphs.append(g)
        return graphs

    def get_graph(self, sentences: List[str], ids=None) -> List[nx.DiGraph]:
        return self._build_graph(self._parse(sentences))


@GENERATORS.register("fully.token")
class TokenFullyConnectedGenerator(BaseTreeGenerator):
    def __init__(self, device: str = 'cuda:0', model_name: str = 'bert-base-uncased'):
        super().__init__(property='fully.token', device=device)
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _parse(self, sentences: List[str]):
        return sentences

    def _build_graph(self, sentences: List[str]) -> List[nx.DiGraph]:
        graphs: List[nx.DiGraph] = []
        for sentence in sentences:
            enc = self.tokenizer(sentence, add_special_tokens=False)
            tokens = self.tokenizer.convert_ids_to_tokens(enc['input_ids'])
            g = nx.DiGraph()
            for i, t in enumerate(tokens, start=1):
                g.add_node(i, text=t, type='token')
            for i in range(1, len(tokens) + 1):
                for j in range(1, len(tokens) + 1):
                    if i == j:
                        continue
                    g.add_edge(i, j, label='fully')
            g.graph['property'] = self.property
            graphs.append(g)
        return graphs

    def get_graph(self, sentences: List[str], ids=None) -> List[nx.DiGraph]:
        return self._build_graph(self._parse(sentences))

