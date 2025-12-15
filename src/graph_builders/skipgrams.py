from __future__ import annotations
from typing import List
import networkx as nx
import stanza

from .base_generator import BaseTreeGenerator
from .registry import GENERATORS


def _stanza_tokenize(sentence: str, nlp) -> List[str]:
    doc = nlp(sentence)
    words: List[str] = []
    for sent in doc.sentences:
        for tok in sent.tokens:
            words.append(tok.text)
    return words


@GENERATORS.register("skipgrams.word")
@GENERATORS.register("skipgrams")
class WordSkipGramGenerator(BaseTreeGenerator):
    def __init__(self, device: str = 'cuda:0', k: int = 2, n: int = 2):
        super().__init__(property=f'skipgrams.word.k{k}.n{n}', device=device)
        self.k = max(0, int(k))
        # Generalized n not commonly used for skip-gram; here we build complete directed edges across positions within k+1 hop distance for n=2
        self.n = max(2, int(n))
        self._nlp = stanza.Pipeline(lang='en', processors='tokenize', tokenize_no_ssplit=True, use_gpu=device.startswith('cuda'))

    def _parse(self, sentences: List[str]):
        return sentences

    def _build_graph(self, sentences: List[str]) -> List[nx.DiGraph]:
        graphs: List[nx.DiGraph] = []
        for sentence in sentences:
            words = _stanza_tokenize(sentence, self._nlp)
            g = nx.DiGraph()
            for i, w in enumerate(words, start=1):
                g.add_node(i, text=w, type='word')
            # Skip-grams as edges between i and i + s + 1 for s in [0..k] (bigrams with skips)
            for i in range(1, len(words)):
                for s in range(0, self.k + 1):
                    j = i + s + 1
                    if j <= len(words):
                        g.add_edge(i, j, label='skipgram')
            g.graph['property'] = self.property
            graphs.append(g)
        return graphs

    def get_graph(self, sentences: List[str], ids=None) -> List[nx.DiGraph]:
        return self._build_graph(self._parse(sentences))


@GENERATORS.register("skipgrams.token")
class TokenSkipGramGenerator(BaseTreeGenerator):
    def __init__(self, device: str = 'cuda:0', k: int = 2, n: int = 2, model_name: str = 'bert-base-uncased'):
        super().__init__(property=f'skipgrams.token.k{k}.n{n}', device=device)
        from transformers import AutoTokenizer
        self.k = max(0, int(k))
        self.n = max(2, int(n))
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
            for i in range(1, len(tokens)):
                for s in range(0, self.k + 1):
                    j = i + s + 1
                    if j <= len(tokens):
                        g.add_edge(i, j, label='skipgram')
            g.graph['property'] = self.property
            graphs.append(g)
        return graphs

    def get_graph(self, sentences: List[str], ids=None) -> List[nx.DiGraph]:
        return self._build_graph(self._parse(sentences))

