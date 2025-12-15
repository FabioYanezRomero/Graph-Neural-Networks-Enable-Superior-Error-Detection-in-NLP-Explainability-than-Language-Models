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


@GENERATORS.register("ngrams.word")
@GENERATORS.register("ngrams")
class WordNGramGenerator(BaseTreeGenerator):
    def __init__(self, device: str = 'cuda:0', n: int = 2):
        super().__init__(property=f'ngrams.word.n{n}', device=device)
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
            # fully connect each sliding n-gram window (directed i->j for i<j)
            for start in range(1, len(words) - self.n + 2):
                group = list(range(start, start + self.n))
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        g.add_edge(group[i], group[j], label='ngram')
            g.graph['property'] = self.property
            graphs.append(g)
        return graphs

    def get_graph(self, sentences: List[str], ids=None) -> List[nx.DiGraph]:
        return self._build_graph(self._parse(sentences))


@GENERATORS.register("ngrams.token")
class TokenNGramGenerator(BaseTreeGenerator):
    def __init__(self, device: str = 'cuda:0', n: int = 2, model_name: str = 'bert-base-uncased'):
        super().__init__(property=f'ngrams.token.n{n}', device=device)
        from transformers import AutoTokenizer
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
            for start in range(1, len(tokens) - self.n + 2):
                group = list(range(start, start + self.n))
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        g.add_edge(group[i], group[j], label='ngram')
            g.graph['property'] = self.property
            graphs.append(g)
        return graphs

    def get_graph(self, sentences: List[str], ids=None) -> List[nx.DiGraph]:
        return self._build_graph(self._parse(sentences))

