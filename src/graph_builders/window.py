from __future__ import annotations
from typing import List, Optional
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


@GENERATORS.register("window.word")
@GENERATORS.register("window")
class WordWindowGenerator(BaseTreeGenerator):
    def __init__(self, device: str = 'cuda:0', k: int = 5, weights_path: Optional[str] = None):
        super().__init__(property=f'window.word.k{k}', device=device)
        self.k = max(0, int(k))
        self.weights_path = weights_path
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
            if self.k > 0:
                for i in range(1, len(words) + 1):
                    for j in range(max(1, i - self.k), min(len(words), i + self.k) + 1):
                        if i == j:
                            continue
                        g.add_edge(i, j, label='window')
            g.graph['property'] = self.property
            graphs.append(g)
        return graphs

    def get_graph(self, sentences: List[str], ids=None) -> List[nx.DiGraph]:
        return self._build_graph(self._parse(sentences))


@GENERATORS.register("window.token")
class TokenWindowGenerator(BaseTreeGenerator):
    def __init__(self, device: str = 'cuda:0', k: int = 5, model_name: str = 'bert-base-uncased', weights_path: Optional[str] = None):
        super().__init__(property=f'window.token.k{k}', device=device)
        from transformers import AutoTokenizer
        self.k = max(0, int(k))
        self.weights_path = weights_path

        resolved_model = model_name
        if weights_path and model_name in (None, '', 'bert-base-uncased'):
            try:
                from pathlib import Path
                import json
                cfg_path = Path(weights_path).with_name('config.json')
                if cfg_path.is_file():
                    with cfg_path.open() as f:
                        data = json.load(f)
                    for key in ('model_name', 'model_name_or_path', 'pretrained_model_name'):
                        candidate = data.get(key)
                        if isinstance(candidate, str) and candidate.strip():
                            resolved_model = candidate.strip()
                            break
            except Exception:
                pass

        self.tokenizer = AutoTokenizer.from_pretrained(resolved_model)

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
            if self.k > 0:
                for i in range(1, len(tokens) + 1):
                    for j in range(max(1, i - self.k), min(len(tokens), i + self.k) + 1):
                        if i == j:
                            continue
                        g.add_edge(i, j, label='window')
            g.graph['property'] = self.property
            graphs.append(g)
        return graphs

    def get_graph(self, sentences: List[str], ids=None) -> List[nx.DiGraph]:
        return self._build_graph(self._parse(sentences))
