from __future__ import annotations
from typing import List, Tuple
import os
import numpy as np
import torch
import networkx as nx

from .base_generator import BaseTreeGenerator
from .registry import GENERATORS


def _stanza_words_and_spans(sentence, nlp) -> Tuple[List[str], List[Tuple[int,int]]]:
    doc = nlp(sentence)
    words: List[str] = []
    spans: List[Tuple[int,int]] = []
    for sent in doc.sentences:
        for tok in sent.tokens:
            words.append(tok.text)
            start = getattr(tok, 'start_char', None)
            end = getattr(tok, 'end_char', None)
            if start is None or end is None:
                # approximate via running pointer
                if not spans:
                    idx = 0
                else:
                    idx = spans[-1][1]
                start = idx
                end = idx + len(tok.text)
            spans.append((int(start), int(end)))
    return words, spans


def _encode_with_offsets(text, tokenizer, device):
    enc = tokenizer(text, return_tensors='pt', return_offsets_mapping=True, truncation=True)
    offsets = enc.pop('offset_mapping')
    enc = {k: v.to(device) for k, v in enc.items()}
    return enc, offsets.squeeze(0).tolist()


def _aggregate_subwords(hidden, offsets, spans):
    out = []
    for (w_start, w_end) in spans:
        idxs = []
        for i, (t_start, t_end) in enumerate(offsets):
            if t_end == t_start:
                continue
            if max(w_start, t_start) < min(w_end, t_end):
                idxs.append(i)
        if not idxs:
            out.append(hidden[0] * 0)
        else:
            out.append(hidden[idxs].mean(axis=0))
    return np.stack(out, axis=0)  # [num_words, hid]


def _cosine_knn(x: np.ndarray, k: int) -> List[List[int]]:
    # x: [n, d]
    x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    sims = x_norm @ x_norm.T  # [n, n]
    np.fill_diagonal(sims, -np.inf)
    idxs = np.argpartition(-sims, kth=min(k, max(1, x.shape[0]-1))-1, axis=1)[:, :k]
    # sort neighbors by similarity
    rows = []
    for i in range(x.shape[0]):
        neigh = idxs[i]
        order = np.argsort(-sims[i, neigh])
        rows.append(neigh[order].tolist())
    return rows


@GENERATORS.register("knn.word")
@GENERATORS.register("knn")
class WordKNNGenerator(BaseTreeGenerator):
    def __init__(self, device: str = 'cuda:0', k: int = 4, model_name: str | None = None):
        super().__init__(property=f'knn.word.k{k}', device=device)
        import stanza
        from transformers import AutoTokenizer, AutoModel
        self.k = max(1, int(k))
        self.model_name = model_name or os.environ.get('GRAPHTEXT_MODEL_NAME', 'bert-base-uncased')
        self._nlp = stanza.Pipeline(lang='en', processors='tokenize', tokenize_no_ssplit=True, use_gpu=device.startswith('cuda'))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to('cuda' if device.startswith('cuda') and torch.cuda.is_available() else 'cpu')
        self.model.eval()
        self.model_device = next(self.model.parameters()).device

    def _parse(self, sentences: List[str]):
        return sentences

    def _build_graph(self, sentences: List[str]) -> List[nx.DiGraph]:
        graphs: List[nx.DiGraph] = []
        for sentence in sentences:
            words, spans = _stanza_words_and_spans(sentence, self._nlp)
            enc, offsets = _encode_with_offsets(sentence, self.tokenizer, self.model_device)
            with torch.no_grad():
                hidden = self.model(**enc).last_hidden_state.squeeze(0).detach().cpu().numpy()  # [T, H]
            word_vecs = _aggregate_subwords(hidden, offsets, spans)  # [W, H]
            # Build graph
            g = nx.DiGraph()
            for i, w in enumerate(words, start=1):
                g.add_node(i, text=w, type='word')
            if len(words) > 1:
                neighs = _cosine_knn(word_vecs, min(self.k, len(words)-1))
                for i, nbrs in enumerate(neighs, start=1):
                    for j0 in nbrs:
                        g.add_edge(i, j0+1, label='knn')
            g.graph['property'] = self.property
            graphs.append(g)
        return graphs

    def get_graph(self, sentences: List[str], ids=None) -> List[nx.DiGraph]:
        return self._build_graph(self._parse(sentences))


@GENERATORS.register("knn.token")
class TokenKNNGenerator(BaseTreeGenerator):
    def __init__(self, device: str = 'cuda:0', k: int = 4, model_name: str | None = None):
        super().__init__(property=f'knn.token.k{k}', device=device)
        from transformers import AutoTokenizer, AutoModel
        self.k = max(1, int(k))
        self.model_name = model_name or os.environ.get('GRAPHTEXT_MODEL_NAME', 'bert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to('cuda' if device.startswith('cuda') and torch.cuda.is_available() else 'cpu')
        self.model.eval()
        self.model_device = next(self.model.parameters()).device

    def _parse(self, sentences: List[str]):
        return sentences

    def _build_graph(self, sentences: List[str]) -> List[nx.DiGraph]:
        graphs: List[nx.DiGraph] = []
        for sentence in sentences:
            enc = self.tokenizer(sentence, add_special_tokens=False, return_tensors='pt').to(self.model_device)
            with torch.no_grad():
                hidden = self.model(**enc).last_hidden_state.squeeze(0).detach().cpu().numpy()  # [T, H]
            n = hidden.shape[0]
            g = nx.DiGraph()
            tokens = self.tokenizer.convert_ids_to_tokens(enc['input_ids'].squeeze(0).tolist())
            for i, t in enumerate(tokens, start=1):
                g.add_node(i, text=t, type='token')
            if n > 1:
                neighs = _cosine_knn(hidden, min(self.k, n-1))
                for i, nbrs in enumerate(neighs, start=1):
                    for j0 in nbrs:
                        g.add_edge(i, j0+1, label='knn')
            g.graph['property'] = self.property
            graphs.append(g)
        return graphs

    def get_graph(self, sentences: List[str], ids=None) -> List[nx.DiGraph]:
        return self._build_graph(self._parse(sentences))

