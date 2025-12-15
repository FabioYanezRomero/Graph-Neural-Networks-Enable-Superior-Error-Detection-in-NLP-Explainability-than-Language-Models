#!/usr/bin/env python3
import argparse
import os
import pickle as pkl
from typing import List, Tuple, Any, Optional

import networkx as nx
from datasets import get_dataset_split_names
import numpy as np


def _load_one_batch(path: str) -> Tuple[List[nx.DiGraph], Any]:
    with open(path, 'rb') as f:
        obj = pkl.load(f)
    # Expected format in this repo: [(graphs_list, labels_tensor)]
    if isinstance(obj, list) and obj and isinstance(obj[0], tuple) and len(obj[0]) >= 1:
        graphs = obj[0][0]
        labels = obj[0][1] if len(obj[0]) > 1 else None
        return graphs, labels
    # Alternate tolerant formats
    if isinstance(obj, list) and obj and isinstance(obj[0], nx.Graph):
        return obj, None
    raise ValueError(f"Unexpected pickle structure in {path}: type={type(obj)}")


def _validate_adjacency(g: nx.DiGraph) -> List[str]:
    errs: List[str] = []
    try:
        import numpy as np
        A = nx.to_numpy_array(g, dtype=float)
        n = g.number_of_nodes()
        if A.shape != (n, n):
            errs.append(f"adjacency shape mismatch {A.shape} vs nodes {n}")
        if np.isnan(A).any():
            errs.append("adjacency contains NaN")
        # No self loops expected unless explicitly added
        if n > 0 and np.any(np.diag(A) != 0):
            errs.append("self-loops present in adjacency")
    except Exception as e:
        errs.append(f"adjacency build error: {e}")
    return errs


def _validate_syntactic(g: nx.DiGraph) -> List[str]:
    errs = []
    if g.number_of_nodes() == 0:
        errs.append("no nodes")
    # At least some nodes should have 'text'
    has_text = any('text' in d and isinstance(d['text'], str) for _, d in g.nodes(data=True))
    if not has_text:
        errs.append("no node has 'text' attr")
    # There should be at least one root (in-degree 0)
    roots = [n for n, d in g.in_degree() if d == 0]
    if len(roots) == 0:
        errs.append("no root node (in-degree 0)")
    errs.extend(_validate_adjacency(g))
    return errs


def _validate_constituency(g: nx.DiGraph) -> List[str]:
    errs = []
    if g.number_of_nodes() == 0:
        errs.append("no nodes")
    # Constituency nodes should carry 'label'
    has_label = any('label' in d for _, d in g.nodes(data=True))
    if not has_label:
        errs.append("no node has 'label' attr")
    # Expect at least some edges labeled as constituency relation
    has_const_edge = any(d.get('label') == 'constituency relation' for _, _, d in g.edges(data=True))
    if not has_const_edge and g.number_of_edges() > 0:
        errs.append("no 'constituency relation' edges found")
    # Acyclic expectation (tree)
    try:
        if not nx.is_directed_acyclic_graph(g):
            errs.append("graph has cycles; expected tree/DAG")
    except Exception:
        pass
    errs.extend(_validate_adjacency(g))
    return errs


def _validate_window(g: nx.DiGraph) -> List[str]:
    errs = []
    if g.number_of_nodes() == 0:
        errs.append("no nodes")
    # Window graphs usually label nodes with type 'word' or 'token'
    node_types = {d.get('type') for _, d in g.nodes(data=True)}
    if not (('word' in node_types) or ('token' in node_types)):
        errs.append("node types missing ('word'/'token')")
    # For k>0, edges often labeled 'window'. If zero edges, don't fail hard.
    if g.number_of_edges() > 0:
        has_window = any(d.get('label') == 'window' for _, _, d in g.edges(data=True))
        if not has_window:
            errs.append("no 'window' edges found")
    # No self-loops expected
    if any(u == v for u, v in g.edges()):
        errs.append("self-loops present")
    errs.extend(_validate_adjacency(g))
    return errs


def _validate_labels(labels: Any, n_graphs: int, dataset: str, split: str) -> List[str]:
    errs: List[str] = []
    if labels is None:
        return errs
    try:
        import torch
        if isinstance(labels, torch.Tensor):
            if labels.ndim != 1 or labels.shape[0] != n_graphs:
                errs.append(f"labels shape mismatch {tuple(labels.shape)} vs graphs {n_graphs}")
            # Expected classes for a few known datasets
            allowed = {"stanfordnlp/sst2": {0, 1}, "SetFit/ag_news": {0, 1, 2, 3}}
            if split != 'test' and dataset in allowed:
                bad = [int(x) for x in labels.tolist() if int(x) not in allowed[dataset]]
                if bad:
                    errs.append(f"labels out of range for {dataset}: e.g., {bad[:5]}")
        else:
            # Accept list/array as well
            try:
                n = len(labels)
                if n != n_graphs:
                    errs.append(f"labels length {n} vs graphs {n_graphs}")
            except Exception:
                errs.append(f"labels type unsupported: {type(labels)}")
    except Exception as e:
        errs.append(f"label validation error: {e}")
    return errs


def validate_dir(dir_path: str, graph_type: str, max_files: int = 3, dataset: Optional[str] = None, split: Optional[str] = None) -> Tuple[bool, List[str]]:
    if not os.path.isdir(dir_path):
        return False, [f"missing directory: {dir_path}"]
    files = [f for f in os.listdir(dir_path) if f.endswith('.pkl')]
    if not files:
        return False, [f"no .pkl files in {dir_path}"]
    files.sort(key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x)
    files = files[:max_files]
    errs: List[str] = []
    for fname in files:
        graphs, labels = _load_one_batch(os.path.join(dir_path, fname))
        if not graphs:
            errs.append(f"{fname}: empty graphs list")
            continue
        # Validate a few graphs
        check = _validate_syntactic if graph_type.startswith('syntactic') else (
            _validate_constituency if graph_type.startswith('constituency') else _validate_window
        )
        for i, g in enumerate(graphs[:3]):
            g_errs = check(g)
            if g_errs:
                errs.append(f"{fname}[{i}]: " + ", ".join(g_errs))
        # Validate labels alignment with graphs
        if dataset and split:
            errs.extend(_validate_labels(labels, len(graphs), dataset, split))
    return len(errs) == 0, errs


def _load_embedding_graphs(path: str) -> List[nx.DiGraph]:
    with open(path, 'rb') as f:
        obj = pkl.load(f)
    if isinstance(obj, list) and obj and isinstance(obj[0], nx.Graph):
        return obj
    # some dumps may be a dict or tuple; try to unwrap lists inside
    if isinstance(obj, list):
        gs = [x for x in obj if isinstance(x, nx.Graph)]
        if gs:
            return gs
    raise ValueError(f"Unexpected embeddings pickle structure in {path}: type={type(obj)}")


def _validate_embeddings_dir(emb_dir: str, max_files: int = 2) -> Tuple[bool, List[str]]:
    if not os.path.isdir(emb_dir):
        return False, [f"missing embeddings dir: {emb_dir}"]
    files = [f for f in os.listdir(emb_dir) if f.endswith('.pkl')]
    if not files:
        return False, [f"no embeddings .pkl files in {emb_dir}"]
    files.sort()
    files = files[:max_files]
    errs: List[str] = []
    dim: Optional[int] = None
    for fname in files:
        try:
            graphs = _load_embedding_graphs(os.path.join(emb_dir, fname))
        except Exception as e:
            errs.append(f"{fname}: load error: {e}")
            continue
        for i, g in enumerate(graphs[:3]):
            embs = []
            for _, d in g.nodes(data=True):
                if 'embedding' in d:
                    arr = np.asarray(d['embedding'])
                    if arr.ndim != 1:
                        errs.append(f"{fname}[{i}]: embedding not 1D for a node: shape {arr.shape}")
                    else:
                        embs.append(arr)
            if not embs:
                errs.append(f"{fname}[{i}]: no node embeddings found")
                continue
            dims = {e.shape[0] for e in embs}
            if len(dims) != 1:
                errs.append(f"{fname}[{i}]: inconsistent node embedding dims: {sorted(dims)}")
            else:
                d0 = next(iter(dims))
                if dim is None:
                    dim = d0
                elif dim != d0:
                    errs.append(f"{fname}[{i}]: embedding dim {d0} != earlier {dim}")
    return len(errs) == 0, errs


def main():
    ap = argparse.ArgumentParser(description="Validate generated graph pickle batches.")
    ap.add_argument('--dataset', required=True, help='HF dataset name, e.g., stanfordnlp/sst2 or SetFit/ag_news')
    ap.add_argument('--out_base', default=os.environ.get('GRAPHTEXT_OUTPUT_DIR', 'outputs') + '/graphs', help='Base output dir for graphs')
    ap.add_argument('--graph_types', nargs='+', default=['syntactic', 'constituency', 'window.word.k5'], help='Graph types to validate')
    ap.add_argument('--splits', nargs='*', default=None, help='Optional splits; if omitted, auto-detect via datasets')
    ap.add_argument('--max_files', type=int, default=3, help='Max pickle files to scan per split')
    ap.add_argument('--check_embeddings', action='store_true', help='Also validate embeddings pickles under outputs/embeddings/...')
    ap.add_argument('--emb_base', default=os.environ.get('GRAPHTEXT_OUTPUT_DIR', 'outputs') + '/embeddings', help='Base output dir for embeddings')
    args = ap.parse_args()

    if args.splits is None:
        try:
            args.splits = list(get_dataset_split_names(args.dataset))
        except Exception:
            args.splits = ['train', 'validation', 'test']

    ok_all = True
    for gt in args.graph_types:
        print(f"\n== Checking graph_type={gt}")
        for split in args.splits:
            path = os.path.join(args.out_base, args.dataset, split, gt)
            ok, errs = validate_dir(path, gt, args.max_files, dataset=args.dataset, split=split)
            status = 'OK' if ok else 'FAIL'
            print(f"  [{status}] {path}")
            if errs:
                for e in errs[:10]:
                    print(f"    - {e}")
            ok_all = ok_all and ok
            if args.check_embeddings:
                emb_dir = os.path.join(args.emb_base, args.dataset, split, gt)
                ok_e, errs_e = _validate_embeddings_dir(emb_dir, max_files=max(1, args.max_files // 2))
                status_e = 'OK' if ok_e else 'FAIL'
                print(f"    [embeddings {status_e}] {emb_dir}")
                if errs_e:
                    for e in errs_e[:10]:
                        print(f"      - {e}")
                ok_all = ok_all and ok_e
    if not ok_all:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
