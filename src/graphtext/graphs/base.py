from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from ..registry import GRAPH_BUILDERS


@dataclass
class BuildArgs:
    graph_type: str
    dataset: str
    subsets: List[str]
    batch_size: int = 256
    device: str = "cuda:0"
    output_dir: str = "./outputs/graphs"
    window_size: int = 5  # for word-window graphs
    model_name: Optional[str] = None


class BaseGraphBuilder:
    name = "base"

    def process_dataset(self, args: BuildArgs) -> None:
        raise NotImplementedError


@GRAPH_BUILDERS.register("constituency")
class ConstituencyBuilder(BaseGraphBuilder):
    name = "constituency"

    def process_dataset(self, args: BuildArgs) -> None:
        # Delegate to the new intuitive path (shim to legacy)
        from src.graph_builders import tree_generator as tg
        tg.process_dataset(
            graph_type="constituency",
            dataset=args.dataset,
            subsets=args.subsets,
            batch_size=args.batch_size,
            device=args.device,
            output_dir=args.output_dir,
            model_name=args.model_name,
        )


@GRAPH_BUILDERS.register("syntactic")
class SyntacticBuilder(BaseGraphBuilder):
    name = "syntactic"

    def process_dataset(self, args: BuildArgs) -> None:
        from src.graph_builders import tree_generator as tg
        tg.process_dataset(
            graph_type="syntactic",
            dataset=args.dataset,
            subsets=args.subsets,
            batch_size=args.batch_size,
            device=args.device,
            output_dir=args.output_dir,
            model_name=args.model_name,
        )


@GRAPH_BUILDERS.register("window")
class WindowWordBuilder(BaseGraphBuilder):
    name = "window"

    def process_dataset(self, args: BuildArgs) -> None:
        """Build word-window graphs: nodes are words; edges connect words within +/- window_size.

        Saves numbered pickle batches matching the existing structure expected by
        the embedding step: each file contains a list with one tuple (graphs, labels).
        """
        import os
        import pickle as pkl
        from datasets import load_dataset
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        import networkx as nx
        import stanza

        # Prepare tokenizer for consistent word segmentation
        nlp = stanza.Pipeline(lang='en', processors='tokenize', tokenize_no_ssplit=True, use_gpu=args.device.startswith('cuda'))

        def build_graph_from_sentence(sentence: str) -> nx.DiGraph:
            doc = nlp(sentence)
            words = []
            for sent in doc.sentences:
                for tok in sent.tokens:
                    words.append(tok.text)
            g = nx.DiGraph()
            # 1-based ids, match stanza word ids convention
            for i, w in enumerate(words, start=1):
                g.add_node(i, text=w, type='word')
            k = max(0, int(args.window_size))
            if k > 0:
                for i in range(1, len(words) + 1):
                    for j in range(max(1, i - k), min(len(words), i + k) + 1):
                        if i == j:
                            continue
                        g.add_edge(i, j, label='window')
            g.graph['property'] = f"window_w{args.window_size}"
            return g

        def subsets_handler(dataset_name: str, subsets: List[str]) -> List[str]:
            # Mirror behavior of tree_generator
            subs = list(subsets)
            if dataset_name == "SetFit/ag_news":
                if "validation" in subs:
                    subs.remove("validation")
            elif dataset_name == "stanfordnlp/sst2":
                if "test" in subs:
                    subs.remove("test")
            return subs

        base_out = args.output_dir
        os.makedirs(base_out, exist_ok=True)
        subsets = subsets_handler(args.dataset, args.subsets)
        for subset in subsets:
            ds = load_dataset(args.dataset, split=subset)
            dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
            out_dir = os.path.join(base_out, args.dataset, subset, f"window_w{args.window_size}")
            os.makedirs(out_dir, exist_ok=True)
            iterator = 0
            for batch in tqdm(dl, desc=f"Processing {args.dataset}/{subset} window(k={args.window_size})"):
                # Find sentence/text and label
                if 'sentence' in batch:
                    texts = batch['sentence']
                elif 'text' in batch:
                    texts = batch['text']
                else:
                    raise ValueError("No 'sentence' or 'text' column in dataset batch")
                labels = batch.get('label', None)
                graphs = [build_graph_from_sentence(str(s)) for s in texts]
                payload = [(graphs, labels)]
                with open(os.path.join(out_dir, f"{iterator}.pkl"), 'wb') as f:
                    pkl.dump(payload, f)
                iterator += 1


# Note: Additional builders (e.g., n-grams, skip-grams, KNN) can be
# added by creating new classes here or in separate modules and registering with
# @GRAPH_BUILDERS.register("new_name").
