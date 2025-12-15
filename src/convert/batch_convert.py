import os
import pickle
import torch
from glob import glob
import json
import argparse
from .nx_to_pyg import nx_list_to_pyg

import re

def find_best_epoch(report_dir):
    """
    Find the best epoch based on the highest weighted avg f1-score in classification_report_test_epoch*.json
    Returns the best epoch number (int)
    """
    best_f1 = -1
    best_epoch = None
    pattern = re.compile(r"classification_report_test_epoch(\d+)\.json")
    # Recursively find all matching files
    for root, dirs, files in os.walk(report_dir):
        for file in files:
            match = pattern.match(file)
            if match:
                epoch = int(match.group(1))
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as jf:
                        report = json.load(jf)
                    weighted = report.get('weighted avg', report.get('weighted', {}))
                    f1 = weighted.get('f1-score', 0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_epoch = epoch
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return best_epoch

def build_label_map(pred_json_path, epoch, split, use_pred=True):
    """
    Build a mapping from data_index to label for the specified epoch and split.
    If use_pred is True, use predicted_label, else use true_label.
    """
    with open(pred_json_path, 'r') as f:
        preds = json.load(f)
    label_map = {}
    for entry in preds:
        if entry['epoch'] == epoch and entry['dataset'] == split:
            idx = entry['data_index']
            label = entry['predicted_label'] if use_pred else entry['true_label']
            label_map[idx] = label
    return label_map

def main(label_source='llm', use_pred=True, hf_dataset_name='stanfordnlp/sst2', graph_type='syntactic', in_base=None, out_base=None, llm_dir=None):
    # Input and output base paths (allow override via args or env)
    default_base = os.environ.get('GRAPHTEXT_OUTPUT_DIR', 'outputs')
    IN_BASE = in_base or os.path.join(default_base, 'embeddings', hf_dataset_name)
    OUT_BASE = out_base or os.path.join(default_base, 'pyg_graphs', hf_dataset_name, graph_type)
    LLM_DIR = llm_dir or os.path.join(default_base, 'llm', hf_dataset_name)
    SPLITS = ['train', 'validation']

    os.makedirs(OUT_BASE, exist_ok=True)

    # If using LLM, determine best epoch and build label maps
    if label_source == 'llm':
        # Find predictions.json using os.walk
        pred_json = None
        pred_json_dir = None
        for root, dirs, files in os.walk(LLM_DIR):
            if 'predictions.json' in files:
                pred_json = os.path.join(root, 'predictions.json')
                pred_json_dir = root
                break
        if pred_json is None:
            raise FileNotFoundError("No predictions.json file found in LLM_DIR or its subdirectories.")

        # Find and select the best epoch from classification_report_test_epoch*.json files
        best_epoch = None
        best_f1 = -1
        import re
        pattern = re.compile(r"classification_report_test_epoch(\d+)\.json")
        # reduce files to a list with the pattern classification_report_test_epoch*.json
        files = glob(os.path.join(LLM_DIR, '**', 'classification_report_test_epoch*.json'), recursive=True)
        for file in files:
            with open(file, 'r') as f:
                f1 = json.load(f)['weighted avg']['f1-score']
                filename = os.path.basename(file)
                match = pattern.match(filename)
                if match:
                    epoch = int(match.group(1))
                    if f1 > best_f1:
                        best_f1 = f1
                        best_epoch = epoch
        if best_epoch is None:
            raise FileNotFoundError("No classification_report_test_epoch*.json files found in the same directory as predictions.json.")
        print(f"Best epoch: {best_epoch} (f1-score: {best_f1})")

        label_maps = {}
        for split in SPLITS:
            label_maps[split] = build_label_map(pred_json, best_epoch, split, use_pred=use_pred)
    elif label_source == 'original':
        import datasets
        ds = datasets.load_dataset(hf_dataset_name)
        label_maps = {split: {i: label for i, label in enumerate(ds[split]['label'])} for split in SPLITS}

    else:
        raise ValueError(f"Invalid label source: {label_source}")

    for split in SPLITS:
        in_dir = os.path.join(IN_BASE, split, graph_type)
        out_dir = os.path.join(OUT_BASE, graph_type, split)
        os.makedirs(out_dir, exist_ok=True)

        # Find all relevant .pkl files and sort numerically
        pkl_files = glob(os.path.join(in_dir, f'*{split}_batch_*.pkl'))
        pkl_files.sort(key=lambda x: int([s for s in x.split('_') if s.isdigit()][-1]))

        print(f"Processing split: {split}, graph type: {graph_type}, {len(pkl_files)} batch files found.")
        running_data_index = 0
        for pkl_file in pkl_files:
            print(f"  Processing {pkl_file} ...")
            with open(pkl_file, 'rb') as f:
                graphs = pickle.load(f)
            pyg_graphs = nx_list_to_pyg(graphs)

            # Assign label to Data.y for each graph, using running_data_index
            for i, graph in enumerate(graphs):
                data_index = running_data_index
                if data_index in label_maps[split]:
                    label = label_maps[split][data_index]
                else:
                    label = -1  # Unknown label
                pyg_graphs[i].y = torch.tensor([label], dtype=torch.long)
                pyg_graphs[i].data_index = data_index
                running_data_index += 1

            pt_file = os.path.join(out_dir, os.path.basename(pkl_file).replace('.pkl', '.pt'))
            torch.save(pyg_graphs, pt_file)
            print(f"  Saved {pt_file}")
    print("Batch conversion complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch convert NetworkX graphs to PyG with label assignment.')
    parser.add_argument('--label-source', choices=['llm','original'], default='llm', help='Source of labels: llm or original')
    parser.add_argument('--hf-dataset-name', default='stanfordnlp/sst2', help='Hugging Face dataset name')
    parser.add_argument('--graph-type', default='syntactic', help='Graph type (e.g., syntactic, constituency)')
    parser.add_argument('--use-pred', action='store_true', help='If set, use predicted_label; otherwise use true_label (llm mode only)')
    parser.add_argument('--in-base', default=None, help='Base directory for embedded graphs (defaults to outputs/embeddings/<dataset>)')
    parser.add_argument('--out-base', default=None, help='Base directory for PyG graphs (defaults to outputs/pyg_graphs/<dataset>/<graph_type>)')
    parser.add_argument('--llm-dir', default=None, help='Directory containing finetuned LLM outputs (defaults to outputs/llm/<dataset>)')
    args = parser.parse_args()
    main(
        label_source=args.label_source,
        use_pred=args.use_pred,
        hf_dataset_name=args.hf_dataset_name,
        graph_type=args.graph_type,
        in_base=args.in_base,
        out_base=args.out_base,
        llm_dir=args.llm_dir,
    )
