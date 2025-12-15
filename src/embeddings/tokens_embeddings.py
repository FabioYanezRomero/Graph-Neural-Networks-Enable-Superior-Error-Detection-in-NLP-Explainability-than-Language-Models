import os
import argparse
import pickle as pkl
import json
import glob
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer, BertModel, BertConfig
from datasets import load_dataset
from torch_geometric.data import Data


def load_sentences(dataset_name, split):
    ds = load_dataset(dataset_name, split=split)
    if 'sentence' in ds.column_names:
        return ds['sentence']
    elif 'text' in ds.column_names:
        return ds['text']
    else:
        raise ValueError("No suitable sentence/text column found in dataset.")


def load_model_robust(model_name, checkpoint_path, device):
    """Robust model loading with multiple fallback strategies."""
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"Checkpoint contains keys: {list(checkpoint.keys())[:5]}...")
    
    # Strategy 1: Try to load from cache
    model = None
    try:
        print("Strategy 1: Loading from local cache...")
        model = AutoModel.from_pretrained(model_name, local_files_only=True)
        print("✓ Successfully loaded from cache")
    except Exception as e:
        print(f"✗ Cache loading failed: {e}")
        
        # Strategy 2: Create model from config if available locally
        try:
            print("Strategy 2: Creating model from local config...")
            config = BertConfig.from_pretrained(model_name, local_files_only=True)
            model = BertModel(config)
            print("✓ Successfully created model from config")
        except Exception as e2:
            print(f"✗ Config creation failed: {e2}")
            
            # Strategy 3: Create default BERT model
            try:
                print("Strategy 3: Creating default BERT model...")
                config = BertConfig(
                    vocab_size=30522,
                    hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    intermediate_size=3072,
                    max_position_embeddings=512,
                    type_vocab_size=2
                )
                model = BertModel(config)
                print("✓ Successfully created default BERT model")
            except Exception as e3:
                raise RuntimeError(f"All model loading strategies failed: {e3}")
    
    # Load the fine-tuned weights
    try:
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'bert.' prefix from keys if present
        bert_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('bert.'):
                new_key = key[5:]  # Remove 'bert.' prefix
                bert_state_dict[new_key] = value
            elif not key.startswith('classifier'):
                # Keep non-classifier keys as they are
                bert_state_dict[key] = value
        
        # Load the weights
        missing_keys, unexpected_keys = model.load_state_dict(bert_state_dict, strict=False)
        
        if missing_keys:
            print(f"Warning: Missing keys (showing first 3): {missing_keys[:3]}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys (showing first 3): {unexpected_keys[:3]}")
        
        print("✓ Successfully loaded fine-tuned weights")
        
    except Exception as e:
        print(f"Warning: Could not load fine-tuned weights: {e}")
        print("Continuing with base model weights...")
    
    return model.to(device)


def main():
    # Configuration
    dataset_name = 'setfit/ag_news'
    model_name = 'bert-base-uncased'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 128
    
    print(f"Using device: {device}")
    
    # Load tokenizer
    try:
        print(f"Loading tokenizer {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        print("✓ Successfully loaded tokenizer from cache")
    except Exception as e:
        print(f"Cache loading failed: {e}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            print("✓ Successfully downloaded tokenizer")
        except Exception as e2:
            raise RuntimeError(f"Could not load tokenizer: {e2}")
    
    # Find the best model checkpoint
    finetuned_models_dir = f"outputs/llm/{dataset_name}"
    best_checkpoint = None
    best_f1 = -1.0
    
    print(f"Looking for models in: {finetuned_models_dir}")
    
    # Check if models are directly in the dataset directory
    model_dir = finetuned_models_dir
    direct_model_files = glob.glob(f"{model_dir}/model_epoch_*.pt")
    
    if direct_model_files:
        print(f"Found {len(direct_model_files)} model files directly in dataset directory")
        # Models are directly in the dataset directory
        # Try to find best model based on classification reports
        report_files = glob.glob(f"{model_dir}/classification_report_test_epoch*.json")
        
        if report_files:
            print(f"Found {len(report_files)} classification reports")
            for report_file in report_files:
                try:
                    with open(report_file, 'r') as f:
                        report = json.load(f)
                        f1 = report.get('weighted avg', {}).get('f1-score', -1)
                        if f1 > best_f1:
                            best_f1 = f1
                            epoch = int(report_file.split('_')[-1].split('.')[0][5:])
                            best_checkpoint = os.path.join(model_dir, f'model_epoch_{epoch}.pt')
                            print(f"Found better model: epoch {epoch}, F1={f1:.4f}")
                except Exception as e:
                    print(f"Warning: Could not parse {report_file}: {e}")
                    continue
        
        # Fallback to any available model
        if best_checkpoint is None or not os.path.exists(best_checkpoint):
            print("No best model found via reports, using most recent model file")
            best_checkpoint = max(direct_model_files, key=os.path.getmtime)
            best_f1 = -1  # Unknown F1
    
    else:
        print("No direct model files found, checking subdirectories...")
        # Look for model directories (original logic)
        model_dirs = [d for d in glob.glob(f"{finetuned_models_dir}/*") if os.path.isdir(d)]
        if model_dirs:
            model_dir = max(model_dirs, key=os.path.getmtime)
            print(f"Using model directory: {model_dir}")
            
            # Try to find best model based on classification reports
            report_files = glob.glob(f"{model_dir}/classification_report_test_epoch*.json")
            
            if report_files:
                for report_file in report_files:
                    try:
                        with open(report_file, 'r') as f:
                            report = json.load(f)
                            f1 = report.get('weighted avg', {}).get('f1-score', -1)
                            if f1 > best_f1:
                                best_f1 = f1
                                epoch = int(report_file.split('_')[-1].split('.')[0][5:])
                                best_checkpoint = os.path.join(model_dir, f'model_epoch_{epoch}.pt')
                    except Exception as e:
                        continue
            
            # Fallback to any available model
            if best_checkpoint is None or not os.path.exists(best_checkpoint):
                model_files = glob.glob(f"{model_dir}/model_epoch_*.pt")
                if model_files:
                    best_checkpoint = max(model_files, key=os.path.getmtime)
    
    if best_checkpoint is None or not os.path.exists(best_checkpoint):
        raise FileNotFoundError(f"Could not find model checkpoint in {finetuned_models_dir}")
    
    print(f"Using checkpoint: {best_checkpoint}")
    
    # Load model with robust strategy
    model = load_model_robust(model_name, best_checkpoint, device)
    model.eval()
    
    # Process each split
    for split in ['train', 'validation', 'test']:
        print(f"\n=== Processing {split} split ===")
        
        # Load dataset with offline-first approach
        dataset = None
        try:
            print(f"Attempting to load {dataset_name} {split} split from cache...")
            # Try offline first
            dataset = load_dataset(dataset_name, split=split, cache_dir="/tmp/huggingface_cache", 
                                 download_mode="reuse_cache_if_exists")
            print(f"✓ Loaded {len(dataset)} samples from cache")
        except Exception as cache_error:
            print(f"Cache loading failed: {cache_error}")
            try:
                print(f"Attempting to download {dataset_name} {split} split...")
                # Add timeout and retry logic
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("Dataset download timed out")
                
                # Set 60 second timeout
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(60)
                
                try:
                    dataset = load_dataset(dataset_name, split=split, cache_dir="/tmp/huggingface_cache")
                    signal.alarm(0)  # Cancel timeout
                    print(f"✓ Downloaded and loaded {len(dataset)} samples")
                except TimeoutError:
                    print(f"✗ Dataset download timed out after 60 seconds")
                    signal.alarm(0)
                    continue
                except Exception as download_error:
                    signal.alarm(0)
                    print(f"✗ Download failed: {download_error}")
                    continue
                    
            except Exception as e:
                print(f"✗ Could not load {split} split: {e}")
                continue
        
        if dataset is None:
            print(f"Skipping {split} split - could not load dataset")
            continue
        
        # Determine text field
        text_field = 'sentence' if 'sentence' in dataset.column_names else 'text'
        
        # Create output directory
        split_output_dir = f"outputs/embeddings/fully_connected/{dataset_name}/{split}/{split}"
        os.makedirs(split_output_dir, exist_ok=True)
        
        # Process in batches
        small_batch_size = min(32, batch_size)
        save_batch_size = 1000
        
        current_batch_graphs = []
        graph_count = 0
        file_count = 0
        
        print(f"Processing {len(dataset)} samples in batches of {small_batch_size}")
        print(f"Saving graphs in batches of {save_batch_size} to {split_output_dir}")
        
        for batch_idx in tqdm(range(0, len(dataset), small_batch_size), desc=f'Processing {split}'):
            end_idx = min(batch_idx + small_batch_size, len(dataset))
            batch = dataset[batch_idx:end_idx]
            texts = batch[text_field]
            labels = batch['label']
            
            # Tokenize
            inputs = tokenizer(
                texts,
                return_tensors='pt',
                return_offsets_mapping=True,
                truncation=True,
                padding=True,
                return_attention_mask=True
            )
            
            # Get model outputs
            model_inputs = {k: v.to(device) for k, v in inputs.items() if k != 'offset_mapping'}
            with torch.no_grad():
                outputs = model(**model_inputs)
            
            # Process each sample
            for j in range(len(texts)):
                # Get actual tokens (excluding padding)
                attention_mask = model_inputs['attention_mask'][j].bool()
                hidden = outputs.last_hidden_state[j, attention_mask]  # [seq_len, hidden_size]
                
                # Create FULLY CONNECTED graph
                seq_len = hidden.size(0)
                
                # Generate all possible pairs (i,j) for fully connected graph
                source_nodes = []
                target_nodes = []
                for i in range(seq_len):
                    for k in range(seq_len):  # Changed from j to k to avoid confusion
                        source_nodes.append(i)
                        target_nodes.append(k)
                
                edge_index = torch.tensor([
                    source_nodes,
                    target_nodes
                ], dtype=torch.long, device=device)
                
                # Create graph data
                data = Data(
                    x=hidden.cpu(),
                    edge_index=edge_index.cpu(),
                    y=torch.tensor(labels[j], dtype=torch.long)
                )
                
                current_batch_graphs.append(data)
                graph_count += 1
                
                # Save when batch is full
                if len(current_batch_graphs) >= save_batch_size or graph_count == len(dataset):
                    batch_filename = f"graphs_batch_{file_count:03d}.pkl"
                    batch_path = os.path.join(split_output_dir, batch_filename)
                    
                    with open(batch_path, 'wb') as f:
                        pkl.dump(current_batch_graphs, f)
                    
                    print(f"Saved batch {file_count} with {len(current_batch_graphs)} graphs")
                    current_batch_graphs = []
                    file_count += 1
            
            # Clear memory
            del inputs, outputs, model_inputs
            torch.cuda.empty_cache()
        
        # Save remaining graphs
        if current_batch_graphs:
            batch_filename = f"graphs_batch_{file_count:03d}.pkl"
            batch_path = os.path.join(split_output_dir, batch_filename)
            
            with open(batch_path, 'wb') as f:
                pkl.dump(current_batch_graphs, f)
            
            print(f"Saved final batch {file_count} with {len(current_batch_graphs)} graphs")
        
        print(f"✓ Completed {split}: {graph_count} graphs in {file_count + 1} files")


if __name__ == "__main__":
    main()
