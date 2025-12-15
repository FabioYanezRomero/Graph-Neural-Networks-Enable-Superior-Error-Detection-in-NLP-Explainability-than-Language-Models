#!/usr/bin/env python3
"""
Main script for fine-tuning language models.
This module provides functionality to fine-tune transformer-based language models
on classification tasks using PyTorch and Hugging Face transformers.
"""

import os
import json
import logging
import argparse
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
import shutil
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_constant_schedule
)
from torch.optim import AdamW
from datasets import load_dataset
from sklearn.metrics import classification_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Default configuration
_BASE_OUT = os.environ.get('GRAPHTEXT_OUTPUT_DIR', 'outputs')
DEFAULT_CONFIG = {
    'num_epochs': 5,
    'learning_rate': 1e-6,
    'weight_decay': 1e-4,
    'adam_epsilon': 1e-8,
    'batch_size': 16,
    'model_name': 'google-bert/bert-base-uncased',
    'dataset_name': 'snli',
    'max_length': 128,
    'fp16': True,
    'lr_scheduler': 'linear',
    'warmup_steps': 0,
    'warmup_proportion': 0.1,
    'output_dir': os.path.join(_BASE_OUT, 'llm'),
    'cuda': True,
    'seed': 42
}

def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def convert_to_native(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer)):
        return int(obj)
    elif isinstance(obj, (np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_native(value) for key, value in obj.items()}
    else:
        return obj

def load_and_process_dataset(dataset_name, tokenizer, max_length):
    """Load and tokenize dataset."""
    logger.info(f"Loading dataset: {dataset_name}")
    
    # Load dataset from Hugging Face datasets
    dataset = load_dataset(dataset_name)
    
    # For SST2, use validation set for testing since test set has no labels
    if dataset_name == 'stanfordnlp/sst2':
        dataset['test'] = dataset['validation']
    
    # Define tokenization function
    def tokenize_function(examples):
        # For SNLI dataset
        if dataset_name == 'snli':
            return tokenizer(
                examples['premise'],
                examples['hypothesis'],
                padding='max_length',
                truncation=True,
                max_length=max_length
            )
        # For SST2 dataset
        elif dataset_name == 'stanfordnlp/sst2':
            return tokenizer(
                examples['sentence'],
                padding='max_length',
                truncation=True,
                max_length=max_length
            )
        # For AG News dataset
        elif dataset_name == 'setfit/ag_news':
            return tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=max_length
            )
        # For other datasets, try common column names
        else:
            # Try to find the text column
            text_column = None
            for column in ['text', 'sentence', 'content']:
                if column in examples:
                    text_column = column
                    break
            
            if text_column is None:
                raise ValueError(f"Could not find text column in dataset. Available columns: {list(examples.keys())}")
                
            return tokenizer(
                examples[text_column],
                padding='max_length',
                truncation=True,
                max_length=max_length
            )
    
    # Add indices to track examples
    def add_indices(examples, indices):
        examples['index'] = indices
        return examples
    
    # Tokenize datasets
    tokenized_datasets = {}
    for split in dataset.keys():
        # Add indices
        dataset[split] = dataset[split].map(
            add_indices,
            with_indices=True,
            batched=True
        )
        
        # Determine columns to remove
        if dataset_name == 'snli':
            columns_to_remove = ['premise', 'hypothesis']
        elif dataset_name == 'stanfordnlp/sst2':
            columns_to_remove = ['sentence', 'idx']
        elif dataset_name == 'setfit/ag_news':
            columns_to_remove = ['text']
        else:
            # Get all columns except label and index
            columns_to_remove = [col for col in dataset[split].column_names if col not in ['label', 'index']]
        
        # Tokenize
        tokenized_datasets[split] = dataset[split].map(
            tokenize_function,
            batched=True,
            remove_columns=columns_to_remove
        )
        
        # Format for PyTorch
        tokenized_datasets[split].set_format(
            type='torch',
            columns=['input_ids', 'attention_mask', 'label', 'index']
        )
    
    return tokenized_datasets

def create_dataloaders(tokenized_datasets, batch_size):
    """Create PyTorch DataLoaders from tokenized datasets."""
    dataloaders = {}
    
    for split, dataset in tokenized_datasets.items():
        dataloaders[split] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train')
        )
    
    return dataloaders

def reporting(y_true, y_pred, epoch, output_dir, split):
    """Generate and save classification report."""
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Print report
    print(f"{split.upper()} CLASSIFICATION REPORT (EPOCH {epoch}):")
    print(classification_report(y_true, y_pred))
    
    # Save report
    report_path = os.path.join(output_dir, f"classification_report_{split}_epoch{epoch}.json")
    with open(report_path, 'w') as fp:
        json.dump(report, fp)
    
    return report

def select_scheduler(optimizer, scheduler_type, num_training_steps, warmup_steps=0):
    """Select learning rate scheduler."""
    if scheduler_type == 'linear':
        return get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=num_training_steps
        )
    elif scheduler_type == 'constant':
        return get_constant_schedule(optimizer)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

def fine_tune(config):
    """Main fine-tuning function."""
    # Set random seed
    set_seed(config['seed'])
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_dir = os.path.join(config['output_dir'], f"{config['dataset_name']}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(results_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and config['cuda'] else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load tokenizer and model
    logger.info(f"Loading model: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    # Set the number of labels based on the dataset
    if config['dataset_name'] == 'setfit/ag_news':
        num_labels = 4  # AG News has 4 classes
    else:
        num_labels = 2
    
    model = AutoModelForSequenceClassification.from_pretrained(
        config['model_name'],
        num_labels=num_labels
    )
    model.to(device)
    
    # Load and process dataset
    tokenized_datasets = load_and_process_dataset(
        config['dataset_name'],
        tokenizer,
        config['max_length']
    )
    
    # Create dataloaders
    dataloaders = create_dataloaders(tokenized_datasets, config['batch_size'])
    
    # Prepare optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        eps=config['adam_epsilon'],
        weight_decay=config['weight_decay']
    )
    
    # Prepare scheduler
    total_steps = len(dataloaders['train']) * config['num_epochs']
    warmup_steps = int(total_steps * config['warmup_proportion']) if config['warmup_steps'] == 0 else config['warmup_steps']
    scheduler = select_scheduler(optimizer, config['lr_scheduler'], total_steps, warmup_steps)
    
    # Initialize FP16 training if enabled
    scaler = GradScaler() if config['fp16'] else None
    
    # Initialize tracking variables
    all_predictions = []
    aggregated_losses = {
        'train': [],
        'validation': [],
        'test': []
    }
    
    # Training loop
    for epoch in range(config['num_epochs']):
        logger.info(f"Starting epoch {epoch+1}/{config['num_epochs']}")
        
        # Training
        model.train()
        train_losses = []
        y_true, y_pred, train_data_references = [], [], []
        
        for batch in tqdm(dataloaders['train'], desc=f"Training epoch {epoch+1}"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            indices = batch['index'].tolist()
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with FP16 if enabled
            if config['fp16']:
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
            
            # Update scheduler
            scheduler.step()
            
            # Track metrics
            train_losses.append(loss.item())
            y_true.extend(labels.cpu().numpy())
            preds = torch.argmax(outputs.logits, axis=1).cpu().numpy()
            y_pred.extend(preds)
            train_data_references.extend(indices)
        
        # Save training predictions
        for true_label, pred_label, data_ref in zip(y_true, y_pred, train_data_references):
            all_predictions.append({
                'epoch': epoch,
                'dataset': 'train',
                'true_label': int(true_label),
                'predicted_label': int(pred_label),
                'data_index': data_ref
            })
        
        # Generate report for training
        reporting(y_true, y_pred, epoch, results_dir, 'train')
        
        # Save training losses
        aggregated_losses['train'].extend(train_losses)
        
        # Validation
        if 'validation' in dataloaders:
            model.eval()
            y_true, y_pred, dev_data_references = [], [], []
            val_losses = []
            
            # Validation loop
            with torch.no_grad():
                for batch in tqdm(dataloaders['validation'], desc=f"Validating epoch {epoch+1}"):
                    input_ids = batch['input_ids'].to(device, non_blocking=True)
                    attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                    labels = batch['label'].to(device, non_blocking=True)
                    indices = batch['index'].tolist()
                    
                    # Forward pass with FP16
                    if config['fp16']:
                        with autocast(device_type='cuda', dtype=torch.bfloat16):
                            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                            loss = outputs.loss
                    else:
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss
                    
                    val_losses.append(loss.item())
                    y_true.extend(labels.cpu().numpy())
                    preds = torch.argmax(outputs.logits, axis=1).cpu().numpy()
                    y_pred.extend(preds)
                    dev_data_references.extend(indices)
            
            # Save validation predictions
            for true_label, pred_label, data_ref in zip(y_true, y_pred, dev_data_references):
                all_predictions.append({
                    'epoch': epoch,
                    'dataset': 'validation',
                    'true_label': int(true_label),
                    'predicted_label': int(pred_label),
                    'data_index': data_ref
                })
            
            # Generate report for validation
            reporting(y_true, y_pred, epoch, results_dir, 'validation')
            
            # Save validation losses
            aggregated_losses['validation'].extend(val_losses)
        
        # Testing
        if 'test' in dataloaders:
            model.eval()
            y_true, y_pred, test_data_references = [], [], []
            test_losses = []
            
            # Testing loop
            with torch.no_grad():
                for batch in tqdm(dataloaders['test'], desc=f"Testing epoch {epoch+1}"):
                    input_ids = batch['input_ids'].to(device, non_blocking=True)
                    attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                    labels = batch['label'].to(device, non_blocking=True)
                    indices = batch['index'].tolist()
                    
                    # Forward pass with FP16
                    if config['fp16']:
                        with autocast(device_type='cuda', dtype=torch.bfloat16):
                            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                            loss = outputs.loss
                    else:
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss
                    
                    test_losses.append(loss.item())
                    y_true.extend(labels.cpu().numpy())
                    preds = torch.argmax(outputs.logits, axis=1).cpu().numpy()
                    y_pred.extend(preds)
                    test_data_references.extend(indices)
            
            # Save test predictions
            for true_label, pred_label, data_ref in zip(y_true, y_pred, test_data_references):
                all_predictions.append({
                    'epoch': epoch,
                    'dataset': 'test',
                    'true_label': int(true_label),
                    'predicted_label': int(pred_label),
                    'data_index': data_ref
                })
            
            # Generate report for testing
            reporting(y_true, y_pred, epoch, results_dir, 'test')
            
            # Save test losses
            aggregated_losses['test'].extend(test_losses)
        
        # Save model checkpoint
        model_path = os.path.join(results_dir, f'model_epoch_{epoch}.pt')
        torch.save(model.state_dict(), model_path)
        
        # Save aggregated losses
        losses_path = os.path.join(results_dir, f'aggregated_losses.json')
        with open(losses_path, 'w') as fp:
            json.dump(aggregated_losses, fp)
        
        # Save all predictions
        predictions_path = os.path.join(results_dir, f'predictions.json')
        with open(predictions_path, 'w') as f:
            json.dump(convert_to_native(all_predictions), f)
    
    # Save final model
    model_path = os.path.join(results_dir, f'model_final.pt')
    torch.save(model.state_dict(), model_path)
    
    logger.info(f"Fine-tuning completed. Results saved to {results_dir}")
    return results_dir

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Fine-tune a language model for classification tasks')
    
    # Required arguments
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name (e.g., snli)')
    
    # Optional arguments with defaults from DEFAULT_CONFIG
    parser.add_argument('--model_name', type=str, default=DEFAULT_CONFIG['model_name'], help='Model name or path')
    parser.add_argument('--num_epochs', type=int, default=DEFAULT_CONFIG['num_epochs'], help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_CONFIG['batch_size'], help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=DEFAULT_CONFIG['learning_rate'], help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=DEFAULT_CONFIG['weight_decay'], help='Weight decay')
    parser.add_argument('--max_length', type=int, default=DEFAULT_CONFIG['max_length'], help='Maximum sequence length')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_CONFIG['output_dir'], help='Output directory')
    parser.add_argument('--seed', type=int, default=DEFAULT_CONFIG['seed'], help='Random seed')
    parser.add_argument('--fp16', action='store_true', default=DEFAULT_CONFIG['fp16'], help='Use mixed precision training')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA even if available')
    parser.add_argument('--lr_scheduler', type=str, default=DEFAULT_CONFIG['lr_scheduler'], choices=['linear', 'constant'], help='Learning rate scheduler')
    parser.add_argument('--warmup_steps', type=int, default=DEFAULT_CONFIG['warmup_steps'], help='Warmup steps (0 to use warmup_proportion)')
    parser.add_argument('--warmup_proportion', type=float, default=DEFAULT_CONFIG['warmup_proportion'], help='Proportion of training steps for warmup')
    
    return parser.parse_args()

def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Create config from arguments
    config = {
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'adam_epsilon': DEFAULT_CONFIG['adam_epsilon'],
        'batch_size': args.batch_size,
        'model_name': args.model_name,
        'dataset_name': args.dataset_name,
        'max_length': args.max_length,
        'fp16': args.fp16,
        'lr_scheduler': args.lr_scheduler,
        'warmup_steps': args.warmup_steps,
        'warmup_proportion': args.warmup_proportion,
        'output_dir': args.output_dir,
        'cuda': not args.no_cuda and torch.cuda.is_available(),
        'seed': args.seed
    }
    
    # Run fine-tuning
    fine_tune(config)

if __name__ == "__main__":
    main()
