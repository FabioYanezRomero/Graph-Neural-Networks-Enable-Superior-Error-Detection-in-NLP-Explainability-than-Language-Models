"""
Test script to verify the GNN training pipeline works with a small subset of data.
"""
import os
import torch
import argparse
import glob
import random
from training import GNNTrainer, SimpleGraphDataset

def create_small_dataset(source_dir, target_dir, max_samples=100):
    """Create a small dataset by copying a subset of files from source to target."""
    os.makedirs(target_dir, exist_ok=True)
    
    # Get all .pkl files from source directory
    source_files = glob.glob(os.path.join(source_dir, "*.pkl"))
    
    # Sort to ensure consistent ordering
    source_files.sort()
    
    # Take just a few files for testing
    selected_files = source_files[:2]  # Just use the first 2 batch files (up to 2000 samples)
    
    # Copy files to target directory
    for src_file in selected_files:
        # Create symlink instead of copying to save space
        dst_file = os.path.join(target_dir, os.path.basename(src_file))
        if not os.path.exists(dst_file):
            os.symlink(src_file, dst_file)
    
    print(f"Created small dataset with {len(selected_files)} batch files in {target_dir}")
    return target_dir

def test_with_subset():
    # Create temporary directories for the small dataset
    base_temp_dir = "/tmp/gnn_test_subset"
    train_temp_dir = os.path.join(base_temp_dir, "train")
    val_temp_dir = os.path.join(base_temp_dir, "validation")
    
    # Create small datasets
    print("Creating small training dataset...")
    train_source = "outputs/embeddings/knn8/stanfordnlp/sst2/train/train"
    create_small_dataset(train_source, train_temp_dir, max_samples=100)
    
    print("\nCreating small validation dataset...")
    val_source = "outputs/embeddings/knn8/stanfordnlp/sst2/validation/validation"
    create_small_dataset(val_source, val_temp_dir, max_samples=20)
    
    # Configuration for testing with a small subset
    config = {
        # Data paths - using our small test datasets
        'train_data_dir': train_temp_dir,
        'val_data_dir': val_temp_dir,
        'test_data_dir': "",  # Skip test data for now
        
        # Model architecture
        'module': 'GCNConv',
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.5,
        'layer_norm': True,
        'residual': True,
        'pooling': 'mean',
        'heads': 1,
        
        # Training settings
        'batch_size': 4,  # Very small batch size for testing
        'epochs': 2,      # Just 2 epochs for testing
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'optimizer': 'Adam',
        'scheduler': 'ReduceLROnPlateau',
        'patience': 1,    # Early stopping patience
        'num_workers': 1, # Minimal workers for testing
        'gradient_accumulation_steps': 1,
        'cache_size': 1,  # Minimal cache for testing
    }
    
    print("="*50)
    print("STARTING TEST WITH SUBSET OF DATA")
    print("="*50)
    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("="*50 + "\n")
    
    try:
        # Initialize trainer
        print("\nInitializing trainer...")
        trainer = GNNTrainer(config)
        
        # Run training
        print("\nStarting training...")
        output_dir = trainer.train()
        
        # If we get here, training completed successfully
        print(f"\nTest completed successfully! Results saved in: {output_dir}")
        
        # Perform a quick validation
        if hasattr(trainer, 'val_loader') and trainer.val_loader is not None:
            print("\nRunning final validation...")
            val_loss, val_acc = trainer.validate()
            print(f"Final validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_with_subset()
    exit(0 if success else 1)
