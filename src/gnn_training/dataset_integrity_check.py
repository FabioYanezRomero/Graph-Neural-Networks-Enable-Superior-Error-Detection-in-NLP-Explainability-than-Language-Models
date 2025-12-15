import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))  # Add project root to path

from torch.utils.data import DataLoader
from src.gnn_training.training import CachedGraphDataset

print("Loading dataset...")
ds = CachedGraphDataset("outputs/embeddings/knn8/stanfordnlp/sst2/train/train")
print(f"Successfully loaded dataset with {len(ds)} samples")

print("\nChecking dataset integrity...")
for i in range(len(ds)):
    try:
        _ = ds[i]
        if (i + 1) % 1000 == 0:
            print(f"Checked {i+1} samples...")
    except Exception as e:
        print(f"\nError at index {i}: {e}")
        break
else:
    print("\nNo errors found in the dataset.")
