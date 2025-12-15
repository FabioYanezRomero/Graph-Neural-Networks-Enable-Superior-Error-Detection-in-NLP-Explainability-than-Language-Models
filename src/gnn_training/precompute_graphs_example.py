"""precompute_graphs_example.py

Example script showing how to precompute k-NN and window graphs for faster training.

This script demonstrates the complete workflow:
1. Precompute k-NN sparsified graphs
2. Precompute window-connected graphs  
3. Test the precomputed datasets

Usage:
    python precompute_graphs_example.py
"""

import os
import subprocess
import argparse


def run_command(cmd, description):
    """Run a command and print its output."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {cmd}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ SUCCESS")
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
    else:
        print("❌ ERROR")
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
    
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='Precompute graphs for faster training')
    parser.add_argument('--input_dir', type=str, 
                       default="outputs/embeddings/fully_connected/stanfordnlp/sst2/train/train",
                       help='Directory containing fully connected graph batch files')
    parser.add_argument('--output_base', type=str,
                       default="outputs/embeddings/precomputed",
                       help='Base directory for precomputed graphs')
    parser.add_argument('--k', type=int, default=8,
                       help='Number of nearest neighbors for k-NN sparsification')
    parser.add_argument('--window_size', type=int, default=5,
                       help='Window size for sliding window connections')
    parser.add_argument('--max_files', type=int, default=5,
                       help='Maximum number of batch files to process (for testing)')
    
    args = parser.parse_args()
    
    # Define output directories
    knn_output_dir = os.path.join(args.output_base, f"knn_k{args.k}", "train")
    window_output_dir = os.path.join(args.output_base, f"window_w{args.window_size}", "train")
    
    print(f"Input directory: {args.input_dir}")
    print(f"k-NN output directory: {knn_output_dir}")
    print(f"Window output directory: {window_output_dir}")
    print(f"k-NN parameter: k={args.k}")
    print(f"Window parameter: window_size={args.window_size}")
    print(f"Max files to process: {args.max_files}")
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"❌ Error: Input directory does not exist: {args.input_dir}")
        return
    
    # 1. Precompute k-NN sparsified graphs
    knn_cmd = f"""python -m src.gnn_training.precompute_knn_graphs \
        --input_dir "{args.input_dir}" \
        --output_dir "{knn_output_dir}" \
        --k {args.k} \
        --max_files {args.max_files}"""
    
    success = run_command(knn_cmd, "Precomputing k-NN sparsified graphs")
    if not success:
        print("❌ Failed to precompute k-NN graphs")
        return
    
    # 2. Precompute window-connected graphs
    window_cmd = f"""python -m src.gnn_training.precompute_window_graphs \
        --input_dir "{args.input_dir}" \
        --output_dir "{window_output_dir}" \
        --window_size {args.window_size} \
        --max_files {args.max_files}"""
    
    success = run_command(window_cmd, "Precomputing window-connected graphs")
    if not success:
        print("❌ Failed to precompute window graphs")
        return
    
    # 3. Test the precomputed datasets
    print(f"\n{'='*60}")
    print("TESTING PRECOMPUTED DATASETS")
    print(f"{'='*60}")
    
    # Test k-NN dataset
    knn_test_cmd = f"""python -m src.gnn_training.precomputed_dataset \
        --data_dir "{knn_output_dir}" \
        --max_files 1"""
    
    success = run_command(knn_test_cmd, "Testing k-NN precomputed dataset")
    
    # Test window dataset
    window_test_cmd = f"""python -m src.gnn_training.precomputed_dataset \
        --data_dir "{window_output_dir}" \
        --max_files 1"""
    
    success = run_command(window_test_cmd, "Testing window precomputed dataset")
    
    # Print usage instructions
    print(f"\n{'='*60}")
    print("PRECOMPUTATION COMPLETED!")
    print(f"{'='*60}")
    print(f"""
✅ k-NN graphs (k={args.k}) saved to: {knn_output_dir}
✅ Window graphs (window_size={args.window_size}) saved to: {window_output_dir}

📖 USAGE INSTRUCTIONS:

1. For k-NN training, modify training_knn.py to use precomputed graphs:
   
   from precomputed_dataset import load_precomputed_data
   
   # Replace FastGraphDataset with:
   train_dataset, train_loader = load_precomputed_data(
       "{knn_output_dir}",
       batch_size=128,
       shuffle=True
   )

2. For window training, modify training_window.py to use precomputed graphs:
   
   from precomputed_dataset import load_precomputed_data
   
   # Replace WindowGraphDataset with:
   train_dataset, train_loader = load_precomputed_data(
       "{window_output_dir}",
       batch_size=128,
       shuffle=True
   )

3. Training will now be MUCH faster since graphs are precomputed!

🚀 PERFORMANCE BENEFITS:
- No on-the-fly k-NN computation (eliminates 18+ seconds per batch)
- No on-the-fly window computation
- Direct loading of precomputed graphs
- Expected speedup: 10-100x faster training
""")


if __name__ == "__main__":
    main()
