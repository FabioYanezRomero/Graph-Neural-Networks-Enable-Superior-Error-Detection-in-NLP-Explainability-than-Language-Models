#!/usr/bin/env python3
"""
Script to process semantic analysis shards incrementally to avoid OOM issues.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

def create_single_entry_config(original_config_path: Path, entry_index: int, shard_index: int | None, output_path: Path):
    """Create a config file with only one subgraphx entry, optionally with only one shard."""
    with open(original_config_path, 'r') as f:
        config = json.load(f)

    if entry_index >= len(config.get('subgraphx', [])):
        return False

    entry = config['subgraphx'][entry_index].copy()

    # If shard_index is specified, keep only that shard
    if shard_index is not None and shard_index < len(entry.get('paths', [])):
        entry['paths'] = [entry['paths'][shard_index]]

    # Keep only the specified entry
    config['subgraphx'] = [entry]

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    return True

def main():
    if len(sys.argv) != 2:
        print("Usage: python process_shards_incrementally.py <config.json>")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    # Load config to count entries
    with open(config_path, 'r') as f:
        config = json.load(f)

    subgraphx_entries = config.get('subgraphx', [])
    print(f"Found {len(subgraphx_entries)} subgraphx entries to process")

    for i, entry in enumerate(subgraphx_entries):
        dataset = entry.get('dataset', 'unknown')
        graph_type = entry.get('graph_type', 'unknown')
        num_shards = len(entry.get('paths', []))

        print(f"\nProcessing entry {i+1}/{len(subgraphx_entries)}: {dataset}:{graph_type} ({num_shards} shards)")

        if num_shards > 1:
            # Process each shard separately for multi-shard entries
            for shard_idx in range(num_shards):
                print(f"  Processing shard {shard_idx+1}/{num_shards}")

                temp_config = config_path.parent / f"temp_config_entry_{i}_shard_{shard_idx}.json"
                if not create_single_entry_config(config_path, i, shard_idx, temp_config):
                    print(f"Failed to create config for entry {i}, shard {shard_idx}")
                    continue

                try:
                    cmd = [
                        sys.executable, "-m", "src.Analytics.semantic.cli_pipeline",
                        "--config", str(temp_config)
                    ]

                    print(f"Running: {' '.join(cmd)}")
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

                    if result.returncode == 0:
                        print(f"Successfully processed shard {shard_idx+1}")
                    else:
                        print(f"Failed to process shard {shard_idx+1}")
                        print("STDOUT:", result.stdout[-500:])  # Last 500 chars
                        print("STDERR:", result.stderr[-500:])

                except subprocess.TimeoutExpired:
                    print(f"Timeout processing shard {shard_idx+1}")
                except Exception as e:
                    print(f"Error processing shard {shard_idx+1}: {e}")
                finally:
                    if temp_config.exists():
                        temp_config.unlink()
                    time.sleep(5)  # Short wait between shards

            # After processing all shards, merge them
            print(f"  Merging results for {dataset}:{graph_type}")
            try:
                temp_config = config_path.parent / f"temp_config_entry_{i}_merge.json"
                if create_single_entry_config(config_path, i, None, temp_config):  # None means all shards
                    cmd = [
                        sys.executable, "-m", "src.Analytics.semantic.cli_pipeline",
                        "--config", str(temp_config)
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                    if result.returncode == 0:
                        print(f"Successfully merged entry {i}")
                    else:
                        print(f"Failed to merge entry {i}")
                if temp_config.exists():
                    temp_config.unlink()
            except Exception as e:
                print(f"Error merging entry {i}: {e}")

        else:
            # Single shard entry - process normally
            temp_config = config_path.parent / f"temp_config_entry_{i}.json"
            if not create_single_entry_config(config_path, i, None, temp_config):
                print(f"Failed to create config for entry {i}")
                continue

            try:
                cmd = [
                    sys.executable, "-m", "src.Analytics.semantic.cli_pipeline",
                    "--config", str(temp_config)
                ]

                print(f"Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

                if result.returncode == 0:
                    print(f"Successfully processed entry {i}")
                else:
                    print(f"Failed to process entry {i}")
                    print("STDOUT:", result.stdout[-1000:])
                    print("STDERR:", result.stderr[-1000:])

            except subprocess.TimeoutExpired:
                print(f"Timeout processing entry {i}")
            except Exception as e:
                print(f"Error processing entry {i}: {e}")
            finally:
                if temp_config.exists():
                    temp_config.unlink()

        # Wait between entries to allow memory cleanup
        print("Waiting 10 seconds for memory cleanup...")
        time.sleep(10)

    print("\nAll entries processed!")

if __name__ == "__main__":
    main()
