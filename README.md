GraphText: Modular NLP→Graph→GNN Pipeline
=========================================

This repo orchestrates a full workflow to build graphs from text, attach LLM embeddings, convert to PyTorch Geometric (PyG), train GNNs, and run explainability. It wraps the current implementation under `src/Clean_Code` with a small, extensible package and CLI for easier scaling and future additions.

What’s New
----------
- `src/graphtext`: a thin, modular layer with registries and a unified CLI
- Intuitive module paths under `src/` (shims): `src/finetuning`, `src/graph_builders`, `src/embeddings`, `src/convert`, `src/gnn_training`, `src/explain`
- Composable subcommands: finetune → build-graphs → embed → to-pyg → train → explain
- Config-driven pipeline runner for end-to-end experiments
- Clear extension points to add new graph-generation and explainability methods
- Consolidated outputs under `outputs/`
- Metadata index at `outputs/metadata/index.json`

Package Layout (new)
--------------------
- `src/graphtext/registry.py`: Simple registries for plug-ins
- `src/graphtext/graphs/`: Graph builders (wrapping existing generators)
- `src/graphtext/embeddings/`: LLM fine-tuning and graph embedding wrappers
- `src/graphtext/convert/`: NetworkX→PyG conversion wrapper
- `src/graphtext/training/`: GNN trainer wrapper
- `src/graphtext/explain/`: Explainers wrapper
- `src/graphtext/cli.py`: Unified CLI
- `src/graphtext/pipeline.py`: Programmatic pipeline runner
Shims to legacy code under intuitive paths:
- `src/finetuning` → `src/Clean_Code/Model_Finetuning`
- `src/graph_builders` → `src/Clean_Code/Tree_Generation`
- `src/embeddings` → `src/Clean_Code/Graph_Embeddings`
- `src/convert` → `src/Clean_Code/NetworkX_to_PyG`
- `src/gnn_training` → `src/Clean_Code/GNN_training`
- `src/explain/subgraphx` → `src/Clean_Code/Optimization`

Quick Start
-----------
Run commands from the project root. Use the `src`-package style:

- Fine-tune an LLM
  - `python -m src.graphtext.cli finetune --dataset_name stanfordnlp/sst2 --output_dir outputs/llm`

- Build graphs (syntactic or constituency)
  - `python -m src.graphtext.cli build-graphs --graph_type syntactic --dataset stanfordnlp/sst2 --output_dir outputs/graphs`

- Embed nodes using an LLM
  - `python -m src.graphtext.cli embed --graph_type syntactic --dataset_name stanfordnlp/sst2 --split validation --tree_dir <trees_dir> --output_dir <emb_out>`

- Convert to PyG with labels
  - `python -m src.graphtext.cli to-pyg --label_source llm --hf_dataset_name stanfordnlp/sst2 --graph_type syntactic`

- Train a GNN
  - `python -m src.graphtext.cli train --train_data_dir <pyg_train_dir> --val_data_dir <pyg_val_dir>`

- Explain a trained GNN (auto-selects SubgraphX for hierarchical graphs)
  - `python -m src.graphtext.cli explain --dataset stanfordnlp/sst2 --graph_type syntactic --split validation --method auto`
  - `python -m src.graphtext.cli explain --dataset ag_news --graph_type skipgrams --backbone SetFit --split test --method auto`
  - Append `--performance_profile fast` to trade a small accuracy drop for significantly faster SubgraphX/GraphSVX sweeps, or `quality` to favour fidelity.
  - Use `--num_jobs 4` (for example) to shard the dataset across four parallel explain processes; each shard writes its own summary under the run directory.
  - Summaries land under each run directory, e.g. `outputs/gnn_models/<backbone>/<dataset>/<graph_type>/<run>/explanations/<method>/<backbone>_<dataset>_<graph_type>_<split>/summary.json`

Pipeline via JSON
-----------------
See `configs/example_pipeline.json` for a complete end-to-end config. Run:

- `python -m src.graphtext.cli run --config configs/example_pipeline.json`

Extending Graph Generation
--------------------------
Add a new builder by registering it:

```
from src.graphtext.registry import GRAPH_BUILDERS
from src.graphtext.graphs.base import BaseGraphBuilder, BuildArgs

@GRAPH_BUILDERS.register("my_builder")
class MyBuilder(BaseGraphBuilder):
    def process_dataset(self, args: BuildArgs) -> None:
        # produce and save graphs under args.output_dir
        ...
```

Notes
-----
- The new layer calls existing scripts/modules under `src/Clean_Code` via shims for now; the codebase will be fully migrated and `src/Clean_Code` removed after outputs are archived.
- Some legacy modules use non-relative imports; the wrappers execute them as modules (`python -m ...`) to ensure imports resolve.
- Default outputs are consolidated under `outputs/`. You can override with `GRAPHTEXT_OUTPUT_DIR`.
- A metadata index is appended after each step at `outputs/metadata/index.json`.

Docker Compose Services
-----------------------
- `app`: Main CUDA-enabled development environment for training and evaluation.
- `graphsvx`: Standalone GraphSVX explainer image cloned from the upstream project.
- `shap`: Lightweight SHAP explainability workspace.
- `tokenshap`: Token-level SHAP container with Hugging Face tooling and GPU support.
- `subgraphx`: DIG-powered SubgraphX explainer image sharing the `/app` volume for direct access to trained GNNs and graphs.

Start any explainer container individually, for example `docker compose up -d subgraphx`, then open an interactive shell with `make subgraphx-shell` to run `python -m src.explain.gnn.subgraphx.main` (or `...graphsvx.main`) or custom scripts.
