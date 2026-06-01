# HyperSVD

HyperSVD is a smart contract vulnerability detection framework based on heterogeneous contract graphs and semantic hypergraphs. It represents each contract as a graph, constructs vulnerability-aware hyperedges, and trains a hypergraph neural network for contract-level binary vulnerability classification.

## Overview

The code supports four vulnerability datasets:

- `Reentrancy`
- `BlockInfoDep`
- `NestedCall`
- `TranStaDep`

The default model is `hgnn`, which combines:

1. **Hypergraph convolution** for message passing over semantic hyperedges.
2. **Subgraph/hyperedge attention** for highlighting important hyperedges.
3. **Contract-level readout** for contract-level vulnerability prediction.

The implementation also includes profiling utilities for runtime, memory, and scalability analysis.

## Repository Structure

```text
.
├── SMVulDetector.py              # Main training and evaluation entry point
├── parser.py                     # Command-line argument parser
├── load_data.py                  # Dataset loading, batching, and cross-validation splits
├── models/
│   └── hgnn_model.py             # Hypergraph neural network model
├── tools/
│   ├── hyperedge_builder.py      # Hyperedge construction utilities
│   └── ExpRecorder.py            # Experiment logging and checkpointing
├── training_data/                # Dataset directory, not included in this package
│   ├── Reentrancy/
│   ├── BlockInfoDep/
│   ├── NestedCall/
│   └── TranStaDep/
└── outputs/                      # Generated logs, checkpoints, ROC files, and profiles
```

## Requirements

The implementation is based on Python and PyTorch. A typical environment includes:

```text
python >= 3.8
torch
numpy
scikit-learn
matplotlib
psutil
```

Install the required packages with:

```bash
pip install torch numpy scikit-learn matplotlib psutil
```

Install PyTorch

Install PyTorch according to your hardware platform by following the official PyTorch installation instructions.

For CPU-only execution, select the CPU build in the official PyTorch installer.

For GPU execution, select a CUDA build compatible with your NVIDIA driver and hardware.

After installation, verify the environment:
```python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"```

If there is something wrong when you install cuad-vision PyTorch, you can use parameter --device with cpu in parser.py. 
## Dataset Format

Datasets are expected to follow a TU-style graph classification format under:

```text
training_data/<DATASET_NAME>/
```

For example:

```text
training_data/Reentrancy/
├── Reentrancy_A.txt
├── Reentrancy_graph_indicator.txt
├── Reentrancy_graph_labels.txt
├── Reentrancy_node_labels.txt
└── Reentrancy_node_attributes.txt
```

The loader automatically searches for the above keywords in file names.

## Quick Start

Run HyperSVD on the Reentrancy dataset:

```bash
python SMVulDetector.py \
  --dataset Reentrancy \
  --model hgnn \
  --epochs 50 \
  --batch_size 16 \
  --device cuda
```

Available datasets:

```bash
--dataset Reentrancy
--dataset BlockInfoDep
--dataset NestedCall
--dataset TranStaDep
```

## Important Arguments

```text
--lr                  Learning rate.
--wd                  Weight decay.
--dropout             Dropout rate.
--filters             Hidden dimensions for graph convolution baselines.
--n_hidden            Hidden dimension for the fully connected layer.
--epochs              Number of training epochs.
--batch_size          Mini-batch size.
--folds               Number of folds for cross-validation.
--alpha_hg            Balance between hypergraph convolution and subgraph attention.
--k_call_ctx          Number of outgoing call-context neighbors.
--d_struct            Structural neighborhood size for hyperedge construction.
--exp_tag             Short tag used in output file names.
```

## Ablation Options

HyperSVD provides branch-level and hyperedge-level ablations.

Disable the subgraph attention branch:

```bash
python SMVulDetector.py --dataset Reentrancy --model hgnn --disable_subattn
```

Disable the hypergraph convolution branch:

```bash
python SMVulDetector.py --dataset Reentrancy --model hgnn --disable_hgconv
```

Remove one hyperedge family:

```bash
python SMVulDetector.py --dataset Reentrancy --model hgnn --no_struct_he
python SMVulDetector.py --dataset Reentrancy --model hgnn --no_coperm_he
python SMVulDetector.py --dataset Reentrancy --model hgnn --no_callctx_he
```

## Profiling

Enable efficiency and scalability profiling:

```bash
python SMVulDetector.py \
  --dataset Reentrancy \
  --model hgnn \
  --profile \
  --exp_tag profile_reentrancy
```

Profiling outputs are written to:

```text
outputs/profile/<DATASET>/<TAG>/
```

The profiling CSV files include per-fold summaries and per-graph records such as preprocessing time, inference time, graph size, hyperedge count, GPU memory, and RAM usage.

## Outputs

Training outputs are saved under:

```text
outputs/runs/<DATASET>/<MODEL>/<TAG>/
```

Typical generated files include:

```text
models/              # best model checkpoints
checkpoints/         # additional checkpoint files
logs/history.csv     # epoch-level metrics
logs/history.json    # full metric history
meta.json            # experiment configuration
```

ROC data can be exported with:

```bash
--save_npz auto
```

This saves ROC arrays to:

```text
outputs/roc/<DATASET>/<TAG>.npz
```

## Notes

- The hyperedge builder supports structural, co-permission, and call-context hyperedges.
- When explicit privilege annotations are unavailable, co-permission hyperedges are approximated using simple structural signatures.
- The implementation uses cross-validation splits generated from the dataset graph ids.
- For reproducibility, set `--seed` explicitly.

## Citation


## License

