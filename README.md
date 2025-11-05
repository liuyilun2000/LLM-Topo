# Topological Analysis Pipeline for Neural Language Models

This pipeline trains language models on graph walk sequences and analyzes their internal representations using topological data analysis (TDA). The pipeline generates graph structures, trains models, extracts representations, and performs persistent homology analysis.

## Overview

The pipeline consists of four main stages:

1. **Stage 01: Data Generation** - Generate graph structures and random walk sequences
2. **Stage 02: Model Training** - Train language models on walk sequences and extract representations
3. **Stage 03: Dimensionality Reduction** - Apply PCA, fuzzy neighborhood, and UMAP
4. **Stage 04: Topology Analysis** - Compute persistent homology and generate persistence diagrams/barcodes

## Quick Start

### 1. Configure Environment

All scripts use a centralized configuration file `src/00_config_env.sh`. Navigate to the `src/` directory before running scripts:

```bash
cd src/

# Default settings (can be overridden)
export H=30              # Grid height
export W=40              # Grid width
export TOPOLOGY=torus    # Topology type
export RUN_NAME=2M_llama # Model run identifier
```

### 2. Run Complete Pipeline

**Option A: Run Full Pipeline (Recommended)**

Run all stages automatically with default settings:

```bash
cd src/
./00_run_full_pipeline.sh
```

This will execute all stages from 01a to 04b sequentially with default settings.

**Option B: Run Individual Stages**

Run stages individually for more control:

```bash
cd src/

# Stage 01: Generate graph and sequences
./01a_graph_generation.sh
./01b_sequence_generation.sh
./01c_dataset_preparation.sh

# Stage 02: Train model and extract representations
./02a_model_training.sh
./02b_representation_extraction.sh

# Stage 03: Dimensionality reduction
./03a_pca_analysis.sh
./03b_fuzzy_neighborhood.sh
./03c_umap_analysis.sh

# Stage 04: Topology analysis
./04a_topology_analysis.sh
./04b_persistence_barcode.sh
```

## Configuration

### Centralized Configuration

All pipeline scripts source `src/00_config_env.sh` which provides:
- `H` (grid height, default: 30)
- `W` (grid width, default: 40)
- `TOPOLOGY` (topology type, default: "torus")
- `DATASET_NAME` (auto-generated from topology and dimensions)
- `RUN_NAME` (model run identifier, default: "2M_llama")

### Available Topologies

- `plane` - Flat plane (no wrapping)
- `cylinder_x`, `cylinder_y` - Cylinder wrapping in x/y direction
- `mobius_x`, `mobius_y` - Möbius strip wrapping in x/y direction
- `torus` - Torus (wraps in both directions)
- `klein_x`, `klein_y` - Klein bottle wrapping in x/y direction
- `proj_plane` - Projective plane
- `sphere_two` - Sphere with two hemispheres
- `hemisphere_n`, `hemisphere_s` - Northern/Southern hemisphere
- `sphere` - Sphere (single point)

### Custom Configuration

Override settings using environment variables (run from `src/` directory):

```bash
cd src/

# Single override
H=100 ./01a_graph_generation.sh

# Multiple overrides
H=100 W=100 TOPOLOGY="sphere" ./01a_graph_generation.sh

# Set for entire session
export H=60 W=90 TOPOLOGY="torus" RUN_NAME="experiment_1"
./01a_graph_generation.sh
./01b_sequence_generation.sh
./02a_model_training.sh
```

## Directory Structure

```
LLM-Topo/
├── src/
│   ├── 00_config_env.sh              # Centralized configuration
│   ├── 00_run_full_pipeline.sh      # Master script: Run all stages
│   ├── 01a_graph_generation.sh        # Stage 01: Graph generation
│   ├── 01b_sequence_generation.sh     # Stage 01: Random walk generation
│   ├── 01c_dataset_preparation.sh    # Stage 01: Dataset preparation
│   ├── 02a_model_training.sh          # Stage 02: Model training
│   ├── 02b_representation_extraction.sh  # Stage 02: Extract representations
│   ├── 03a_pca_analysis.sh           # Stage 03: PCA analysis
│   ├── 03b_fuzzy_neighborhood.sh      # Stage 03: Fuzzy neighborhood
│   ├── 03c_umap_analysis.sh           # Stage 03: UMAP analysis
│   ├── 04a_topology_analysis.sh       # Stage 04: Persistent homology
│   ├── 04b_persistence_barcode.sh    # Stage 04: Barcode visualization
│   │
│   ├── scripts/                       # Python scripts
│   │   ├── 01a_graph_generation.py
│   │   ├── 01b_sequence_generation.py
│   │   ├── 01c_dataset_preparation.py
│   │   ├── 02a_model_training.py
│   │   ├── 02b_representation_extraction.py
│   │   ├── 03a_pca_analysis.py
│   │   ├── 03b_fuzzy_neighborhood.py
│   │   ├── 03c_umap_analysis.py
│   │   ├── 04a_topology_analysis.py
│   │   ├── 04b_persistence_barcode.py
│   │   ├── quotient_space_topology.py
│   │   └── utils.py
│   │
│   ├── configs/                      # Model configurations
│   │   ├── config_400K_llama.json
│   │   ├── config_2M_llama.json
│   │   ├── config_6M_llama.json
│   │   └── config_12M_llama.json
│   │
│   ├── results/                      # All outputs (data and analysis)
│   │   └── {DATASET_NAME}/
│   │       ├── graph/                 # Graph structures (Stage 01)
│   │       ├── sequences/             # Random walk sequences (Stage 01)
│   │       ├── dataset/               # Prepared datasets (Stage 01)
│   │       └── {RUN_NAME}/            # Run-specific outputs
│   │           ├── final_model/           # Trained models (Stage 02)
│   │           ├── token_representations/ # Extracted representations (Stage 02)
│   │           ├── pca_result/           # PCA results (Stage 03)
│   │           ├── fuzzy_neighborhood/    # Fuzzy distance matrices (Stage 03)
│   │           ├── umap_result_*/        # UMAP results (Stage 03)
│   │           ├── topology_analysis/     # Persistence diagrams (Stage 04)
│   │           └── persistence_barcode/   # Barcode visualizations (Stage 04)
│   │
│   ├── README_01.md                   # Stage 01 documentation
│   ├── README_02.md                   # Stage 02 documentation
│   ├── README_03.md                   # Stage 03 documentation
│   └── README_04.md                   # Stage 04 documentation
│
└── README.md                          # Main documentation (this file)
```

## Workflow Summary

### Stage 01: Data Generation
- Generate graph adjacency matrices and coordinates
- Generate random walk sequences
- Prepare HuggingFace datasets

**See:** [src/README_01.md](src/README_01.md)

### Stage 02: Model Training
- Train Llama models on walk sequences
- Extract internal representations (hidden states, FFN activations)

**See:** [src/README_02.md](src/README_02.md)

### Stage 03: Dimensionality Reduction
- Apply PCA to reduce dimensionality
- Compute fuzzy neighborhood distance matrices
- Apply UMAP for visualization

**See:** [src/README_03.md](src/README_03.md)

### Stage 04: Topology Analysis
- Compute persistent homology using ripser
- Generate persistence diagrams
- Visualize persistence barcodes

**See:** [src/README_04.md](src/README_04.md)

## Model Configurations

Available model presets in `configs/`:

| Config | Parameters | Hidden | Layers | Heads | FFN |
|--------|-----------|--------|--------|-------|-----|
| config_400K_llama.json | ~400K | 64 | 6 | 4 | 256 |
| config_2M_llama.json | ~2M | 128 | 8 | 8 | 512 |
| config_6M_llama.json | ~6M | 256 | 6 | 8 | 1024 |
| config_12M_llama.json | ~12M | 384 | 8 | 8 | 1536 |

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `transformers` - HuggingFace transformers
- `datasets` - HuggingFace datasets
- `numpy`, `scipy` - Numerical computing
- `umap-learn` - UMAP dimensionality reduction
- `ripser` - Persistent homology computation
- `persim` - Persistence diagram visualization
- `matplotlib`, `plotly` - Visualization

## Output Files

All outputs are stored in `results/{DATASET_NAME}/` structure:

### Stage 01 Outputs
**Location:** `results/{DATASET_NAME}/graph/` and `results/{DATASET_NAME}/sequences/`
- `graph/A_{topology}_{H}x{W}.npy` - Adjacency matrix
- `graph/nodes_{topology}_{H}x{W}.csv` - Node information
- `graph/coords_{topology}_{H}x{W}.npy` - 3D coordinates
- `graph/distance_matrix_{topology}_{H}x{W}.npy` - Shortest path distances
- `sequences/walks_{topology}_{H}x{W}.csv` - Random walk sequences
- `dataset/` - Prepared HuggingFace datasets

### Stage 02 Outputs
**Location:** `results/{DATASET_NAME}/{RUN_NAME}/`
- `final_model/` - Trained model checkpoints
- `token_representations/token_representations.npz` - Extracted representations

### Stage 03 Outputs
**Location:** `results/{DATASET_NAME}/{RUN_NAME}/`
- `pca_result/{key}_pca.npz` - PCA-reduced data
- `fuzzy_neighborhood/{key}_fuzzy_dist.npz` - Fuzzy distance matrices
- `umap_result_{n}d/{key}_umap_{n}d.npz` - UMAP embeddings

### Stage 04 Outputs
**Location:** `results/{DATASET_NAME}/{RUN_NAME}/`
- `topology_analysis/{key}_persistence_diagram.png` - Persistence diagrams
- `topology_analysis/{key}_topology.json` - Persistence diagram data
- `persistence_barcode/{key}_barcode.png` - Persistence barcodes

## Troubleshooting

### Model Not Found
```bash
cd src/

# Check model exists
ls results/{DATASET_NAME}/{RUN_NAME}/final_model/

# Re-train if needed
./02a_model_training.sh
```

### Memory Issues
```bash
cd src/

# Use smaller model
CONFIG=configs/config_400K_llama.json ./02a_model_training.sh

# Reduce batch size
BATCH_SIZE=4 ./02a_model_training.sh
```

### Missing Dependencies
```bash
# Install ripser for topology analysis
pip install ripser

# Install UMAP
pip install umap-learn

# Install plotly for interactive visualizations
pip install plotly
```

## References

- **HuggingFace Transformers**: https://huggingface.co/docs/transformers
- **UMAP**: https://umap-learn.readthedocs.io/
- **Ripser**: https://ripser.scikit-tda.org/
- **Persim**: https://persim.scikit-tda.org/

## Documentation

- [src/README_01.md](src/README_01.md) - Data generation (graphs, sequences, datasets)
- [src/README_02.md](src/README_02.md) - Model training and representation extraction
- [src/README_03.md](src/README_03.md) - Dimensionality reduction (PCA, fuzzy neighborhood, UMAP)
- [src/README_04.md](src/README_04.md) - Topology analysis (persistent homology, barcodes)
