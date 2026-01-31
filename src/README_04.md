# Stage 04: Topology Analysis

This stage computes persistent homology and generates persistence diagrams and barcodes for topological analysis.

## Overview

Stage 04 consists of two steps:
1. **Topology Analysis** (`04a`) - Compute persistent homology and generate persistence diagrams
2. **Persistence Barcode** (`04b`) - Visualize persistence barcodes

## Step 1: Topology Analysis

Compute persistent homology using ripser and generate persistence diagrams.

### Script: `04a_topology_analysis.sh`

```bash
./04a_topology_analysis.sh
```

### Configuration

```bash
# Use fuzzy distance matrices (recommended)
INPUT_MODE=distance_matrix FUZZY_DIR=./fuzzy_neighborhood ./04a_topology_analysis.sh

# Use data representations (PCA/UMAP)
INPUT_MODE=data_representation DATA_TYPE=umap ./04a_topology_analysis.sh
```

**Input Modes:**

1. **Distance Matrix Mode** (`INPUT_MODE=distance_matrix`)
   - Uses fuzzy neighborhood distance matrices
   - Better for capturing topological structure
   - Source: `fuzzy_neighborhood/` directory

2. **Data Representation Mode** (`INPUT_MODE=data_representation`)
   - Uses data embeddings (PCA, UMAP, downsampled PCA)
   - Computes distances from point cloud
   - Source: `pca_result/`, `umap_result_*/`, etc.

**Parameters:**
- `INPUT_MODE` - Input mode: `distance_matrix` or `data_representation` (default: `data_representation`)
- `FUZZY_DIR` - Fuzzy neighborhood directory (for distance_matrix mode)
- `REPRESENTATION_DIR` - Data representation directory (for data_representation mode)
- `DATA_TYPE` - Data type: `auto`, `pca`, `umap`, `downsampled` (default: `auto`)
- `OUTPUT_DIR` - Output directory (default: `results/{DATASET_NAME}/{RUN_NAME}/topology_analysis`)
- `RIPSER_THRESH` - Ripser filtration threshold (default: None = full filtration)
- `RIPSER_MAXDIM` - Maximum homology dimension (default: 2, computes H⁰, H¹, H²)
- `RIPSER_COEFF` - Homology coefficients (default: 47, Z/47Z)

### Persistent Homology

Persistent homology tracks topological features across scales:

- **H⁰ (β₀)**: Connected components
- **H¹ (β₁)**: 1-dimensional holes (loops)
- **H² (β₂)**: 2-dimensional holes (voids)

### Output Files

Generated in `results/{DATASET_NAME}/{RUN_NAME}/topology_analysis/`:

- `{key}_persistence_diagram.png` - Persistence diagram visualization
- `{key}_topology.json` - Persistence diagram data
  ```json
  {
    "persistence_diagrams": {
      "H0": [[birth, death], ...],
      "H1": [[birth, death], ...],
      "H2": [[birth, death], ...]
    }
  }
  ```

### Persistence Diagrams

Each persistence diagram shows:
- **H⁰ (red)**: Connected components (diagonal points)
- **H¹ (blue)**: Loops (below diagonal)
- **H² (green)**: Voids (further below diagonal)

Points far from the diagonal indicate persistent topological features.

### Example

```bash
# Using fuzzy distance matrices (recommended)
INPUT_MODE=distance_matrix \
RIPSER_MAXDIM=2 \
RIPSER_COEFF=47 \
./04a_topology_analysis.sh

# Using UMAP embeddings
INPUT_MODE=data_representation \
DATA_TYPE=umap \
REPRESENTATION_DIR=./umap_result_6d \
./04a_topology_analysis.sh

# Using PCA embeddings
INPUT_MODE=data_representation \
DATA_TYPE=pca \
./04a_topology_analysis.sh
```

## Step 2: Persistence Barcode

Generate persistence barcode visualizations from persistence diagrams.

### Script: `04b_persistence_barcode.sh`

```bash
./04b_persistence_barcode.sh
```

### Configuration

```bash
# Custom barcode parameters
MAX_BARS=30 TOP_K=2 ./04b_persistence_barcode.sh
```

**Parameters:**
- `TOPOLOGY_DIR` - Topology analysis directory (default: `results/{DATASET_NAME}/{RUN_NAME}/topology_analysis`)
- `OUTPUT_DIR` - Output directory (default: `results/{DATASET_NAME}/{RUN_NAME}/persistence_barcode`)
- `MAX_BARS` - Maximum bars per dimension (default: 30)
- `TOP_K` - Top-k longest bars globally (default: 2)

### Barcode Visualization

Barcodes show:
- **H⁰ bars**: Connected components (birth to death)
- **H¹ bars**: Loops (birth to death)
- **H² bars**: Voids (birth to death)

Long bars indicate persistent topological features. Top-k longest bars are marked with stars (★).

### Output Files

Generated in `results/{DATASET_NAME}/{RUN_NAME}/persistence_barcode/`:

- `{key}_barcode.png` - Persistence barcode plot (PNG)
- `{key}_barcode.pdf` - Persistence barcode plot (PDF)
- `{key}_statistics.json` - Barcode statistics
  ```json
  {
    "H0": {
      "total_bars": 100,
      "persistent_bars": 5,
      "longest_bar_length": 0.5,
      "top_k_bars": [...]
    },
    ...
  }
  ```

### Example

```bash
# Default settings
./04b_persistence_barcode.sh

# Show more bars
MAX_BARS=50 TOP_K=5 ./04b_persistence_barcode.sh
```

## Complete Stage 04 Workflow

```bash
# 1. Compute persistent homology
INPUT_MODE=distance_matrix \
RIPSER_MAXDIM=2 \
./04a_topology_analysis.sh

# 2. Generate barcodes
MAX_BARS=30 TOP_K=2 \
./04b_persistence_barcode.sh
```

## Output Structure

```
results/{DATASET_NAME}/{RUN_NAME}/
├── topology_analysis/
│   ├── {key}_persistence_diagram.png
│   └── {key}_topology.json
│
└── persistence_barcode/
    ├── {key}_barcode.png
    ├── {key}_barcode.pdf
    └── {key}_statistics.json
```

## Interpreting Results

### Persistence Diagrams

- **Points on diagonal**: Noise (short-lived features)
- **Points far from diagonal**: Persistent features (important topological structure)
- **H¹ points**: Indicates loops in the data manifold
- **H² points**: Indicates voids/cavities

### Persistence Barcodes

- **Long H⁰ bars**: Multiple connected components
- **Long H¹ bars**: Persistent loops (torus-like structure)
- **Long H² bars**: Persistent voids (sphere-like structure)

### Expected Topology

For a torus topology (H¹=2, H²=1):
- **H⁰**: One long bar (single connected component)
- **H¹**: Two long bars (two independent loops)
- **H²**: One long bar (one void)

## Troubleshooting

### Ripser Not Installed

```bash
# Install ripser
pip install ripser

# Install persim for visualization
pip install persim
```

### Memory Issues

```bash
# Use smaller filtration threshold
RIPSER_THRESH=10.0 ./04a_topology_analysis.sh

# Use lower homology dimension
RIPSER_MAXDIM=1 ./04a_topology_analysis.sh
```

### No Persistence Features

If no persistent features are found:
- Check if data has sufficient structure
- Try different distance matrices (adjust fuzzy neighborhood parameters)
- Try different homology dimensions
- Check if ripser threshold is appropriate

## References

- **Ripser**: https://ripser.scikit-tda.org/
- **Persim**: https://persim.scikit-tda.org/
- **Topological Data Analysis**: https://www.math.upenn.edu/~ghrist/preprints/barcodes.pdf

## Complete Pipeline

After completing all stages:

```bash
# Stage 01: Data generation
./01a_graph_generation.sh
./01b_sequence_generation.sh
./01c_dataset_preparation.sh

# Stage 02: Model training
./02a_model_training.sh
./02b_representation_extraction.sh

# Stage 03: Dimensionality reduction
./03a_pca_analysis.sh
./03b_fuzzy_neighborhood.sh
./03c_umap_visualize.sh
./03d_umap_analysis.sh

# Stage 04: Topology analysis
./04a_topology_analysis.sh
./04b_persistence_barcode.sh
```

