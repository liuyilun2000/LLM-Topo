# Stage 03: Dimensionality Reduction

This stage applies dimensionality reduction techniques to token representations for visualization and topology analysis.

## Overview

Stage 03 consists of four steps:
1. **PCA Analysis** (`03a`) - Principal Component Analysis
2. **Fuzzy Neighborhood** (`03b`) - Compute fuzzy neighborhood distance matrices
3. **UMAP Visualization** (`03c`) - UMAP with 3D default for visualization
4. **UMAP Analysis** (`03d`) - UMAP with 6D default for topology analysis

## Step 1: PCA Analysis

Apply Principal Component Analysis to reduce dimensionality while preserving variance.

### Script: `03a_pca_analysis.sh`

```bash
./03a_pca_analysis.sh
```

### Configuration

```bash
# Use variance threshold (recommended)
PCA_VARIANCE=0.95 ./03a_pca_analysis.sh

# Or use fixed number of components
PCA_N_COMPONENTS=50 ./03a_pca_analysis.sh
```

**Parameters:**
- `REPRESENTATION_DIR` - Input directory (default: `results/{DATASET_NAME}/{RUN_NAME}/token_representations`)
- `OUTPUT_DIR` - Output directory (default: `results/{DATASET_NAME}/{RUN_NAME}/pca_result`)
- `PCA_VARIANCE` - Variance threshold (default: 0.95)
- `PCA_N_COMPONENTS` - Fixed number of components (alternative to variance)

### PCA Method

1. **Standardize** data (zero mean, unit variance)
2. **Compute** principal components
3. **Select** components based on variance threshold or fixed count
4. **Transform** data to reduced space

### Output Files

Generated in `results/{DATASET_NAME}/{RUN_NAME}/pca_result/`:

- `{key}_pca.npz` - PCA-reduced data
  - `pca_reduced`: Reduced embeddings `[n_points, n_components]`
  - `scaler`: StandardScaler model
  - `pca_model`: PCA model
- `{key}_pca_model.pkl` - Saved PCA model and scaler
- `{key}_pca_info.json` - PCA information
  - `n_components`: Number of components
  - `explained_variance_ratio`: Variance explained per component
  - `cumulative_variance`: Cumulative variance explained

### Example

```bash
# Retain 95% variance
PCA_VARIANCE=0.95 ./03a_pca_analysis.sh

# Use 50 components
PCA_N_COMPONENTS=50 ./03a_pca_analysis.sh
```

## Step 2: Fuzzy Neighborhood

Compute fuzzy neighborhood distance matrices for persistent homology analysis.

### Script: `03b_fuzzy_neighborhood.sh`

```bash
./03b_fuzzy_neighborhood.sh
```

### Configuration

```bash
# Adjust neighborhood size
N_NEIGHBORS=100 METRIC=cosine ./03b_fuzzy_neighborhood.sh
```

**Parameters:**
- `PCA_DIR` - Input PCA directory (default: `results/{DATASET_NAME}/{RUN_NAME}/pca_result`)
- `OUTPUT_DIR` - Output directory (default: `results/{DATASET_NAME}/{RUN_NAME}/fuzzy_neighborhood`)
- `N_NEIGHBORS` - Number of nearest neighbors (default: 20)
- `METRIC` - Distance metric: `euclidean`, `cosine`, `manhattan` (default: `cosine`)
- `SYM_METHOD` - Symmetrization method: `fuzzy_union`, `average`, `max`, `min` (default: `fuzzy_union`)
- `EPSILON` - Numerical stability constant (default: 1e-10)
- `SPARSITY_THRESHOLD` - Sparsification threshold (default: 0.0)
- `TARGET_ENTROPY` - Target entropy (`auto` uses log2(n_neighbors)) (default: `auto`)

### Fuzzy Neighborhood Algorithm

1. **Compute base distances** - Pairwise distances (cosine, euclidean, etc.)
2. **Find k-nearest neighbors** - For each point, find k nearest neighbors
3. **Compute local scales** - Adaptive bandwidth per point
4. **Compute membership strengths** - Fuzzy membership based on distances
5. **Symmetrize** - Make membership matrix symmetric
6. **Convert to distance** - Distance = -log(membership)

### Effect of N_NEIGHBORS

- **Small k (20-50)**: More local structure, sparse connections
- **Medium k (100-400)**: Balanced local/global structure
- **Large k (800-2000)**: More global structure, dense connections

### Output Files

Generated in `results/{DATASET_NAME}/{RUN_NAME}/fuzzy_neighborhood/`:

- `{key}_fuzzy_dist.npz` - Fuzzy distance matrix
  - `distance_matrix`: Distance matrix `[n_points, n_points]`
  - `pca_reduced`: Original PCA data (for reference)
- `{key}_fuzzy_info.json` - Computation information
  - `n_neighbors`: Number of neighbors used
  - `metric`: Distance metric
  - `sym_method`: Symmetrization method

### Example

```bash
# Local structure (small k)
N_NEIGHBORS=50 METRIC=cosine ./03b_fuzzy_neighborhood.sh

# Global structure (large k)
N_NEIGHBORS=800 METRIC=cosine ./03b_fuzzy_neighborhood.sh

# Euclidean distance
N_NEIGHBORS=200 METRIC=euclidean ./03b_fuzzy_neighborhood.sh
```

## Step 3: UMAP Visualization

Apply UMAP for 3D visualization (2D/3D scatter plots).

### Script: `03c_umap_visualize.sh`

```bash
./03c_umap_visualize.sh
```

### Configuration

```bash
# Use fuzzy distance matrices (default: 3D with visualizations)
USE_FUZZY=true ./03c_umap_visualize.sh

# Use regular PCA
USE_FUZZY=false UMAP_N_COMPONENTS=3 ./03c_umap_visualize.sh
```

**Parameters:**
- `USE_FUZZY` - Use fuzzy distance matrices (default: true)
- `FUZZY_DIR` - Fuzzy neighborhood directory (default: `results/{DATASET_NAME}/{RUN_NAME}/fuzzy_neighborhood`)
- `PCA_DIR` - PCA directory (default: `results/{DATASET_NAME}/{RUN_NAME}/pca_result`)
- `REPRESENTATION_DIR` - Representation directory (default: `results/{DATASET_NAME}/{RUN_NAME}/token_representations`)
- `OUTPUT_DIR` - Output directory (default: `results/{DATASET_NAME}/{RUN_NAME}/umap_result_{n}d`)
- `UMAP_N_COMPONENTS` - Target dimensions (default: 3)
- `UMAP_MIN_DIST` - Minimum distance (default: 0.2)
- `UMAP_N_NEIGHBORS` - Number of neighbors (default: 20)
- `UMAP_METRIC` - Distance metric (default: cosine)
- `SAVE_UMAP_RESULT` - Save UMAP embeddings (default: true)
- `GENERATE_VISUALIZATIONS` - Generate plots (default: true)

## Step 4: UMAP Analysis

Apply UMAP for higher-dimensional embeddings (6D default) for topology analysis.

### Script: `03d_umap_analysis.sh`

```bash
./03d_umap_analysis.sh
```

### Configuration

```bash
# Use fuzzy distance matrices (recommended, 6D for topology)
USE_FUZZY=true UMAP_N_COMPONENTS=6 ./03d_umap_analysis.sh

# Use regular PCA
USE_FUZZY=false UMAP_N_COMPONENTS=6 ./03d_umap_analysis.sh
```

**Parameters:**
- `USE_FUZZY` - Use fuzzy distance matrices (default: true)
- `FUZZY_DIR` - Fuzzy neighborhood directory (default: `results/{DATASET_NAME}/{RUN_NAME}/fuzzy_neighborhood`)
- `PCA_DIR` - PCA directory (default: `results/{DATASET_NAME}/{RUN_NAME}/pca_result`)
- `REPRESENTATION_DIR` - Representation directory (default: `results/{DATASET_NAME}/{RUN_NAME}/token_representations`)
- `OUTPUT_DIR` - Output directory (default: `results/{DATASET_NAME}/{RUN_NAME}/umap_result_{n}d`)
- `UMAP_N_COMPONENTS` - Target dimensions (default: 6)
- `UMAP_MIN_DIST` - Minimum distance (default: 0.2)
- `UMAP_N_NEIGHBORS` - Number of neighbors (default: 20)
- `UMAP_METRIC` - Distance metric (default: cosine)
- `SAVE_UMAP_RESULT` - Save UMAP embeddings (default: true)
- `GENERATE_VISUALIZATIONS` - Generate plots (default: false)

### UMAP Parameters

**n_components:**
- `2` or `3` - For visualization
- `6` or higher - For topology analysis (preserves more structure)

**min_dist:**
- `0.0` - Tight clusters, more local structure
- `0.2` - Balanced (default)
- `1.0` - Spread out, more global structure

**n_neighbors:**
- Small (10-20) - More local structure
- Medium (20-50) - Balanced
- Large (50-100) - More global structure

### Output Files

Generated in `results/{DATASET_NAME}/{RUN_NAME}/umap_result_{n}d/`:

- `{key}_umap_{n}d.npz` - UMAP embeddings
  - `umap_reduced`: UMAP embeddings `[n_points, n_components]`
- `{key}_umap_{n}d.png` - 2D/3D scatter plots (if visualization enabled)
- `{key}_umap_{n}d.html` - Interactive 3D plots (if visualization enabled)

### Example

```bash
# 3D visualization (03c)
./03c_umap_visualize.sh

# 6D for topology analysis (03d)
./03d_umap_analysis.sh

# Regular PCA (no fuzzy)
USE_FUZZY=false UMAP_N_COMPONENTS=3 ./03c_umap_visualize.sh
USE_FUZZY=false UMAP_N_COMPONENTS=6 ./03d_umap_analysis.sh
```

## Complete Stage 03 Workflow

```bash
# 1. PCA analysis
PCA_VARIANCE=0.95 ./03a_pca_analysis.sh

# 2. Fuzzy neighborhood
N_NEIGHBORS=200 METRIC=cosine ./03b_fuzzy_neighborhood.sh

# 3. UMAP visualization (3D)
./03c_umap_visualize.sh

# 4. UMAP analysis (6D for topology)
USE_FUZZY=true UMAP_N_COMPONENTS=6 ./03d_umap_analysis.sh
```

## Output Structure

```
results/{DATASET_NAME}/{RUN_NAME}/
├── pca_result/
│   ├── {key}_pca.npz
│   ├── {key}_pca_model.pkl
│   └── {key}_pca_info.json
│
├── fuzzy_neighborhood/
│   ├── {key}_fuzzy_dist.npz
│   └── {key}_fuzzy_info.json
│
└── umap_result_6d/
    ├── {key}_umap_6d.npz
    └── {key}_umap_6d.png (if visualization enabled)
```

## Data Flow

```
Token Representations (high-dimensional)
    ↓
PCA Analysis (variance-preserving reduction)
    ↓
Fuzzy Neighborhood (topologically-aware distances)
    ↓
UMAP (manifold-preserving reduction)
    ↓
Low-dimensional embeddings (for visualization/analysis)
```

## Next Steps

After Stage 03, proceed to:
- [README_04.md](README_04.md) - Topology analysis (persistent homology, barcodes)

