# Stage 01: Data Generation

This stage generates graph structures and random walk sequences for training language models.

## Overview

Stage 01 consists of three steps:
1. **Graph Generation** (`01a`) - Generate graph adjacency matrices and coordinates
2. **Sequence Generation** (`01b`) - Generate random walk sequences on the graph
3. **Dataset Preparation** (`01c`) - Convert sequences to HuggingFace dataset format

## Step 1: Graph Generation

Generate graph representation for a given topology.

### Script: `01a_graph_generation.sh`

```bash
./01a_graph_generation.sh
```

### Configuration

```bash
# Override defaults
H=30 W=40 TOPOLOGY=torus NEIGH=8 ./01a_graph_generation.sh
```

**Parameters:**
- `H` - Grid height (default: 30)
- `W` - Grid width (default: 40)
- `TOPOLOGY` - Topology type (default: "torus")
- `NEIGH` - Neighborhood type: 4 (von Neumann) or 8 (Moore) (default: 8)

### Output Files

Generated in `results/{DATASET_NAME}/graph/`:

- `A_{topology}_{H}x{W}_labeled.csv` - Adjacency matrix with node labels
- `A_{topology}_{H}x{W}.npy` - Adjacency matrix as numpy array
- `nodes_{topology}_{H}x{W}.csv` - Node information (node_id, rowcol_index, i, j, layer)
- `coords_{topology}_{H}x{W}.csv` - 3D coordinates for visualization
- `coords_{topology}_{H}x{W}.npy` - Coordinates as numpy array
- `distance_matrix_{topology}_{H}x{W}.npy` - Shortest path distance matrix
- `graph_info_{topology}_{H}x{W}.json` - Graph metadata

### Example

```bash
# Generate torus graph
H=30 W=40 TOPOLOGY=torus ./01a_graph_generation.sh

# Generate multiple topologies
H=30 W=40 TOPOLOGY=torus,sphere,klein_x ./01a_graph_generation.sh

# Generate all topologies
H=30 W=40 TOPOLOGY=all ./01a_graph_generation.sh
```

## Step 2: Sequence Generation

Generate random walk sequences on the pre-computed graph.

### Script: `01b_sequence_generation.sh`

```bash
./01b_sequence_generation.sh
```

### Configuration

```bash
# Custom walk parameters
MAX_LENGTH=128 MAX_SEQS=120000 NO_REPEAT_WINDOW=32 ./01b_sequence_generation.sh
```

**Parameters:**
- `MAX_LENGTH` - Maximum walk length (default: 128)
- `MAX_SEQS` - Maximum number of sequences (default: 120000)
- `MIN_VISITS_PER_NODE` - Minimum visits per node (default: 10000000000)
- `NO_REPEAT_WINDOW` - Window size for no-repeat constraint (default: 32)
- `RESTART_PROB` - Restart probability (default: 0)
- `TEMPERATURE` - Temperature for neighbor selection (default: 1)
- `SEED` - Random seed (default: 42)

### Walk Algorithm

1. Start at a random node
2. At each step, choose from neighbors (excluding previous node)
3. Avoid repeating nodes within a sliding window
4. Continue until max length or restart condition

### Output Files

Generated in `results/{DATASET_NAME}/sequences/`:

- `walks_{topology}_{H}x{W}.csv` - Random walk sequences
  - Columns: `walk_id`, `length`, `sequence_labels`
  - Format: space-separated node IDs
- `visit_counts_{topology}_{H}x{W}.csv` - Node visit statistics

### Example

```bash
# Generate walks with custom parameters
MAX_LENGTH=256 MAX_SEQS=50000 NO_REPEAT_WINDOW=64 ./01b_sequence_generation.sh
```

## Step 3: Dataset Preparation

Convert CSV walk sequences to HuggingFace dataset format.

### Script: `01c_dataset_preparation.sh`

```bash
./01c_dataset_preparation.sh
```

### Configuration

```bash
# Custom train split
TRAIN_SPLIT=0.9 ./01c_dataset_preparation.sh
```

**Parameters:**
- `INPUT_CSV` - Input CSV file path (default: `results/{DATASET_NAME}/sequences/walks_{DATASET_NAME}.csv`)
- `DATASET_DIR` - Output dataset directory (default: `results/{DATASET_NAME}/dataset`)
- `TRAIN_SPLIT` - Fraction of data for training (default: 0.95)

### Output Files

Generated in `results/{DATASET_NAME}/dataset/`:

- `train/` - Training split (HuggingFace dataset)
- `validation/` - Validation split (HuggingFace dataset)
- `vocab_info.json` - Vocabulary statistics
  - `vocab_size` - Number of unique tokens
  - `min_id`, `max_id` - Token ID range

### Dataset Format

Each example contains:
- `walk_id` - Unique walk identifier
- `length` - Sequence length
- `input_ids` - List of token IDs

### Example

```bash
# Prepare dataset with custom split
TRAIN_SPLIT=0.9 ./01c_dataset_preparation.sh
```

## Complete Stage 01 Workflow

```bash
# 1. Generate graph
H=30 W=40 TOPOLOGY=torus ./01a_graph_generation.sh

# 2. Generate sequences
MAX_LENGTH=128 MAX_SEQS=120000 ./01b_sequence_generation.sh

# 3. Prepare dataset
TRAIN_SPLIT=0.95 ./01c_dataset_preparation.sh
```

## Output Structure

```
results/{DATASET_NAME}/
├── graph/
│   ├── A_torus_30x40_labeled.csv
│   ├── A_torus_30x40.npy
│   ├── nodes_torus_30x40.csv
│   ├── coords_torus_30x40.csv
│   ├── distance_matrix_torus_30x40.npy
│   └── graph_info_torus_30x40.json
│
├── sequences/
│   ├── walks_torus_30x40.csv
│   └── visit_counts_torus_30x40.csv
│
└── dataset/
    ├── train/
    ├── validation/
    └── vocab_info.json
```

## Next Steps

After Stage 01, proceed to:
- [README_02.md](README_02.md) - Train model and extract representations

