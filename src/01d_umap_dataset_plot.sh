#!/bin/bash

# Visualize UMAP embedding directly from dataset neighboring matrix with trajectory overlay
#
# This script:
# 1. Loads a neighboring matrix (adjacency or distance matrix) from the graph directory
# 2. Applies UMAP to get 3D embeddings
# 3. Loads a specified trajectory from walks CSV
# 4. Creates a 3D visualization with:
#    - Point cloud colored with blue-green-yellow gradient
#    - Trajectory overlaid with red-orange-yellow gradient based on time
# 5. Saves as PNG, HTML, and metadata JSON

set -e

echo "=========================================="
echo "UMAP Visualization from Dataset Matrix with Trajectory"
echo "=========================================="

# Load shared configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/00_config_env.sh"

# Variables GRAPH_DIR, DATA_DIR, N, K_EDGE, ITERS, TOPOLOGY_RULE are loaded from 00_config_env.sh

# Walks CSV file - construct dataset name to match sequence generation output
# Use prefix + topology rule: {PREFIX}_{TOPOLOGY_RULE}_n{N}_k{K_EDGE}_iter{ITERS}
DATASET_NAME_PYTHON="${TOPOLOGY_PREFIX}_${TOPOLOGY_RULE}_n${N}_k${K_EDGE}_iter${ITERS}"
WALKS_CSV="${WALKS_CSV:-./${DATA_DIR}/sequences/walks_${DATASET_NAME_PYTHON}.csv}"

# Matrix type: "adjacency" or "distance" (default: distance if available, else adjacency)
MATRIX_TYPE="${MATRIX_TYPE:-auto}"

# Trajectory selection
WALK_ID="${WALK_ID:-}"  # If set, uses this walk_id
TRAJECTORY_IDX="${TRAJECTORY_IDX:-128}"  # If WALK_ID not set, uses this index

# UMAP parameters
UMAP_N_COMPONENTS="${UMAP_N_COMPONENTS:-3}"
UMAP_MIN_DIST="${UMAP_MIN_DIST:-0.2}"
UMAP_N_NEIGHBORS="${UMAP_N_NEIGHBORS:-20}"
UMAP_RANDOM_STATE="${UMAP_RANDOM_STATE:-100}"  # Set to integer for reproducibility, empty for random

# Trajectory parameters
MAX_LENGTH="${MAX_LENGTH:-128}"  # Maximum number of points to plot in trajectory

# Output directory
OUTPUT_DIR="${OUTPUT_DIR:-./${DATA_DIR}/umap_dataset_plot}"

echo ""
echo "Configuration:"
echo "  Graph dir: $GRAPH_DIR"
echo "  Dataset: $DATASET_NAME_PYTHON"
echo "  Matrix type: $MATRIX_TYPE"
echo "  Walks CSV: $WALKS_CSV"
if [ -n "$WALK_ID" ]; then
    echo "  Walk ID: $WALK_ID"
else
    echo "  Trajectory index: $TRAJECTORY_IDX"
fi
echo "  Output dir: $OUTPUT_DIR"
echo "  UMAP target dimensions: ${UMAP_N_COMPONENTS}D"
echo "  UMAP min_dist: $UMAP_MIN_DIST"
echo "  UMAP n_neighbors: $UMAP_N_NEIGHBORS"
if [ -n "$UMAP_RANDOM_STATE" ]; then
    echo "  UMAP random_state: $UMAP_RANDOM_STATE"
else
    echo "  UMAP random_state: (random)"
fi
echo "  Max trajectory length: $MAX_LENGTH points"
echo ""

# Check if graph directory exists
if [ ! -d "$GRAPH_DIR" ]; then
    echo "Error: Graph directory not found: $GRAPH_DIR"
    echo "Please run ./01a_graph_generation.sh first"
    exit 1
fi

# Check for distance matrix first (preferred), then adjacency matrix
DISTANCE_FILE="${GRAPH_DIR}/distance_matrix_${DATASET_NAME_PYTHON}.npy"
ADJACENCY_FILE="${GRAPH_DIR}/A_${DATASET_NAME_PYTHON}.npy"

if [ "$MATRIX_TYPE" = "auto" ]; then
    if [ -f "$DISTANCE_FILE" ]; then
        MATRIX_FILE="$DISTANCE_FILE"
        MATRIX_TYPE="distance"
        echo "Found distance matrix: $DISTANCE_FILE"
    elif [ -f "$ADJACENCY_FILE" ]; then
        MATRIX_FILE="$ADJACENCY_FILE"
        MATRIX_TYPE="adjacency"
        echo "Found adjacency matrix: $ADJACENCY_FILE"
    else
        echo "Error: Neither distance matrix nor adjacency matrix found"
        echo "  Expected: $DISTANCE_FILE"
        echo "  Or: $ADJACENCY_FILE"
        exit 1
    fi
elif [ "$MATRIX_TYPE" = "distance" ]; then
    MATRIX_FILE="$DISTANCE_FILE"
    if [ ! -f "$MATRIX_FILE" ]; then
        echo "Error: Distance matrix not found: $MATRIX_FILE"
        exit 1
    fi
    echo "Using distance matrix: $MATRIX_FILE"
elif [ "$MATRIX_TYPE" = "adjacency" ]; then
    MATRIX_FILE="$ADJACENCY_FILE"
    if [ ! -f "$MATRIX_FILE" ]; then
        echo "Error: Adjacency matrix not found: $MATRIX_FILE"
        exit 1
    fi
    echo "Using adjacency matrix: $MATRIX_FILE"
else
    echo "Error: Invalid MATRIX_TYPE: $MATRIX_TYPE (must be 'auto', 'distance', or 'adjacency')"
    exit 1
fi

# Check if walks CSV exists
if [ ! -f "$WALKS_CSV" ]; then
    echo "Error: Walks CSV file not found: $WALKS_CSV"
    echo "Please ensure walks have been generated first"
    exit 1
fi

echo "Found walks CSV: $WALKS_CSV"
echo ""
echo "Generating visualization..."

# Build and run Python script
if [ -n "$WALK_ID" ]; then
    python ../scripts/01d_umap_dataset_plot.py \
        --graph_dir "$GRAPH_DIR" \
        --dataset_name "$DATASET_NAME_PYTHON" \
        --matrix_type "$MATRIX_TYPE" \
        --walks_csv "$WALKS_CSV" \
        --walk_id "$WALK_ID" \
        --output_dir "$OUTPUT_DIR" \
        --umap_n_components "$UMAP_N_COMPONENTS" \
        --umap_min_dist "$UMAP_MIN_DIST" \
        --umap_n_neighbors "$UMAP_N_NEIGHBORS" \
        $(if [ -n "$UMAP_RANDOM_STATE" ]; then echo "--umap_random_state $UMAP_RANDOM_STATE"; fi) \
        --max_length "$MAX_LENGTH"
else
    python ../scripts/01d_umap_dataset_plot.py \
        --graph_dir "$GRAPH_DIR" \
        --dataset_name "$DATASET_NAME_PYTHON" \
        --matrix_type "$MATRIX_TYPE" \
        --walks_csv "$WALKS_CSV" \
        --trajectory_idx "$TRAJECTORY_IDX" \
        --output_dir "$OUTPUT_DIR" \
        --umap_n_components "$UMAP_N_COMPONENTS" \
        --umap_min_dist "$UMAP_MIN_DIST" \
        --umap_n_neighbors "$UMAP_N_NEIGHBORS" \
        $(if [ -n "$UMAP_RANDOM_STATE" ]; then echo "--umap_random_state $UMAP_RANDOM_STATE"; fi) \
        --max_length "$MAX_LENGTH"
fi
