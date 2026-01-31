#!/bin/bash

# Visualize UMAP embedding from graph matrix with trajectory overlay
#
# This script:
# 1. Loads a distance/adjacency matrix from the graph directory
# 2. Applies UMAP to get 3D embeddings
# 3. Loads a trajectory from walks CSV
# 4. Creates 3D visualization (point cloud + trajectory overlay)
# 5. Saves as HTML and metadata JSON
#
# Uses 01d_graph_umap_visualize.py (same as 01e for analysis mode).
# For UMAP 6D + topology on graph: ./01e → ./01f → ./01g

set -e

echo "=========================================="
echo "Graph UMAP Visualization (3D + trajectory)"
echo "=========================================="

# Load shared configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/00_config_env.sh"

WALKS_CSV="${WALKS_CSV:-./${DATA_DIR}/sequences/walks_${DATASET_NAME}.csv}"
MATRIX_TYPE="${MATRIX_TYPE:-auto}"
WALK_ID="${WALK_ID:-}"
TRAJECTORY_IDX="${TRAJECTORY_IDX:-128}"
UMAP_N_COMPONENTS="${UMAP_N_COMPONENTS:-3}"
UMAP_MIN_DIST="${UMAP_MIN_DIST:-0.2}"
UMAP_N_NEIGHBORS="${UMAP_N_NEIGHBORS:-200}"
UMAP_RANDOM_STATE="${UMAP_RANDOM_STATE:-42}"
MAX_LENGTH="${MAX_LENGTH:-128}"
OUTPUT_DIR="${OUTPUT_DIR:-./${DATA_DIR}/graph_umap_visualize}"

echo ""
echo "Configuration:"
echo "  Graph dir: $GRAPH_DIR"
echo "  Dataset: $DATASET_NAME"
echo "  Matrix type: $MATRIX_TYPE"
echo "  Walks CSV: $WALKS_CSV"
[ -n "$WALK_ID" ] && echo "  Walk ID: $WALK_ID" || echo "  Trajectory index: $TRAJECTORY_IDX"
echo "  Output dir: $OUTPUT_DIR"
echo "  UMAP: ${UMAP_N_COMPONENTS}D, min_dist=$UMAP_MIN_DIST, n_neighbors=$UMAP_N_NEIGHBORS"
echo ""

# Check graph directory
if [ ! -d "$GRAPH_DIR" ]; then
    echo "Error: Graph directory not found: $GRAPH_DIR"
    echo "Please run ./01a_graph_generation.sh first"
    exit 1
fi

# Check matrix files
DISTANCE_FILE="${GRAPH_DIR}/distance_matrix_${DATASET_NAME}.npy"
ADJACENCY_FILE="${GRAPH_DIR}/A_${DATASET_NAME}.npy"
if [ "$MATRIX_TYPE" = "auto" ]; then
    [ ! -f "$DISTANCE_FILE" ] && [ ! -f "$ADJACENCY_FILE" ] && {
        echo "Error: Neither distance nor adjacency matrix found"
        exit 1
    }
elif [ "$MATRIX_TYPE" = "distance" ] && [ ! -f "$DISTANCE_FILE" ]; then
    echo "Error: Distance matrix not found: $DISTANCE_FILE"
    exit 1
elif [ "$MATRIX_TYPE" = "adjacency" ] && [ ! -f "$ADJACENCY_FILE" ]; then
    echo "Error: Adjacency matrix not found: $ADJACENCY_FILE"
    exit 1
fi

# Check walks CSV
[ ! -f "$WALKS_CSV" ] && {
    echo "Error: Walks CSV not found: $WALKS_CSV"
    exit 1
}

echo "Generating visualization..."
PYTHON_ARGS=(
    --graph_dir "$GRAPH_DIR"
    --dataset_name "$DATASET_NAME"
    --matrix_type "$MATRIX_TYPE"
    --walks_csv "$WALKS_CSV"
    --output_dir "$OUTPUT_DIR"
    --umap_n_components "$UMAP_N_COMPONENTS"
    --umap_min_dist "$UMAP_MIN_DIST"
    --umap_n_neighbors "$UMAP_N_NEIGHBORS"
    --umap_random_state "$UMAP_RANDOM_STATE"
    --max_length "$MAX_LENGTH"
    --generate_visualizations
)
[ -n "$WALK_ID" ] && PYTHON_ARGS+=(--walk_id "$WALK_ID") || PYTHON_ARGS+=(--trajectory_idx "$TRAJECTORY_IDX")

python ../scripts/01d_graph_umap_visualize.py "${PYTHON_ARGS[@]}"

echo ""
echo "For UMAP 6D + topology analysis on graph: ./01e → ./01f → ./01g"
echo ""
