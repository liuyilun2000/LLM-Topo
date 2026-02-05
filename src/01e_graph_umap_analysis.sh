#!/bin/bash

# Apply UMAP 6D on graph distance/adjacency matrix for topology analysis
#
# This script mirrors 03d_umap_analysis.sh but operates on the graph matrix
# (distance/adjacency matrix) instead of model representations.
# Output is compatible with 04a_topology_analysis.py (data_representation mode).
#
# Pipeline: 01a (graph) → 01e (UMAP 6D) → 01f (topology) → 01g (barcode)

set -e

echo "=========================================="
echo "Graph UMAP Analysis (6D for topology)"
echo "=========================================="

# Load shared configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/00_config_env.sh"

# MATRIX_TYPE, GRAPH_UMAP_* from 00_config_env.sh (topology pipeline uses GRAPH_UMAP_N_COMPONENTS_TOPOLOGY)
UMAP_N_COMPONENTS="${UMAP_N_COMPONENTS:-${GRAPH_UMAP_N_COMPONENTS_TOPOLOGY}}"
UMAP_MIN_DIST="${UMAP_MIN_DIST:-${GRAPH_UMAP_MIN_DIST}}"
UMAP_N_NEIGHBORS="${UMAP_N_NEIGHBORS:-${GRAPH_UMAP_N_NEIGHBORS}}"
UMAP_RANDOM_STATE="${UMAP_RANDOM_STATE:-${GRAPH_UMAP_RANDOM_STATE}}"
OUTPUT_DIR="${OUTPUT_DIR:-${GRAPH_UMAP_TOPOLOGY_RESULT_DIR}}"

echo ""
echo "Configuration:"
echo "  Graph dir: $GRAPH_DIR"
echo "  Dataset: $DATASET_NAME"
echo "  Matrix type: $MATRIX_TYPE"
echo "  Output dir: $OUTPUT_DIR"
echo "  UMAP target dimensions: ${UMAP_N_COMPONENTS}D"
echo "  UMAP min_dist: $UMAP_MIN_DIST"
echo "  UMAP n_neighbors: $UMAP_N_NEIGHBORS"
echo "  UMAP random_state: $UMAP_RANDOM_STATE"
echo ""

# Check if graph directory exists
if [ ! -d "$GRAPH_DIR" ]; then
    echo "Error: Graph directory not found: $GRAPH_DIR"
    echo "Please run ./01a_graph_generation.sh first"
    exit 1
fi

# Check for matrix files
DISTANCE_FILE="${GRAPH_DIR}/distance_matrix_${DATASET_NAME}.npy"
ADJACENCY_FILE="${GRAPH_DIR}/A_${DATASET_NAME}.npy"
if [ "$MATRIX_TYPE" = "auto" ]; then
    if [ ! -f "$DISTANCE_FILE" ] && [ ! -f "$ADJACENCY_FILE" ]; then
        echo "Error: Neither distance nor adjacency matrix found"
        echo "  Expected: $DISTANCE_FILE"
        echo "  Or: $ADJACENCY_FILE"
        exit 1
    fi
elif [ "$MATRIX_TYPE" = "distance" ] && [ ! -f "$DISTANCE_FILE" ]; then
    echo "Error: Distance matrix not found: $DISTANCE_FILE"
    exit 1
elif [ "$MATRIX_TYPE" = "adjacency" ] && [ ! -f "$ADJACENCY_FILE" ]; then
    echo "Error: Adjacency matrix not found: $ADJACENCY_FILE"
    exit 1
fi

echo "Applying UMAP on graph matrix..."
python ../scripts/01d_graph_umap_visualize.py \
    --save_umap_result \
    --graph_dir "$GRAPH_DIR" \
    --dataset_name "$DATASET_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --matrix_type "$MATRIX_TYPE" \
    --umap_n_components "$UMAP_N_COMPONENTS" \
    --umap_min_dist "$UMAP_MIN_DIST" \
    --umap_n_neighbors "$UMAP_N_NEIGHBORS" \
    --umap_random_state "$UMAP_RANDOM_STATE"

echo ""
echo "=========================================="
echo "Graph UMAP analysis complete!"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Next step: Run topology analysis on graph"
echo "  ./01f_graph_topology_analysis.sh"
echo ""
