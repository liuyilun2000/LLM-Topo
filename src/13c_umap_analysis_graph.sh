#!/bin/bash

# Apply UMAP using the original graph distance matrix
#
# This script applies UMAP (Uniform Manifold Approximation and Projection) to
# the original graph distance matrix from the graph dataset. This is used for
# validation - to examine whether the current methodology's working on 
# representations in LLM's results is valid or not by comparing against
# the ground truth graph topology.
#
# The graph distance matrix is loaded from:
#   results/${DATASET_NAME}/graph/distance_matrix_${DATASET_NAME}.npy
# where DATASET_NAME is determined by TOPOLOGY, H, W from 00_config_env.sh

set -e

echo "=========================================="
echo "UMAP Analysis (Graph Ground Truth)"
echo "=========================================="

# Load shared configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/00_config_env.sh"

# Local configuration

# Graph directory (contains the original distance matrix)
GRAPH_DIR="${GRAPH_DIR:-./${DATA_DIR}/graph}"

# UMAP parameters
UMAP_N_COMPONENTS="${UMAP_N_COMPONENTS:-6}"
UMAP_MIN_DIST="${UMAP_MIN_DIST:-0.2}"
UMAP_N_NEIGHBORS="${UMAP_N_NEIGHBORS:-20}"

# Output control flags
SAVE_UMAP_RESULT="${SAVE_UMAP_RESULT:-true}"
GENERATE_VISUALIZATIONS="${GENERATE_VISUALIZATIONS:-false}"

# Output directory (includes dimensionality)
OUTPUT_DIR="${OUTPUT_DIR:-./${WORK_DIR}/umap_result_graph_${UMAP_N_COMPONENTS}d}"

echo ""
echo "Configuration:"
echo "  Graph dir: $GRAPH_DIR"
echo "  Dataset name: $DATASET_NAME"
echo "  Output dir: $OUTPUT_DIR"
echo "  UMAP target dimensions: ${UMAP_N_COMPONENTS}D"
echo "  UMAP min_dist: $UMAP_MIN_DIST (lower=more local, higher=more global)"
echo "  UMAP n_neighbors: $UMAP_N_NEIGHBORS (lower=more local, higher=more global)"
echo "  UMAP metric: precomputed (using graph distance matrix)"
echo "  Save UMAP result: $SAVE_UMAP_RESULT"
echo "  Generate visualizations: $GENERATE_VISUALIZATIONS"
echo ""

# Check if graph distance matrix exists
DISTANCE_MATRIX_FILE="${GRAPH_DIR}/distance_matrix_${DATASET_NAME}.npy"
if [ ! -f "$DISTANCE_MATRIX_FILE" ]; then
    echo "Error: Graph distance matrix not found: $DISTANCE_MATRIX_FILE"
    echo "Please ensure the graph has been generated first"
    echo "  Expected location: $DISTANCE_MATRIX_FILE"
    exit 1
fi

echo "Found graph distance matrix: $DISTANCE_MATRIX_FILE"
echo ""
echo "Applying UMAP to graph distance matrix..."

python scripts/13c_umap_analysis_graph.py \
    --graph_dir "$GRAPH_DIR" \
    --dataset_name "$DATASET_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --umap_n_components "$UMAP_N_COMPONENTS" \
    --umap_min_dist "$UMAP_MIN_DIST" \
    --umap_n_neighbors "$UMAP_N_NEIGHBORS" \
    $(if [ "$SAVE_UMAP_RESULT" = "true" ]; then echo "--save_umap_result"; fi) \
    $(if [ "$GENERATE_VISUALIZATIONS" = "true" ]; then echo "--generate_visualizations"; fi)

echo ""
echo "=========================================="
echo "UMAP Analysis complete!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
if [ "$SAVE_UMAP_RESULT" = "true" ]; then
    echo "UMAP embeddings saved as: ground_truth_umap_${UMAP_N_COMPONENTS}d.npz"
    echo "  Each file contains: umap_reduced (shape: [n_points, ${UMAP_N_COMPONENTS}])"
fi
if [ "$GENERATE_VISUALIZATIONS" = "true" ]; then
    echo "Visualizations saved as: ground_truth_umap_${UMAP_N_COMPONENTS}d.{png,html}"
fi
echo ""
echo "Next step: Run topology analysis with"
echo "  ./14a_topology_analysis_graph.sh"
echo ""

