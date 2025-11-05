#!/bin/bash

# Generate persistence diagrams from the original graph distance matrix
#
# This script generates persistence diagrams from the original graph distance
# matrix. This is used for validation - to examine whether the current 
# methodology's working on representations in LLM's results is valid or not
# by comparing against the ground truth graph topology.
#
# The graph distance matrix is loaded from:
#   results/${DATASET_NAME}/graph/distance_matrix_${DATASET_NAME}.npy
# where DATASET_NAME is determined by TOPOLOGY, H, W from 00_config_env.sh

set -e

echo "=========================================="
echo "Topology Analysis (Graph Ground Truth)"
echo "=========================================="

# Load shared configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/00_config_env.sh"

# Local configuration

# Graph directory (contains the original distance matrix)
GRAPH_DIR="${GRAPH_DIR:-./${DATA_DIR}/graph}"

# Output directory
OUTPUT_DIR="${OUTPUT_DIR:-./${WORK_DIR}/topology_analysis_graph}"

# Ripser parameters for persistence diagram generation
RIPSER_THRESH="${RIPSER_THRESH:-}"
RIPSER_MAXDIM="${RIPSER_MAXDIM:-2}"
RIPSER_COEFF="${RIPSER_COEFF:-47}"

echo ""
echo "Configuration:"
echo "  Graph dir: $GRAPH_DIR"
echo "  Dataset name: $DATASET_NAME"
echo "  Output dir: $OUTPUT_DIR"
if [ -z "$RIPSER_THRESH" ]; then
    echo "  Ripser threshold: (empty = full filtration)"
else
    echo "  Ripser threshold: $RIPSER_THRESH"
fi
echo "  Ripser maxdim: $RIPSER_MAXDIM"
echo "  Ripser coeff: Z$RIPSER_COEFF"
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
echo "Generating persistence diagrams from graph distance matrix..."

# Build ripser arguments
RIPSER_ARGS=""
if [ -n "$RIPSER_THRESH" ]; then
    RIPSER_ARGS="$RIPSER_ARGS --ripser_thresh $RIPSER_THRESH"
fi
RIPSER_ARGS="$RIPSER_ARGS --ripser_maxdim $RIPSER_MAXDIM"
RIPSER_ARGS="$RIPSER_ARGS --ripser_coeff $RIPSER_COEFF"

python scripts/14a_topology_analysis_graph.py \
    --graph_dir "$GRAPH_DIR" \
    --dataset_name "$DATASET_NAME" \
    --output_dir "$OUTPUT_DIR" \
    $RIPSER_ARGS

echo ""
echo "=========================================="
echo "Persistence diagram generation complete!"
echo "=========================================="
echo ""
echo "Persistence diagrams (PNG) and data (JSON) saved to: $OUTPUT_DIR"
echo ""
echo "Results saved as: ground_truth_topology.json and ground_truth_persistence_diagram.png"
echo ""
echo "Next step: Visualize persistence barcodes with"
echo "  ./14b_persistence_barcode_graph.sh"
echo ""

