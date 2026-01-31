#!/bin/bash

# Topology analysis (persistence diagrams) on graph UMAP embeddings
#
# This script mirrors 04a_topology_analysis.sh but operates on graph UMAP
# results from 01e_graph_umap_analysis.sh instead of model representations.
# Reuses 04a_topology_analysis.py with data_representation mode.
#
# Pipeline: 01e (UMAP 6D) → 01f (topology) → 01g (barcode)

set -e

echo "=========================================="
echo "Graph Topology Analysis (Persistence Diagrams)"
echo "=========================================="

# Load shared configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/00_config_env.sh"

# Input: UMAP graph results from 01e
INPUT_DIR="${INPUT_DIR:-./${DATA_DIR}/graph_umap_result_6d}"
DATA_TYPE="${DATA_TYPE:-umap}"

# Output directory
OUTPUT_DIR="${OUTPUT_DIR:-./${DATA_DIR}/graph_topology}"

# Ripser parameters
RIPSER_THRESH="${RIPSER_THRESH:-}"
RIPSER_MAXDIM="${RIPSER_MAXDIM:-2}"
RIPSER_COEFF="${RIPSER_COEFF:-3}"

echo ""
echo "Configuration:"
echo "  Input dir (graph UMAP): $INPUT_DIR"
echo "  Data type: $DATA_TYPE"
echo "  Output dir: $OUTPUT_DIR"
echo "  Ripser maxdim: $RIPSER_MAXDIM"
echo "  Ripser coeff: Z$RIPSER_COEFF"
if [ -z "$RIPSER_THRESH" ]; then
    echo "  Ripser threshold: (empty = full filtration)"
else
    echo "  Ripser threshold: $RIPSER_THRESH"
fi
echo ""

# Check if UMAP graph results exist
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Graph UMAP results not found: $INPUT_DIR"
    echo "Please run ./01e_graph_umap_analysis.sh first"
    exit 1
fi

if [ -z "$(ls -A $INPUT_DIR/*_umap_*d.npz 2>/dev/null)" ]; then
    echo "Error: No UMAP result files (*_umap_*d.npz) found in $INPUT_DIR"
    echo "Please run ./01e_graph_umap_analysis.sh first"
    exit 1
fi

echo "Generating persistence diagrams..."
RIPSER_ARGS=""
[ -n "$RIPSER_THRESH" ] && RIPSER_ARGS="$RIPSER_ARGS --ripser_thresh $RIPSER_THRESH"
RIPSER_ARGS="$RIPSER_ARGS --ripser_maxdim $RIPSER_MAXDIM"
RIPSER_ARGS="$RIPSER_ARGS --ripser_coeff $RIPSER_COEFF"

python ../scripts/04a_topology_analysis.py \
    --data_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --data_type "$DATA_TYPE" \
    $RIPSER_ARGS

echo ""
echo "=========================================="
echo "Topology analysis complete!"
echo "=========================================="
echo "Persistence diagrams saved to: $OUTPUT_DIR"
echo ""
echo "Next step: Visualize persistence barcodes"
echo "  ./01g_graph_persistence_barcode.sh"
echo ""
