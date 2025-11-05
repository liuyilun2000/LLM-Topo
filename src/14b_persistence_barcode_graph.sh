#!/bin/bash

# Plot persistence barcodes from graph topology analysis results
#
# This script generates barcode diagrams for H0, H1, H2 with top N longest bars,
# ordered by birth. Uses global top-k method: marks top-k longest bars globally
# (across all dimensions) as significant. Top-k longest bars globally are marked
# with stars (★).
#
# This version uses results from the graph ground truth analysis (from
# 14a_topology_analysis_graph.sh) to validate the methodology.

set -e

echo "=========================================="
echo "Persistence Barcode Visualization (Graph Ground Truth)"
echo "=========================================="

# Load shared configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/00_config_env.sh"

# Local configuration
TOPOLOGY_DIR="${TOPOLOGY_DIR:-./${WORK_DIR}/topology_analysis_graph}"
OUTPUT_DIR="${OUTPUT_DIR:-./${WORK_DIR}/persistence_barcode_graph}"

# Barcode parameters
MAX_BARS="${MAX_BARS:-30}"
TOP_K="${TOP_K:-2}"

echo ""
echo "Configuration:"
echo "  Topology dir: $TOPOLOGY_DIR"
echo "  Output dir: $OUTPUT_DIR"
echo "  Max bars per dimension: $MAX_BARS"
echo "  Top-k: $TOP_K (global across all dimensions)"
echo "  Statistical method: Global top-k method"
echo "  Source: Graph ground truth topology analysis"
echo ""

# Check if topology results exist
if [ ! -d "$TOPOLOGY_DIR" ] || [ ! -f "$TOPOLOGY_DIR/ground_truth_topology.json" ]; then
    echo "Error: Graph topology results not found in $TOPOLOGY_DIR"
    echo "Expected file: ground_truth_topology.json"
    echo "Please run ./14a_topology_analysis_graph.sh first"
    exit 1
fi

echo "Generating persistence barcodes from graph ground truth..."
python scripts/04b_persistence_barcode.py \
    --topology_dir "$TOPOLOGY_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --max_bars "$MAX_BARS" \
    --top_k "$TOP_K" \
    --keys ground_truth

echo ""
echo "=========================================="
echo "Barcode visualization complete!"
echo "=========================================="
echo ""
echo "Plots and statistics saved to: $OUTPUT_DIR"
echo ""
echo "Results saved as:"
echo "  - ground_truth_barcode.png"
echo "  - ground_truth_barcode.pdf"
echo "  - ground_truth_statistics.json"
echo ""
echo "To view results:"
echo "  ls $OUTPUT_DIR/ground_truth_barcode.*"
echo "  ls $OUTPUT_DIR/ground_truth_statistics.json"
echo ""

