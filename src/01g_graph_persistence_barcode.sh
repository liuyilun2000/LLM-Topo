#!/bin/bash

# Persistence barcode visualization for graph topology results
#
# This script mirrors 04b_persistence_barcode.sh but operates on graph
# topology results from 01f_graph_topology_analysis.sh.
# Reuses 04b_persistence_barcode.py.
#
# Pipeline: 01f (topology) → 01g (barcode)

set -e

echo "=========================================="
echo "Graph Persistence Barcode"
echo "=========================================="

# Load shared configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/00_config_env.sh"

# Input: topology results from 01f
TOPOLOGY_DIR="${TOPOLOGY_DIR:-./${DATA_DIR}/graph_topology}"
OUTPUT_DIR="${OUTPUT_DIR:-./${DATA_DIR}/graph_persistence_barcode}"

# MAX_BARS and TOP_K are loaded from 00_config_env.sh

echo ""
echo "Configuration:"
echo "  Topology dir: $TOPOLOGY_DIR"
echo "  Output dir: $OUTPUT_DIR"
echo "  Max bars per dimension: $MAX_BARS"
echo "  Top-k: $TOP_K"
echo ""

# Check if topology results exist
if [ ! -d "$TOPOLOGY_DIR" ] || [ -z "$(ls -A $TOPOLOGY_DIR/*_topology.json 2>/dev/null)" ]; then
    echo "Error: Topology results not found in $TOPOLOGY_DIR"
    echo "Please run ./01f_graph_topology_analysis.sh first"
    exit 1
fi

echo "Generating persistence barcodes..."
python ../scripts/04b_persistence_barcode.py \
    --topology_dir "$TOPOLOGY_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --max_bars "$MAX_BARS" \
    --top_k "$TOP_K"

echo ""
echo "=========================================="
echo "Barcode visualization complete!"
echo "=========================================="
echo "Plots and statistics saved to: $OUTPUT_DIR"
echo ""
echo "To view results:"
echo "  ls $OUTPUT_DIR/*_barcode.png"
echo "  ls $OUTPUT_DIR/*_barcode.pdf"
echo "  ls $OUTPUT_DIR/*_statistics.json"
echo ""
