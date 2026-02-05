#!/bin/bash

# Plot persistence barcodes from topology analysis results
#
# This script generates barcode diagrams for H0, H1, H2 with top N longest bars,
# ordered by birth. Uses global top-k method: marks top-k longest bars globally
# (across all dimensions) as significant. Top-k longest bars globally are marked
# with stars (★).

set -e

echo "=========================================="
echo "Persistence Barcode Visualization"
echo "=========================================="

# Load shared configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/00_config_env.sh"

# MAX_BARS and TOP_K are loaded from 00_config_env.sh

echo ""
echo "Configuration:"
echo "  Topology dir: $TOPOLOGY_ANALYSIS_DIR"
echo "  Output dir: $PERSISTENCE_BARCODE_DIR"
echo "  Max bars per dimension: $MAX_BARS"
echo "  Top-k: $TOP_K (global across all dimensions)"
echo "  Statistical method: Global top-k method"
echo ""

# Check if topology results exist
if [ ! -d "$TOPOLOGY_ANALYSIS_DIR" ] || [ -z "$(ls -A $TOPOLOGY_ANALYSIS_DIR/*_topology.json 2>/dev/null)" ]; then
    echo "Error: Topology results not found in $TOPOLOGY_ANALYSIS_DIR"
    echo "Please run ./04a_topology_analysis.sh first"
    exit 1
fi

echo "Generating persistence barcodes..."
python ../scripts/04b_persistence_barcode.py \
    --topology_dir "$TOPOLOGY_ANALYSIS_DIR" \
    --output_dir "$PERSISTENCE_BARCODE_DIR" \
    --max_bars "$MAX_BARS" \
    --top_k "$TOP_K"

echo ""
echo "=========================================="
echo "Barcode visualization complete!"
echo "=========================================="
echo ""
echo "Plots and statistics saved to: $PERSISTENCE_BARCODE_DIR"
echo ""
echo "To view results:"
echo "  ls $PERSISTENCE_BARCODE_DIR/*_barcode.png"
echo "  ls $PERSISTENCE_BARCODE_DIR/*_barcode.pdf"
echo "  ls $PERSISTENCE_BARCODE_DIR/*_statistics.json"
echo ""

