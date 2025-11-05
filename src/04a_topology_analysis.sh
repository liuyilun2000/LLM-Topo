#!/bin/bash

# Generate persistence diagrams from data representations or distance matrices
#
# This script generates persistence diagrams (no analysis, no Betti numbers).
# Two input modes are supported:
#
# 1. Distance matrix mode (INPUT_MODE=distance_matrix)
#    - Uses fuzzy neighborhood distance matrices
#    - Source: fuzzy_neighborhood directory (from 03b_fuzzy_neighborhood.sh)
#
# 2. Data representation mode (INPUT_MODE=data_representation)
#    - Uses data embeddings (PCA, UMAP, downsampled PCA)
#    - Source: pca_analysis, umap_result_*, or density_downsampling directory
#    - Data type: auto (detect), pca, umap, or downsampled

set -e

echo "=========================================="
echo "Topology Analysis with Persistence Diagram Generation"
echo "=========================================="

# Load shared configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/00_config_env.sh"

# Local configuration

# Input mode: "distance_matrix" or "data_representation"
#  - distance_matrix: Uses fuzzy neighborhood distance matrices
#  - data_representation: Uses data embeddings (PCA, UMAP, downsampled PCA)
INPUT_MODE="${INPUT_MODE:-data_representation}"

# For distance_matrix mode
if [ "$INPUT_MODE" = "distance_matrix" ]; then
    FUZZY_DIR="${FUZZY_DIR:-./${WORK_DIR}/fuzzy_neighborhood}"
    REPRESENTATION_DIR=""
    DATA_TYPE=""
    echo "Input mode: Distance matrix (fuzzy neighborhood distance matrices)"
# For data_representation mode
else
    FUZZY_DIR=""
    REPRESENTATION_DIR="${REPRESENTATION_DIR:-./${WORK_DIR}/umap_result_6d}"
    # Data type: 'auto' (detect), 'pca', 'umap', or 'downsampled'
    DATA_TYPE="${DATA_TYPE:-auto}"
    echo "Input mode: Data representation (PCA, UMAP, downsampled PCA)"
    echo "  Data dir: $REPRESENTATION_DIR"
    echo "  Data type: $DATA_TYPE"
fi

OUTPUT_DIR="${OUTPUT_DIR:-./${WORK_DIR}/topology_analysis}"

# Ripser parameters for persistence diagram generation
RIPSER_THRESH="${RIPSER_THRESH:-}"
RIPSER_MAXDIM="${RIPSER_MAXDIM:-2}"
RIPSER_COEFF="${RIPSER_COEFF:-3}"

echo ""
echo "Configuration:"
echo "  Input mode: $INPUT_MODE"
if [ "$INPUT_MODE" = "distance_matrix" ]; then
    echo "  Fuzzy dir: $FUZZY_DIR"
else
    echo "  Data dir: $REPRESENTATION_DIR"
    echo "  Data type: $DATA_TYPE"
fi
echo "  Output dir: $OUTPUT_DIR"
if [ -z "$RIPSER_THRESH" ]; then
    echo "  Ripser threshold: (empty = full filtration)"
else
    echo "  Ripser threshold: $RIPSER_THRESH"
fi
echo "  Ripser maxdim: $RIPSER_MAXDIM"
echo "  Ripser coeff: Z$RIPSER_COEFF"
echo ""

# Check if results exist
if [ "$INPUT_MODE" = "distance_matrix" ]; then
    if [ ! -d "$FUZZY_DIR" ] || [ -z "$(ls -A $FUZZY_DIR/*_fuzzy_dist.npz 2>/dev/null)" ]; then
        echo "Error: Fuzzy distance matrices not found in $FUZZY_DIR"
        echo "Please run ./03b_fuzzy_neighborhood.sh first"
        exit 1
    fi
else
    if [ ! -d "$REPRESENTATION_DIR" ]; then
        echo "Error: Data representation directory not found: $REPRESENTATION_DIR"
        echo "Please run appropriate preprocessing step:"
        echo "  - ./03a_pca_analysis.sh (for PCA)"
        echo "  - ./03c_umap_analysis.sh (for UMAP)"
        exit 1
    fi
    # Check if any relevant files exist
    if [ "$DATA_TYPE" = "umap" ] || [ "$DATA_TYPE" = "auto" ]; then
        if [ -z "$(ls -A $REPRESENTATION_DIR/*_umap_*d.npz 2>/dev/null)" ]; then
            if [ "$DATA_TYPE" = "umap" ]; then
                echo "Error: UMAP results not found in $REPRESENTATION_DIR"
                echo "Please run ./03c_umap_analysis.sh first"
                exit 1
            fi
        fi
    fi
fi

echo "Generating persistence diagrams..."

# Build ripser arguments
RIPSER_ARGS=""
if [ -n "$RIPSER_THRESH" ]; then
    RIPSER_ARGS="$RIPSER_ARGS --ripser_thresh $RIPSER_THRESH"
fi
RIPSER_ARGS="$RIPSER_ARGS --ripser_maxdim $RIPSER_MAXDIM"
RIPSER_ARGS="$RIPSER_ARGS --ripser_coeff $RIPSER_COEFF"

if [ "$INPUT_MODE" = "distance_matrix" ]; then
    # Mode 1: Distance matrix input
    python scripts/04a_topology_analysis.py \
        --fuzzy_dir "$FUZZY_DIR" \
        --output_dir "$OUTPUT_DIR" \
        $RIPSER_ARGS
else
    # Mode 2: Data representation input
    python scripts/04a_topology_analysis.py \
        --data_dir "$REPRESENTATION_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --data_type "$DATA_TYPE" \
        $RIPSER_ARGS
fi

echo ""
echo "=========================================="
echo "Persistence diagram generation complete!"
echo "=========================================="
echo ""
echo "Persistence diagrams (PNG) and data (JSON) saved to: $OUTPUT_DIR"
echo ""
echo "Next step: Visualize persistence barcodes with"
echo "  ./04b_persistence_barcode.sh"
echo ""

