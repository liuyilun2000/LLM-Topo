#!/bin/bash

# Visualize UMAP embedding from fuzzy neighborhood distance matrix with trajectory overlay
#
# This script:
# 1. Loads a fuzzy neighborhood distance matrix
# 2. Applies UMAP to get 3D embeddings
# 3. Loads a specified trajectory from walks CSV
# 4. Creates a 3D visualization with:
#    - Point cloud colored with blue-green-yellow gradient
#    - Trajectory overlaid with red-orange-yellow gradient based on time
# 5. Saves as PNG, HTML, and metadata JSON

set -e

echo "=========================================="
echo "UMAP Visualization with Trajectory"
echo "=========================================="

# Load shared configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/00_config_env.sh"

# Local configuration

# Fuzzy neighborhood directory
FUZZY_DIR="${FUZZY_DIR:-./${WORK_DIR}/fuzzy_neighborhood}"

# Walks CSV file
WALKS_CSV="${WALKS_CSV:-./${DATA_DIR}/sequences/walks_${DATASET_NAME}.csv}"

# Representation key (e.g., "layer_1_after_block", "input_embeds", etc.)
KEY="${KEY:-layer_5_hidden}"

# Trajectory selection
WALK_ID="${WALK_ID:-}"  # If set, uses this walk_id
TRAJECTORY_IDX="${TRAJECTORY_IDX:-256}"  # If WALK_ID not set, uses this index

# UMAP parameters
UMAP_N_COMPONENTS="${UMAP_N_COMPONENTS:-3}"
UMAP_MIN_DIST="${UMAP_MIN_DIST:-0.2}"
UMAP_N_NEIGHBORS="${UMAP_N_NEIGHBORS:-20}"
UMAP_RANDOM_STATE="${UMAP_RANDOM_STATE:-100}"  # Set to integer for reproducibility, empty for random

# Trajectory parameters
MAX_LENGTH="${MAX_LENGTH:-128}"  # Maximum number of points to plot in trajectory

# Output directory
OUTPUT_DIR="${OUTPUT_DIR:-./${WORK_DIR}/umap_plot}"

echo ""
echo "Configuration:"
echo "  Fuzzy dir: $FUZZY_DIR"
echo "  Key: $KEY"
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

# Check if fuzzy distance matrix exists
FUZZY_FILE="${FUZZY_DIR}/${KEY}_fuzzy_dist.npz"
if [ ! -f "$FUZZY_FILE" ]; then
    echo "Error: Fuzzy distance matrix not found: $FUZZY_FILE"
    echo "Please run ./03b_fuzzy_neighborhood.sh first"
    exit 1
fi

# Check if walks CSV exists
if [ ! -f "$WALKS_CSV" ]; then
    echo "Error: Walks CSV file not found: $WALKS_CSV"
    echo "Please ensure walks have been generated first"
    exit 1
fi

echo "Found fuzzy distance matrix: $FUZZY_FILE"
echo "Found walks CSV: $WALKS_CSV"
echo ""
echo "Generating visualization..."

# Build and run Python script
if [ -n "$WALK_ID" ]; then
    python scripts/93c_umap_analysis_plot.py \
        --fuzzy_dir "$FUZZY_DIR" \
        --key "$KEY" \
        --walks_csv "$WALKS_CSV" \
        --walk_id "$WALK_ID" \
        --output_dir "$OUTPUT_DIR" \
        --umap_n_components "$UMAP_N_COMPONENTS" \
        --umap_min_dist "$UMAP_MIN_DIST" \
        --umap_n_neighbors "$UMAP_N_NEIGHBORS" \
        $(if [ -n "$UMAP_RANDOM_STATE" ]; then echo "--umap_random_state $UMAP_RANDOM_STATE"; fi) \
        --max_length "$MAX_LENGTH"
else
    python scripts/93c_umap_analysis_plot.py \
        --fuzzy_dir "$FUZZY_DIR" \
        --key "$KEY" \
        --walks_csv "$WALKS_CSV" \
        --trajectory_idx "$TRAJECTORY_IDX" \
        --output_dir "$OUTPUT_DIR" \
        --umap_n_components "$UMAP_N_COMPONENTS" \
        --umap_min_dist "$UMAP_MIN_DIST" \
        --umap_n_neighbors "$UMAP_N_NEIGHBORS" \
        $(if [ -n "$UMAP_RANDOM_STATE" ]; then echo "--umap_random_state $UMAP_RANDOM_STATE"; fi) \
        --max_length "$MAX_LENGTH"
fi
