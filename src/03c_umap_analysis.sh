#!/bin/bash

# Apply UMAP and generate dimensionality reduction results
#
# This script applies UMAP (Uniform Manifold Approximation and Projection) to
# reduce the dimensionality of the token representations. It supports multiple
# input modes:
#   - Fuzzy neighborhood distance matrices (USE_FUZZY=true)
#   - Downsampled PCA data (USE_DOWNSAMPLED=true)
#   - Regular PCA data (default)

set -e

echo "=========================================="
echo "UMAP Analysis"
echo "=========================================="

# Load shared configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/00_config_env.sh"

# Local configuration

# Determine which data to use
USE_DOWNSAMPLED="${USE_DOWNSAMPLED:-false}"
USE_FUZZY="${USE_FUZZY:-true}"

if [ "$USE_FUZZY" = "true" ]; then
    FUZZY_DIR="${FUZZY_DIR:-./${WORK_DIR}/fuzzy_neighborhood}"
    echo "Input mode: Fuzzy neighborhood distance matrices"
elif [ "$USE_DOWNSAMPLED" = "true" ]; then
    PCA_DIR="${PCA_DIR:-./${WORK_DIR}/density_downsampling}"
    echo "Input mode: Downsampled PCA data"
else
    PCA_DIR="${PCA_DIR:-./${WORK_DIR}/pca_result}"
    echo "Input mode: Regular PCA data"
fi

REPRESENTATION_DIR="${REPRESENTATION_DIR:-./${WORK_DIR}/token_representations}"

# UMAP parameters
UMAP_N_COMPONENTS="${UMAP_N_COMPONENTS:-6}"
UMAP_MIN_DIST="${UMAP_MIN_DIST:-0.2}"
UMAP_N_NEIGHBORS="${UMAP_N_NEIGHBORS:-20}"
UMAP_METRIC="${UMAP_METRIC:-cosine}"
UMAP_RANDOM_STATE="${UMAP_RANDOM_STATE:-42}"  # Set to integer for reproducibility, empty for random

# Output control flags
SAVE_UMAP_RESULT="${SAVE_UMAP_RESULT:-true}"
GENERATE_VISUALIZATIONS="${GENERATE_VISUALIZATIONS:-false}"

# Output directory (includes dimensionality)
OUTPUT_DIR="${OUTPUT_DIR:-./${WORK_DIR}/umap_result_${UMAP_N_COMPONENTS}d}"

echo ""
echo "Configuration:"
if [ "$USE_FUZZY" = "true" ]; then
    echo "  Fuzzy dir: $FUZZY_DIR"
else
    echo "  PCA dir: $PCA_DIR"
fi
echo "  Representation dir: $REPRESENTATION_DIR"
echo "  Output dir: $OUTPUT_DIR"
echo "  UMAP target dimensions: ${UMAP_N_COMPONENTS}D"
echo "  UMAP min_dist: $UMAP_MIN_DIST (lower=more local, higher=more global)"
echo "  UMAP n_neighbors: $UMAP_N_NEIGHBORS (lower=more local, higher=more global)"
if [ "$USE_FUZZY" = "true" ]; then
    echo "  UMAP metric: precomputed (using fuzzy neighborhood distance matrix)"
else
    echo "  UMAP metric: $UMAP_METRIC"
fi
if [ -n "$UMAP_RANDOM_STATE" ]; then
    echo "  UMAP random_state: $UMAP_RANDOM_STATE"
else
    echo "  UMAP random_state: (random)"
fi
echo "  Save UMAP result: $SAVE_UMAP_RESULT"
echo "  Generate visualizations: $GENERATE_VISUALIZATIONS"
echo ""

# Check if results exist
if [ "$USE_FUZZY" = "true" ]; then
    if [ ! -d "$FUZZY_DIR" ] || [ -z "$(ls -A $FUZZY_DIR/*_fuzzy_dist.npz 2>/dev/null)" ]; then
        echo "Error: Fuzzy neighborhood distance matrices not found in $FUZZY_DIR"
        echo "Please run ./03b_fuzzy_neighborhood.sh first"
        exit 1
    fi
    USE_FUZZY_FLAG="--use_fuzzy"
    USE_DOWNSAMPLED_FLAG=""
    FUZZY_DIR_ARG="--fuzzy_dir"
elif [ "$USE_DOWNSAMPLED" = "true" ]; then
    if [ ! -d "$PCA_DIR" ] || [ -z "$(ls -A $PCA_DIR/*_pca_downsampled.npz 2>/dev/null)" ]; then
        echo "Error: Downsampled PCA results not found in $PCA_DIR"
        echo "Please run density downsampling step first"
        exit 1
    fi
    USE_FUZZY_FLAG=""
    USE_DOWNSAMPLED_FLAG="--use_downsampled"
    FUZZY_DIR_ARG=""
else
    if [ ! -d "$PCA_DIR" ] || [ -z "$(ls -A $PCA_DIR/*_pca.npz 2>/dev/null)" ]; then
        echo "Error: PCA results not found in $PCA_DIR"
        echo "Please run ./03a_pca_analysis.sh first"
        exit 1
    fi
    USE_FUZZY_FLAG=""
    USE_DOWNSAMPLED_FLAG=""
    FUZZY_DIR_ARG=""
fi

echo "Applying UMAP..."

if [ "$USE_FUZZY" = "true" ]; then
    python scripts/03c_umap_analysis.py \
        --fuzzy_dir "$FUZZY_DIR" \
        --representation_dir "$REPRESENTATION_DIR" \
        --output_dir "$OUTPUT_DIR" \
        $USE_FUZZY_FLAG \
        --umap_n_components "$UMAP_N_COMPONENTS" \
        --umap_min_dist "$UMAP_MIN_DIST" \
        --umap_n_neighbors "$UMAP_N_NEIGHBORS" \
        $(if [ -n "$UMAP_RANDOM_STATE" ]; then echo "--umap_random_state $UMAP_RANDOM_STATE"; fi) \
        $(if [ "$SAVE_UMAP_RESULT" = "true" ]; then echo "--save_umap_result"; fi) \
        $(if [ "$GENERATE_VISUALIZATIONS" = "true" ]; then echo "--generate_visualizations"; fi)
else
    python scripts/03c_umap_analysis.py \
        --pca_dir "$PCA_DIR" \
        --representation_dir "$REPRESENTATION_DIR" \
        --output_dir "$OUTPUT_DIR" \
        $USE_DOWNSAMPLED_FLAG \
        --umap_n_components "$UMAP_N_COMPONENTS" \
        --umap_min_dist "$UMAP_MIN_DIST" \
        --umap_n_neighbors "$UMAP_N_NEIGHBORS" \
        --umap_metric "$UMAP_METRIC" \
        $(if [ -n "$UMAP_RANDOM_STATE" ]; then echo "--umap_random_state $UMAP_RANDOM_STATE"; fi) \
        $(if [ "$SAVE_UMAP_RESULT" = "true" ]; then echo "--save_umap_result"; fi) \
        $(if [ "$GENERATE_VISUALIZATIONS" = "true" ]; then echo "--generate_visualizations"; fi)
fi

echo ""
echo "=========================================="
echo "UMAP Analysis complete!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
if [ "$SAVE_UMAP_RESULT" = "true" ]; then
    echo "UMAP embeddings saved as: {key}_umap_${UMAP_N_COMPONENTS}d.npz"
    echo "  Each file contains: umap_reduced (shape: [n_points, ${UMAP_N_COMPONENTS}])"
fi
if [ "$GENERATE_VISUALIZATIONS" = "true" ]; then
    echo "Visualizations saved as: {key}_umap_${UMAP_N_COMPONENTS}d.{png,html}"
fi
echo ""
echo "Next step: Run topology analysis with"
echo "  ./04a_topology_analysis.sh (with INPUT_MODE=data_representation)"
echo ""

