#!/bin/bash

# Perform PCA dimensionality reduction analysis
#
# This script applies Principal Component Analysis (PCA) to the extracted
# token representations. PCA can be configured to retain a specific number of
# components or a target variance threshold.

set -e

echo "=========================================="
echo "PCA Dimensionality Reduction Analysis"
echo "=========================================="

# Load shared configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/00_config_env.sh"

# PCA_N_COMPONENTS, PCA_VARIANCE from 00_config_env.sh (use one or the other)

echo ""
echo "Configuration:"
echo "  Representation dir: $REPRESENTATION_DIR"
echo "  Output dir: $PCA_DIR"

# Check which PCA parameter is set
if [ -n "$PCA_N_COMPONENTS" ]; then
    echo "  PCA target dimensionality: $PCA_N_COMPONENTS"
    PCA_OPTION="--n_components $PCA_N_COMPONENTS"
elif [ -n "$PCA_VARIANCE" ]; then
    echo "  PCA variance threshold: $PCA_VARIANCE"
    PCA_OPTION="--variance_threshold $PCA_VARIANCE"
else
    echo "Error: Either PCA_N_COMPONENTS or PCA_VARIANCE must be set"
    exit 1
fi
echo ""

# Check if representations exist
if [ ! -f "$REPRESENTATION_DIR/token_representations.npz" ]; then
    echo "Error: Token representations not found in $REPRESENTATION_DIR"
    echo "Please run ./02b_representation_extraction.sh first"
    exit 1
fi

echo "Running PCA analysis..."
python ../scripts/03a_pca_analysis.py \
    --representation_dir "$REPRESENTATION_DIR" \
    --output_dir "$PCA_DIR" \
    $PCA_OPTION

echo ""
echo "=========================================="
echo "PCA analysis complete!"
echo "=========================================="
echo ""
echo "Results saved to: $PCA_DIR"
echo ""
echo "Generated files per representation:"
echo "  - {key}_pca.npz (PCA-reduced data)"
echo "  - {key}_pca_model.pkl (PCA model + scaler)"
echo "  - {key}_pca_info.json (PCA information)"
echo ""
echo "Next step options:"
echo "  1. Compute fuzzy neighborhood distance matrix for persistent homology:"
echo "     ./03b_fuzzy_neighborhood.sh"
echo "  2. Apply UMAP dimensionality reduction:"
echo "     ./03c_umap_visualize.sh"
echo ""

