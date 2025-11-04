#!/bin/bash

# Extract token representations from trained model
#
# This script extracts internal representations from specified points in the model
# architecture for each token in the vocabulary. The representations are saved
# in NPZ format for subsequent analysis.
#
# Supported representation types:
#   residual_before  - Residual stream state BEFORE each decoder block (input to block)
#   after_attention  - Hidden state AFTER attention, before FFN
#   after_block      - Hidden state AFTER entire decoder block (attention + FFN + residuals) [default]
#   ffn_gate         - FFN gate projection activations
#   ffn_up           - FFN up projection activations

set -e

echo "=========================================="
echo "Token Representation Extraction"
echo "=========================================="

# Load shared configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/00_config_env.sh"

# Local configuration
MODEL_DIR="${MODEL_DIR:-./${WORK_DIR}/final_model}"
REPRESENTATION_DIR="${REPRESENTATION_DIR:-./${WORK_DIR}/token_representations}"

# Representation types to extract (default: after_block)
# Options: residual_before, after_attention, after_block, ffn_gate, ffn_up
# Multiple can be specified: REPRESENTATIONS="after_block ffn_gate ffn_up"
REPRESENTATIONS="${REPRESENTATIONS:-after_block}"

echo ""
echo "Configuration:"
echo "  Model: $MODEL_DIR"
echo "  Output dir: $REPRESENTATION_DIR"
echo "  Representations: $REPRESENTATIONS"
echo ""

# Check if model exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model directory $MODEL_DIR not found!"
    echo "Please run ./02a_model_training.sh first"
    exit 1
fi

# Check if already extracted
if [ -f "$REPRESENTATION_DIR/token_representations.npz" ]; then
    echo "Token representations already exist in $REPRESENTATION_DIR"
    echo ""
    read -p "Delete and re-extract? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping extraction."
        exit 0
    fi
    rm -rf "$REPRESENTATION_DIR"
fi

echo "Extracting token representations..."
python scripts/02b_representation_extraction.py \
    --model_dir "$MODEL_DIR" \
    --output_dir "$REPRESENTATION_DIR" \
    --representations $REPRESENTATIONS

echo ""
echo "=========================================="
echo "Extraction complete!"
echo "=========================================="
echo ""
echo "Representations saved to: $REPRESENTATION_DIR"
echo ""
echo "Next step: Run PCA analysis with"
echo "  ./03a_pca_analysis.sh"
echo ""


