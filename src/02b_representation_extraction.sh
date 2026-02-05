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

# REPRESENTATIONS, REPRESENTATION_UPSEAMPLE, REPRESENTATION_UPSEAMPLE_N, REPRESENTATION_UPSEAMPLE_SEED from 00_config_env.sh

echo ""
echo "Configuration:"
echo "  Model: $MODEL_DIR"
echo "  Output dir: $REPRESENTATION_DIR"
echo "  Representations: $REPRESENTATIONS"
echo "  Upsample: $REPRESENTATION_UPSEAMPLE (N=$REPRESENTATION_UPSEAMPLE_N, seed=$REPRESENTATION_UPSEAMPLE_SEED)"
echo ""

# When upsampling, dataset is required
if [ "$REPRESENTATION_UPSEAMPLE" = "true" ]; then
    if [ ! -d "$DATASET_DIR" ]; then
        echo "Error: Upsampling is enabled but dataset directory $DATASET_DIR not found!"
        echo "Please run ./01c_dataset_preparation.sh first or set REPRESENTATION_UPSEAMPLE=false"
        exit 1
    fi
fi

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
EXTRA_ARGS=()
if [ "$REPRESENTATION_UPSEAMPLE" = "true" ]; then
    EXTRA_ARGS+=(--upsample --dataset_dir "$DATASET_DIR" --upsample_n "$REPRESENTATION_UPSEAMPLE_N" --seed "$REPRESENTATION_UPSEAMPLE_SEED")
fi
python ../scripts/02b_representation_extraction.py \
    --model_dir "$MODEL_DIR" \
    --output_dir "$REPRESENTATION_DIR" \
    --representations $REPRESENTATIONS \
    "${EXTRA_ARGS[@]}"

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


