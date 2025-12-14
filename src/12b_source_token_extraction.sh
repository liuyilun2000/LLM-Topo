#!/bin/bash

# Extract GRAPH token representations from trained model
#
# This script extracts internal representations for GRAPH tokens (<GRAPH_0>, <GRAPH_1>, 
# ..., <GRAPH_1199>) that were added to the vocabulary during combined dataset preparation.
#
# IMPORTANT: Extracts GRAPH tokens from sequences of the form [SOURCE_START, GRAPH_xxx].
# Only the GRAPH token's hidden state (position 1) is extracted, NOT SOURCE_START.
# Uses token IDs directly to avoid tokenizer interpretation issues.
#
# Supported representation types:
#   residual_before  - Residual stream state BEFORE each decoder block (input to block)
#   after_attention  - Hidden state AFTER attention, before FFN
#   after_block      - Hidden state AFTER entire decoder block (attention + FFN + residuals) [default]
#   ffn_gate         - FFN gate projection activations
#   ffn_up           - FFN up projection activations

set -e

echo "=========================================="
echo "Source Token Representation Extraction"
echo "=========================================="

# Load shared configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/00_config_env.sh"

# Representation types to extract (default: after_block)
# Options: residual_before, after_attention, after_block, ffn_gate, ffn_up
# Multiple can be specified: REPRESENTATIONS="after_block ffn_gate ffn_up"
REPRESENTATIONS="${REPRESENTATIONS:-after_block}" # ffn_gate ffn_up}"

# Use combined dataset directory
DATASET_DIR="${COMBINED_DATASET_DIR}"

# Output directory for source token representations
# Use same variable name as 02b for consistency
REPRESENTATION_DIR="${REPRESENTATION_DIR:-${MODEL_DIR}/token_representations}"

echo ""
echo "Configuration:"
echo "  Model: $MODEL_DIR"
echo "  Dataset dir: $DATASET_DIR (combined dataset)"
echo "  Output dir: $REPRESENTATION_DIR"
echo "  Representations: $REPRESENTATIONS"
echo ""

# Check if model exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model directory $MODEL_DIR not found!"
    echo "Please run ./12a_combined_model_training.sh first"
    exit 1
fi

# Check if dataset exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Combined dataset directory $DATASET_DIR not found!"
    echo "Please run ./11c_combined_dataset_preparation.sh first"
    exit 1
fi

# Check if vocab_info.json exists
if [ ! -f "$DATASET_DIR/vocab_info.json" ]; then
    echo "Error: vocab_info.json not found in $DATASET_DIR"
    echo "This file is required to identify source tokens"
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

echo "Extracting source token representations..."
python scripts/12b_source_token_extraction.py \
    --model_dir "$MODEL_DIR" \
    --dataset_dir "$DATASET_DIR" \
    --output_dir "$REPRESENTATION_DIR" \
    --representations $REPRESENTATIONS

echo ""
echo "=========================================="
echo "Extraction complete!"
echo "=========================================="
echo ""
echo "Token representations saved to: $REPRESENTATION_DIR"
echo ""
echo "Next step: Run topology analysis with"
echo "  ./04a_topology_analysis.sh"
echo ""


