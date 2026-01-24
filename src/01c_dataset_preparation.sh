#!/bin/bash

# Prepare dataset from CSV for model training
#
# This script converts the generated walk CSV into a HuggingFace dataset
# format suitable for training, with train/validation splits.

set -e

echo "=========================================="
echo "Dataset Preparation"
echo "=========================================="

# Load shared configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/00_config_env.sh"

# Local configuration
# Construct dataset name to match sequence generation output
# Use prefix + topology rule: {PREFIX}_{TOPOLOGY_RULE}_n{N}_k{K_EDGE}_iter{ITERS}
DATASET_NAME_PYTHON="${TOPOLOGY_PREFIX}_${TOPOLOGY_RULE}_n${N}_k${K_EDGE}_iter${ITERS}"
INPUT_CSV=${INPUT_CSV:-./${DATA_DIR}/sequences/walks_${DATASET_NAME_PYTHON}.csv}
TRAIN_SPLIT=${TRAIN_SPLIT:-0.95}

echo ""
echo "Configuration:"
echo "  Input CSV: $INPUT_CSV"
echo "  Output dir: $DATASET_DIR"
echo "  Train split: $TRAIN_SPLIT"
echo ""

# Check if input CSV exists
if [ ! -f "$INPUT_CSV" ]; then
    echo "Error: Input CSV file not found: $INPUT_CSV"
    echo "Please run ./01b_sequence_generation.sh first"
    exit 1
fi

echo "Preparing dataset..."
python ../scripts/01c_dataset_preparation.py \
    --input_csv "$INPUT_CSV" \
    --output_dir "$DATASET_DIR" \
    --train_split "$TRAIN_SPLIT"

echo ""
echo "=========================================="
echo "Dataset preparation complete!"
echo "=========================================="
echo ""
echo "Dataset saved to: $DATASET_DIR"
echo ""
echo "Next step: Train model with"
echo "  ./02a_model_training.sh"
echo ""

