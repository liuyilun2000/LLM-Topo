#!/bin/bash

# View examples from combined dataset
#
# This script displays sample sequences from the combined dataset,
# showing how source tokens are inserted into natural language text.

set -e

echo "=========================================="
echo "Combined Dataset Viewer"
echo "=========================================="

# Load shared configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/00_config_env.sh"

# Local configuration
NUM_EXAMPLES=${NUM_EXAMPLES:-10}
SPLIT=${SPLIT:-train}
MAX_DISPLAY_LENGTH=${MAX_DISPLAY_LENGTH:-2000}

# Use combined dataset directory
DATASET_DIR="${COMBINED_DATASET_DIR}"

echo ""
echo "Configuration:"
echo "  Dataset dir: $DATASET_DIR"
echo "  Split: $SPLIT"
echo "  Number of examples: $NUM_EXAMPLES"
echo "  Max display length: $MAX_DISPLAY_LENGTH"
echo ""

# Check if dataset exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Combined dataset directory $DATASET_DIR not found!"
    echo "Please run ./11c_combined_dataset_preparation.sh first"
    exit 1
fi

echo "Viewing dataset examples..."
python ../scripts/12c_view_combined_dataset.py \
    --dataset_dir "$DATASET_DIR" \
    --num_examples "$NUM_EXAMPLES" \
    --split "$SPLIT" \
    --max_length "$MAX_DISPLAY_LENGTH"

echo ""
echo "=========================================="
echo "View complete!"
echo "=========================================="
echo ""

