#!/bin/bash

# Train model on prepared dataset
#
# This script trains a language model on the prepared walk sequences.
# Training parameters can be customized via environment variables.

set -e

echo "=========================================="
echo "Model Training"
echo "=========================================="

# Load shared configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/00_config_env.sh"

# Local configuration
CONFIG="${CONFIG:-configs/config_${RUN_NAME}.json}"
DATASET_DIR="${DATASET_DIR:-./${DATA_DIR}/dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-./${WORK_DIR}}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-50}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
MAX_LENGTH="${MAX_LENGTH:-128}"
SAVE_STEPS="${SAVE_STEPS:-200}"
EVAL_STEPS="${EVAL_STEPS:-200}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}}"
USE_CPU="${USE_CPU:-false}"

echo ""
echo "Configuration:"
echo "  Model config: $CONFIG"
echo "  Dataset dir: $DATASET_DIR"
echo "  Output dir: $OUTPUT_DIR"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Gradient accumulation: $GRAD_ACCUM"
echo "  Learning rate: $LEARNING_RATE"
echo "  Max length: $MAX_LENGTH"
echo "  Save steps: $SAVE_STEPS"
echo "  Eval steps: $EVAL_STEPS"
echo "  Logging steps: $LOGGING_STEPS"
echo "  Logging dir: $LOG_DIR"
echo "  Device: $([ "$USE_CPU" = "true" ] && echo "CPU" || echo "GPU")"
echo ""

# Check if dataset exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Dataset directory $DATASET_DIR not found!"
    echo "Please run ./01c_dataset_preparation.sh first"
    exit 1
fi

echo "Training model..."

if [ "$USE_CPU" = "true" ]; then
    python 02a_model_training.py \
        --dataset_dir "$DATASET_DIR" \
        --config "$CONFIG" \
        --output_dir "$OUTPUT_DIR" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --learning_rate "$LEARNING_RATE" \
        --max_length "$MAX_LENGTH" \
        --save_steps "$SAVE_STEPS" \
        --eval_steps "$EVAL_STEPS" \
        --logging_steps "$LOGGING_STEPS" \
        --logging_dir "$LOG_DIR" \
        --no_cuda
else
    python 02a_model_training.py \
        --dataset_dir "$DATASET_DIR" \
        --config "$CONFIG" \
        --output_dir "$OUTPUT_DIR" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --learning_rate "$LEARNING_RATE" \
        --max_length "$MAX_LENGTH" \
        --save_steps "$SAVE_STEPS" \
        --eval_steps "$EVAL_STEPS" \
        --logging_steps "$LOGGING_STEPS" \
        --logging_dir "$LOG_DIR"
fi

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
echo ""
echo "Model saved to: $OUTPUT_DIR/final_model"
echo ""
echo "Next step: Extract activations with"
echo "  ./02b_activation_extraction.sh"
echo ""


