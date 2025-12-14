#!/bin/bash

# Train model on combined dataset (target + source tokens)
#
# This script trains a language model on the combined dataset that includes
# both natural language text and source graph tokens. Training parameters
# can be customized via environment variables.

set -e

echo "=========================================="
echo "Combined Model Training"
echo "=========================================="

# Load shared configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/00_config_env.sh"

# Local configuration
CONFIG="${CONFIG:-configs/config_${RUN_NAME}.json}"
OUTPUT_DIR="${OUTPUT_DIR:-./${WORK_DIR}}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
MAX_LENGTH="${MAX_LENGTH:-640}"
SAVE_STEPS="${SAVE_STEPS:-20}"     # should be integer multiple of eval_steps
EVAL_STEPS="${EVAL_STEPS:-20}"
LOGGING_STEPS="${LOGGING_STEPS:-2}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-all}"  # Number of checkpoints to keep, or "all" to save all
USE_CPU="${USE_CPU:-false}"

# Use combined dataset directory
DATASET_DIR="${COMBINED_DATASET_DIR}"

echo ""
echo "Configuration:"
echo "  Model config: $CONFIG"
echo "  Dataset dir: $DATASET_DIR (combined dataset)"
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
echo "  Save total limit: $SAVE_TOTAL_LIMIT"
echo "  Device: $([ "$USE_CPU" = "true" ] && echo "CPU" || echo "GPU")"
if [ -n "$MODEL_DIR" ]; then
    echo "  Model dir (checkpoint source): $MODEL_DIR"
fi
echo ""

# Check if dataset exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Combined dataset directory $DATASET_DIR not found!"
    echo "Please run ./11c_combined_dataset_preparation.sh first"
    exit 1
fi

# Check if MODEL_DIR exists and contains a checkpoint
RESUME_CHECKPOINT=""
if [ -n "$MODEL_DIR" ] && [ -d "$MODEL_DIR" ]; then
    # Check if it's a valid checkpoint (has model files or training state)
    if [ -f "$MODEL_DIR/pytorch_model.bin" ] || \
       [ -f "$MODEL_DIR/model.safetensors" ] || \
       [ -f "$MODEL_DIR/training_state.json" ] || \
       [ -f "$MODEL_DIR/config.json" ]; then
        RESUME_CHECKPOINT="$MODEL_DIR"
        echo "Found checkpoint at: $MODEL_DIR"
        echo "  Will resume training from this checkpoint"
    else
        echo "Warning: MODEL_DIR ($MODEL_DIR) exists but doesn't appear to be a valid checkpoint"
        echo "  Starting training from scratch"
    fi
else
    echo "No checkpoint found, starting training from scratch"
fi

echo ""
echo "Training model on combined dataset..."
echo "  (This includes both natural language and source graph tokens)"

# Build command arguments
TRAIN_ARGS=(
    --dataset_dir "$DATASET_DIR"
    --config "$CONFIG"
    --output_dir "$OUTPUT_DIR"
    --epochs "$EPOCHS"
    --batch_size "$BATCH_SIZE"
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --learning_rate "$LEARNING_RATE"
    --max_length "$MAX_LENGTH"
    --save_steps "$SAVE_STEPS"
    --eval_steps "$EVAL_STEPS"
    --logging_steps "$LOGGING_STEPS"
    --logging_dir "$LOG_DIR"
    --save_total_limit "$SAVE_TOTAL_LIMIT"
)

# Add resume checkpoint if found
if [ -n "$RESUME_CHECKPOINT" ]; then
    TRAIN_ARGS+=(--resume_from_checkpoint "$RESUME_CHECKPOINT")
fi

# Add CPU flag if needed
if [ "$USE_CPU" = "true" ]; then
    TRAIN_ARGS+=(--no_cuda)
fi

# Run training
python scripts/02a_model_training.py "${TRAIN_ARGS[@]}"

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
echo ""
echo "Model saved to: $OUTPUT_DIR/final_model"
echo ""
echo "Next step: Extract source token representations with"
echo "  ./12b_source_token_extraction.sh"
echo ""

