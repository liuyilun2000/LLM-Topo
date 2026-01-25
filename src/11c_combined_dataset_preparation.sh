#!/bin/bash

# Prepare combined dataset from target natural language dataset and source graph walks
#
# This script combines a target dataset (e.g., TinyStories) with source graph walk
# sequences. It extends the tokenizer vocabulary with source tokens and inserts
# source tokens into target sequences following a power law distribution to simulate
# natural language patterns (e.g., days of week, months appearing occasionally).

set -e

echo "=========================================="
echo "Combined Dataset Preparation"
echo "=========================================="

# Load shared configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/00_config_env.sh"

# Local configuration
# HuggingFace cache directory (default: tinystories workspace)
# Set to workspace directory to avoid using ~/.cache
HF_CACHE_DIR=${HF_CACHE_DIR:-"/hkfs/work/workspace/scratch/hgf_mxv5488-tinystories"}

# Target dataset configuration
TARGET_DATASET_NAME=${TARGET_DATASET_NAME:-"roneneldan/TinyStories"}
TARGET_DATASET_CONFIG=${TARGET_DATASET_CONFIG:-""}
TARGET_DATASET_SPLIT=${TARGET_DATASET_SPLIT:-"train"}
TARGET_DATASET_TEXT_FIELD=${TARGET_DATASET_TEXT_FIELD:-"text"}
TARGET_TOKENIZER_NAME=${TARGET_TOKENIZER_NAME:-""}
MAX_TARGET_SAMPLES=${MAX_TARGET_SAMPLES:-""}

# Source dataset (from previous steps)
# DATASET_NAME is loaded from 00_config_env.sh
SOURCE_CSV=${SOURCE_CSV:-./${DATA_DIR}/sequences/walks_${DATASET_NAME}.csv}

# Output directory
COMBINED_DATASET_DIR=${COMBINED_DATASET_DIR:-./${DATA_DIR}/combined_dataset}

# Insertion parameters
# Power law distribution: most sequences have few source tokens, few have many
MIN_SOURCE_PER_SEQ=${MIN_SOURCE_PER_SEQ:-4}
MAX_SOURCE_PER_SEQ=${MAX_SOURCE_PER_SEQ:-""}
POWER_LAW_ALPHA=${POWER_LAW_ALPHA:-2.0}
SOURCE_RATIO=${SOURCE_RATIO:-0}
MAX_LENGTH=${MAX_LENGTH:-512}
SOURCE_TOKEN_PREFIX=${SOURCE_TOKEN_PREFIX:-"<GRAPH_"}
SOURCE_TOKEN_START=${SOURCE_TOKEN_START:-"<SOURCE_START>"}
SEED=${SEED:-42}

echo ""
echo "Configuration:"
echo "  HuggingFace cache dir: ${HF_CACHE_DIR}"
echo "  Target dataset: ${TARGET_DATASET_NAME}"
if [ -n "${TARGET_DATASET_CONFIG}" ]; then
    echo "  Dataset config: ${TARGET_DATASET_CONFIG}"
fi
echo "  Target dataset split: ${TARGET_DATASET_SPLIT}"
if [ -n "${MAX_TARGET_SAMPLES}" ]; then
    echo "  Max target samples: ${MAX_TARGET_SAMPLES}"
fi
echo "  Source CSV: ${SOURCE_CSV}"
echo "  Output dir: ${COMBINED_DATASET_DIR}"
echo ""
echo "Insertion parameters:"
echo "  Distribution: Power law (alpha=${POWER_LAW_ALPHA})"
echo "  Min source tokens per seq: ${MIN_SOURCE_PER_SEQ}"
if [ -n "${MAX_SOURCE_PER_SEQ}" ]; then
    echo "  Max source tokens per seq: ${MAX_SOURCE_PER_SEQ}"
else
    echo "  Max source tokens per seq: unlimited"
fi
echo "  Source sequence ratio: ${SOURCE_RATIO}"
echo "  Max sequence length: ${MAX_LENGTH}"
echo "  Source token prefix: ${SOURCE_TOKEN_PREFIX}"
echo "  Source token start: ${SOURCE_TOKEN_START}"
echo "  Seed: ${SEED}"
echo ""

# Check if source CSV exists
if [ ! -f "$SOURCE_CSV" ]; then
    echo "Error: Source CSV file not found: $SOURCE_CSV"
    echo "Please run ./01b_sequence_generation.sh first to create source walks."
    exit 1
fi

# Build command
CMD="python ../scripts/11c_combined_dataset_preparation.py \
    --target_dataset_name \"${TARGET_DATASET_NAME}\" \
    --target_dataset_split \"${TARGET_DATASET_SPLIT}\" \
    --target_dataset_text_field \"${TARGET_DATASET_TEXT_FIELD}\" \
    --source_csv \"${SOURCE_CSV}\" \
    --output_dir \"${COMBINED_DATASET_DIR}\" \
    --min_source_per_seq ${MIN_SOURCE_PER_SEQ} \
    --power_law_alpha ${POWER_LAW_ALPHA} \
    --source_ratio ${SOURCE_RATIO} \
    --max_length ${MAX_LENGTH} \
    --source_token_prefix \"${SOURCE_TOKEN_PREFIX}\" \
    --source_token_start \"${SOURCE_TOKEN_START}\" \
    --seed ${SEED}"

# Add optional arguments
if [ -n "${TARGET_DATASET_CONFIG}" ]; then
    CMD="$CMD --target_dataset_config \"${TARGET_DATASET_CONFIG}\""
fi

if [ -n "${TARGET_TOKENIZER_NAME}" ]; then
    CMD="$CMD --target_tokenizer_name \"${TARGET_TOKENIZER_NAME}\""
fi

if [ -n "${MAX_TARGET_SAMPLES}" ]; then
    CMD="$CMD --max_target_samples ${MAX_TARGET_SAMPLES}"
fi

if [ -n "${MAX_SOURCE_PER_SEQ}" ]; then
    CMD="$CMD --max_source_per_seq ${MAX_SOURCE_PER_SEQ}"
fi

# Set HuggingFace cache environment variables to use workspace directory
# This ensures zero increase in local ~/.cache
export HF_HOME="${HF_CACHE_DIR}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_CACHE_DIR}/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}/.cache/huggingface/transformers"

# Create cache directories if they don't exist
mkdir -p "${HF_DATASETS_CACHE}"
mkdir -p "${TRANSFORMERS_CACHE}"

echo ""
echo "HuggingFace cache configuration:"
echo "  HF_HOME: ${HF_HOME}"
echo "  HF_DATASETS_CACHE: ${HF_DATASETS_CACHE}"
echo "  TRANSFORMERS_CACHE: ${TRANSFORMERS_CACHE}"
echo ""

echo "Preparing combined dataset..."
eval $CMD

echo ""
echo "=========================================="
echo "Combined dataset preparation complete!"
echo "=========================================="
echo ""
echo "Combined dataset saved to: ${COMBINED_DATASET_DIR}"
echo ""
echo "The dataset includes:"
echo "  - Target sequences with source tokens inserted"
echo "  - Pure source sequences"
echo "  - Extended tokenizer with source tokens"
echo ""
echo "Next step: Train model with combined dataset using"
echo "  ./02a_model_training.sh"
echo "  (Set DATASET_DIR=${COMBINED_DATASET_DIR})"
echo ""

