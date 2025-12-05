#!/bin/bash

# ============================================================================
# Visualize Persistence Bars Across Checkpoints
# ============================================================================
# This script visualizes how the top k-longest persistence bars vary across
# training checkpoints for one or more specified representations.
#
# Usage:
#   ./05_visualize_persistence_across_checkpoints.sh [OPTIONS]
#
# Options:
#   --representation REP [REP ...]  Representation name(s) (required, can specify multiple)
#                                   Examples: final_hidden, input_embeds, layer_0_after_block
#   --k K1 [K2 ...]                Top k values for each dimension
#                                   Single value applies to all dimensions
#   --k-H0 K                       Top k for H0 dimension
#   --k-H1 K                       Top k for H1 dimension
#   --k-H2 K                       Top k for H2 dimension
#   --output PATH                  Output plot path (optional, only used for single representation)
# ============================================================================

set -e

# Load shared configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/00_config_env.sh"

# Default values
REPRESENTATIONS=("final_hidden" "input_embeds" \
    "layer_0_after_block" "layer_0_ffn_up" "layer_0_ffn_gate" \
    "layer_1_after_block" "layer_1_ffn_up" "layer_1_ffn_gate" \
    "layer_2_after_block" "layer_2_ffn_up" "layer_2_ffn_gate" \
    "layer_3_after_block" "layer_3_ffn_up" "layer_3_ffn_gate" \
    "layer_4_after_block" "layer_4_ffn_up" "layer_4_ffn_gate" \
    "layer_5_after_block" "layer_5_ffn_up" "layer_5_ffn_gate")
K_VALUES=()
K_H0="1"
K_H1="1"
K_H2="1"
OUTPUT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --representation)
            shift
            # Collect all representation names until next option
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                REPRESENTATIONS+=("$1")
                shift
            done
            ;;
        --k)
            shift
            K_VALUES=()
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                K_VALUES+=("$1")
                shift
            done
            ;;
        --k-H0)
            K_H0="$2"
            shift 2
            ;;
        --k-H1)
            K_H1="$2"
            shift 2
            ;;
        --k-H2)
            K_H2="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --representation REP [REP ...] [OPTIONS]"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ ${#REPRESENTATIONS[@]} -eq 0 ]; then
    echo "Error: --representation is required (can specify multiple)"
    echo ""
    echo "Usage: $0 --representation REP [REP ...] [OPTIONS]"
    echo ""
    echo "Examples:"
    echo "  $0 --representation final_hidden --k 5"
    echo "  $0 --representation input_embeds --k-H0 10 --k-H1 5 --k-H2 3"
    echo "  $0 --representation layer_0_after_block --k 5 3 2"
    echo "  $0 --representation final_hidden input_embeds layer_0_after_block --k 1 2 1"
    exit 1
fi

# Validate output path (only makes sense for single representation)
if [ ${#REPRESENTATIONS[@]} -gt 1 ] && [ -n "$OUTPUT" ]; then
    echo "Warning: --output is ignored when multiple representations are specified"
    echo "         Each representation will use its default output path"
    OUTPUT=""
fi

# Display configuration
echo "============================================================================"
echo "Visualize Persistence Bars Across Checkpoints"
echo "============================================================================"
echo ""
echo "Configuration:"
echo "  Topology: ${TOPOLOGY}"
echo "  Grid size: ${H}x${W}"
echo "  Dataset: ${DATASET_NAME}"
echo "  Run name: ${RUN_NAME}"
echo "  Work directory: ${WORK_DIR}"
echo "  Representations: ${REPRESENTATIONS[*]}"
echo ""

if [ ${#K_VALUES[@]} -gt 0 ]; then
    echo "  K values: ${K_VALUES[*]}"
fi

if [ -n "$K_H0" ]; then
    echo "  K for H0: ${K_H0}"
fi

if [ -n "$K_H1" ]; then
    echo "  K for H1: ${K_H1}"
fi

if [ -n "$K_H2" ]; then
    echo "  K for H2: ${K_H2}"
fi

echo ""

# Python script path
PYTHON_SCRIPT="${SCRIPT_DIR}/05_visualize_persistence_across_checkpoints.py"

# Process each representation
TOTAL=${#REPRESENTATIONS[@]}
CURRENT=0

for REP in "${REPRESENTATIONS[@]}"; do
    CURRENT=$((CURRENT + 1))
    
    echo "============================================================================"
    echo "Processing representation ${CURRENT}/${TOTAL}: ${REP}"
    echo "============================================================================"
    echo ""
    
    # Build Python command base
    PYTHON_CMD="python3 ${PYTHON_SCRIPT} --work-dir ${WORK_DIR} --representation ${REP}"
    
    # Add k values
    if [ ${#K_VALUES[@]} -gt 0 ]; then
        PYTHON_CMD="${PYTHON_CMD} --k ${K_VALUES[*]}"
    fi
    
    if [ -n "$K_H0" ]; then
        PYTHON_CMD="${PYTHON_CMD} --k-H0 ${K_H0}"
    fi
    
    if [ -n "$K_H1" ]; then
        PYTHON_CMD="${PYTHON_CMD} --k-H1 ${K_H1}"
    fi
    
    if [ -n "$K_H2" ]; then
        PYTHON_CMD="${PYTHON_CMD} --k-H2 ${K_H2}"
    fi
    
    # Add output path only for single representation
    if [ ${#REPRESENTATIONS[@]} -eq 1 ] && [ -n "$OUTPUT" ]; then
        PYTHON_CMD="${PYTHON_CMD} --output ${OUTPUT}"
    fi
    
    # Run Python script
    if ${PYTHON_CMD}; then
        echo ""
        echo "✓ Successfully processed: ${REP}"
    else
        echo ""
        echo "✗ Failed to process: ${REP}"
        exit 1
    fi
    
    echo ""
done

echo "============================================================================"
echo "Visualization Complete"
echo "============================================================================"
echo ""
echo "Processed ${TOTAL} representation(s):"
for REP in "${REPRESENTATIONS[@]}"; do
    echo "  - ${REP}"
done
echo ""

