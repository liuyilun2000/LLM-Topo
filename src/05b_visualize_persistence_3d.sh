#!/bin/bash

# ============================================================================
# Visualize Persistence Features in 3D (Steps x Layers)
# ============================================================================
# This script visualizes the emergence of topological features across
# checkpoints (steps) and layers (representations) as 3D plots.
#
# Usage:
#   ./05b_visualize_persistence_3d.sh [OPTIONS]
#
# Options:
#   --representation REP [REP ...]  Representation name(s) (required, can specify multiple)
#                                   Examples: final_hidden, input_embeds, layer_0_after_block
#   --k K1 [K2 ...]                Top k values for each dimension
#                                   Single value applies to all dimensions
#   --k-H0 K                       Top k for H0 dimension
#   --k-H1 K                       Top k for H1 dimension
#   --k-H2 K                       Top k for H2 dimension
# ============================================================================

set -e

# Load shared configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/00_config_env.sh"

# Default values
REPRESENTATIONS=("final_hidden" "input_embeds" \
    "layer_0_after_block" "layer_1_after_block" "layer_2_after_block" "layer_3_after_block" "layer_4_after_block" "layer_5_after_block" \
    "layer_6_after_block" "layer_7_after_block" "layer_8_after_block" "layer_9_after_block" "layer_10_after_block" "layer_11_after_block" )

K_VALUES=()
K_H0="1"
K_H1="2"
K_H2="1"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --representation)
            shift
            # Clear defaults and collect all representation names until next option
            REPRESENTATIONS=()
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
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --representation REP [REP ...] [OPTIONS]"
            exit 1
            ;;
    esac
done

# Note: If no representations specified, defaults will be used
# (defaults are set at the top of the script)

# Display configuration
echo "============================================================================"
echo "Visualize Persistence Features in 3D (Steps x Layers)"
echo "============================================================================"
echo ""
echo "Configuration:"
echo "  Topology rule: ${TOPOLOGY_RULE}"
echo "  Total points: ${N_TOTAL}"
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
PYTHON_SCRIPT="${SCRIPT_DIR}/../scripts/05b_visualize_persistence_3d.py"

# Build Python command
PYTHON_CMD="python3 ${PYTHON_SCRIPT} --work-dir ${WORK_DIR}"

# Add representations
for REP in "${REPRESENTATIONS[@]}"; do
    PYTHON_CMD="${PYTHON_CMD} --representation ${REP}"
done

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

# Run Python script
if ${PYTHON_CMD}; then
    echo ""
    echo "✓ Successfully generated 3D visualizations"
else
    echo ""
    echo "✗ Failed to generate 3D visualizations"
    exit 1
fi

echo ""
echo "============================================================================"
echo "3D Visualization Complete"
echo "============================================================================"
echo ""
