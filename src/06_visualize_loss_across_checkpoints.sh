#!/bin/bash

# ============================================================================
# Visualize Training and Evaluation Loss Across Checkpoints
# ============================================================================
# This script visualizes how training loss and evaluation loss vary across
# training checkpoints by loading trainer_state.json from each checkpoint.
#
# Usage:
#   ./06_visualize_loss_across_checkpoints.sh [OPTIONS]
#
# Options:
#   --output PATH    Output plot path (optional, default: work_dir/loss_evolution.pdf)
# ============================================================================

set -e

# Load shared configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/00_config_env.sh"

# Default values
OUTPUT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--output PATH]"
            exit 1
            ;;
    esac
done

# Display configuration
echo "============================================================================"
echo "Visualize Training and Evaluation Loss Across Checkpoints"
echo "============================================================================"
echo ""
echo "Configuration:"
echo "  Topology rule: ${TOPOLOGY_RULE}"
echo "  Polygon n: ${N}"
echo "  K_edge: ${K_EDGE}"
echo "  Iters: ${ITERS}"
echo "  Dataset: ${DATASET_NAME}"
echo "  Run name: ${RUN_NAME}"
echo "  Work directory: ${WORK_DIR}"
echo ""

if [ -n "$OUTPUT" ]; then
    echo "  Output path: ${OUTPUT}"
fi

echo ""

# Python script path
PYTHON_SCRIPT="${SCRIPT_DIR}/06_visualize_loss_across_checkpoints.py"

# Build Python command
PYTHON_CMD="python3 ${PYTHON_SCRIPT} --work-dir ${WORK_DIR}"

if [ -n "$OUTPUT" ]; then
    PYTHON_CMD="${PYTHON_CMD} --output ${OUTPUT}"
fi

# Run Python script
echo "Running visualization..."
echo ""
${PYTHON_CMD}

echo ""
echo "============================================================================"
echo "Visualization Complete"
echo "============================================================================"

