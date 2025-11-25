#!/bin/bash

# Generate random walk sequences from saved graph representation
#
# This script generates random walk sequences on a pre-computed graph structure.
# The graph representation should be generated first using 01a_graph_generation.sh.

set -e

echo "=========================================="
echo "Random Walk Generation"
echo "=========================================="

# Load shared configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/00_config_env.sh"

# Local configuration
MAX_LENGTH=${MAX_LENGTH:-128}
MAX_SEQS=${MAX_SEQS:-120000}
MIN_VISITS_PER_NODE=${MIN_VISITS_PER_NODE:-10000000000}
NO_REPEAT_WINDOW=${NO_REPEAT_WINDOW:-32}
RESTART_PROB=${RESTART_PROB:-0}
TEMPERATURE=${TEMPERATURE:-1}
SEED=${SEED:-42}

OUT=${OUT:-./${DATA_DIR}/sequences/walks_${DATASET_NAME}.csv}
COUNTS_OUT=${COUNTS_OUT:-./${DATA_DIR}/sequences/visit_counts_${DATASET_NAME}.csv}

echo ""
echo "Configuration:"
echo "  Topology: ${TOPOLOGY}"
echo "  Grid size: ${H}x${W}"
echo "  Dataset: ${DATASET_NAME}"
echo "  Graph directory: ${GRAPH_DIR}"
echo "  Output: ${OUT}"
echo "  Counts output: ${COUNTS_OUT}"
echo ""
echo "Walk parameters:"
echo "  Max length: ${MAX_LENGTH}"
echo "  Max sequences: ${MAX_SEQS}"
echo "  Min visits per node: ${MIN_VISITS_PER_NODE}"
echo "  No-repeat window: ${NO_REPEAT_WINDOW}"
echo "  Restart probability: ${RESTART_PROB}"
echo "  Temperature: ${TEMPERATURE}"
echo "  Seed: ${SEED}"
echo ""

# Check if graph files exist
GRAPH_BASE="${GRAPH_DIR}/A_${DATASET_NAME}_labeled.csv"
if [ ! -f "$GRAPH_BASE" ]; then
    echo "Error: Graph file not found: $GRAPH_BASE"
    echo "Please run ./01a_graph_generation.sh first to create the graph representation."
    exit 1
fi

echo "Generating random walks..."
python scripts/01b_sequence_generation.py \
    --graph_dir "${GRAPH_DIR}" \
    --topology "${TOPOLOGY}" --H ${H} --W ${W} \
    --max_length ${MAX_LENGTH} --max_seqs ${MAX_SEQS} \
    --min_visits_per_node ${MIN_VISITS_PER_NODE} \
    --no_repeat_window ${NO_REPEAT_WINDOW} \
    --restart_prob ${RESTART_PROB} --temperature ${TEMPERATURE} --seed ${SEED} \
    --out ${OUT} --counts_out ${COUNTS_OUT}

echo ""
echo "Random walk generation complete!"
echo ""
echo "Output files:"
echo "  Walks: ${OUT}"
echo "  Visit counts: ${COUNTS_OUT}"
echo ""
echo "Next step: Prepare dataset with"
echo "  ./01c_dataset_preparation.sh"
echo ""
