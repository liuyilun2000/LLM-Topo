#!/bin/bash

# Generate random walk sequences from saved graph representation
#
# This script generates random walk sequences on a pre-computed graph structure.
# The graph representation should be generated first using 01a_graph_generation.sh.

set -e

echo "=========================================="
echo "Random Walk Generation (Manifold)"
echo "=========================================="

# Load shared configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/00_config_env.sh"

# N_TOTAL, ITERS, TOPOLOGY_RULE, SEQUENCE_DIR, DATA_DIR, DATASET_NAME from 00_config_env.sh
# Walk parameters: MAX_LENGTH, MAX_SEQS, MIN_VISITS_PER_NODE, NO_REPEAT_WINDOW, RESTART_PROB, TEMPERATURE, SEED from config
mkdir -p "${SEQUENCE_DIR}"

OUT=${OUT:-${SEQUENCE_DIR}/walks_${DATASET_NAME}.csv}
COUNTS_OUT=${COUNTS_OUT:-${SEQUENCE_DIR}/visit_counts_${DATASET_NAME}.csv}

echo ""
echo "Configuration:"
echo "  Topology prefix: ${TOPOLOGY_PREFIX}"
echo "  Topology rule: ${TOPOLOGY_RULE}"
echo "  Total points: ${N_TOTAL}"
echo "  Iters: ${ITERS}"
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
python ../scripts/01b_sequence_generation.py \
    --graph_dir "${GRAPH_DIR}" \
    --topology "${TOPOLOGY_RULE}" \
    --prefix "${TOPOLOGY_PREFIX}" \
    --N_total ${N_TOTAL} \
    --iters ${ITERS} \
    --max_length ${MAX_LENGTH} \
    --max_seqs ${MAX_SEQS} \
    --min_visits_per_node ${MIN_VISITS_PER_NODE} \
    --no_repeat_window ${NO_REPEAT_WINDOW} \
    --restart_prob ${RESTART_PROB} \
    --temperature ${TEMPERATURE} \
    --seed ${SEED} \
    --out "${OUT}" \
    --counts_out "${COUNTS_OUT}"

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
