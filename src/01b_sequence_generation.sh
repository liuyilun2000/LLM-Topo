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

# Local configuration - must match graph generation parameters
N=${N:-2}                    # n in 2n-polygon (must match 01a)
K_EDGE=${K_EDGE:-25}         # Points per edge (must match 01a)
ITERS=${ITERS:-200}          # Relaxation iterations (must match 01a)
TOPOLOGY_RULE=${TOPOLOGY_RULE:-"abAB"}  # Topology gluing rule: capital letters for reversed edges (A=a^-1, B=b^-1, etc.), must match 01a

# Walk generation parameters
MAX_LENGTH=${MAX_LENGTH:-128}
MAX_SEQS=${MAX_SEQS:-120000}
MIN_VISITS_PER_NODE=${MIN_VISITS_PER_NODE:-10000000000}
NO_REPEAT_WINDOW=${NO_REPEAT_WINDOW:-32}
RESTART_PROB=${RESTART_PROB:-0}
TEMPERATURE=${TEMPERATURE:-1}
SEED=${SEED:-42}

# Output paths
SEQUENCE_DIR="${SEQUENCE_DIR:-./${DATA_DIR}/sequences}"
mkdir -p "${SEQUENCE_DIR}"

# Construct dataset name (must match what 01a_graph_generation.py produces)
# Use topology rule directly (already in capital letter form)
DATASET_NAME_PYTHON="${TOPOLOGY_RULE}_n${N}_k${K_EDGE}_iter${ITERS}"
OUT=${OUT:-${SEQUENCE_DIR}/walks_${DATASET_NAME_PYTHON}.csv}
COUNTS_OUT=${COUNTS_OUT:-${SEQUENCE_DIR}/visit_counts_${DATASET_NAME_PYTHON}.csv}

echo ""
echo "Configuration:"
echo "  Topology rule: ${TOPOLOGY_RULE}"
echo "  Polygon n: ${N}"
echo "  K_edge: ${K_EDGE}"
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
GRAPH_BASE="${GRAPH_DIR}/A_${DATASET_NAME_PYTHON}_labeled.csv"
if [ ! -f "$GRAPH_BASE" ]; then
    echo "Error: Graph file not found: $GRAPH_BASE"
    echo "Please run ./01a_graph_generation.sh first to create the graph representation."
    exit 1
fi

echo "Generating random walks..."
python scripts/01b_sequence_generation.py \
    --graph_dir "${GRAPH_DIR}" \
    --topology "${TOPOLOGY_RULE}" \
    --n ${N} \
    --K_edge ${K_EDGE} \
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
