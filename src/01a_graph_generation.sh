#!/bin/bash

# Generate graph representation for topology analysis
#
# This script generates the graph structure using polygon-based repulsive force method.
# It creates nodes, edges, adjacency matrix, coordinates, and distance matrix
# for the configured topology and saves them to files for later use in random walk generation.

set -e

echo "=========================================="
echo "Graph Generation (Manifold)"
echo "=========================================="

# Load shared configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/00_config_env.sh"

# Variables N_TOTAL, DENSITY_FACTOR, ITERS, PLOT_INTERVAL, TOPOLOGY_RULE are loaded from 00_config_env.sh
# N is automatically computed from TOPOLOGY_RULE in 00_config_env.sh
# They can be overridden via environment variables before running this script

echo ""
echo "Configuration:"
echo "  Topology prefix: ${TOPOLOGY_PREFIX}"
echo "  Topology rule: ${TOPOLOGY_RULE}"
echo "  Total points: ${N_TOTAL}"
echo "  Density factor: ${DENSITY_FACTOR}"
echo "  Iters: ${ITERS}"
echo "  Plot interval: ${PLOT_INTERVAL}"
echo "  Output directory: ${GRAPH_DIR}"
echo "  Dataset: ${DATASET_NAME}"
echo ""

# Construct dataset name based on topology rule and parameters
# Note: The Python script will handle the actual dataset name construction
# This is just for display purposes
echo "Generating graph representation..."

# Build command arguments
CMD_ARGS=(
    --iters "${ITERS}"
    --plot_interval "${PLOT_INTERVAL}"
    --topology "${TOPOLOGY_RULE}"
    --prefix "${TOPOLOGY_PREFIX}"
    --output_dir "${GRAPH_DIR}"
)

# Add N_total if set (not empty)
if [ -n "${N_TOTAL}" ]; then
    CMD_ARGS+=(--N_total "${N_TOTAL}")
fi

# Add density_factor if set (not empty) - only used if N_total is not provided
if [ -n "${DENSITY_FACTOR}" ]; then
    CMD_ARGS+=(--density_factor "${DENSITY_FACTOR}")
fi

python ../scripts/01a_graph_generation.py "${CMD_ARGS[@]}"

echo ""
echo "Graph generation complete!"
echo ""
echo "Graph files saved to: ${GRAPH_DIR}"
echo "  - A_*_labeled.csv (adjacency matrix with labels)"
echo "  - A_*.npy (adjacency matrix)"
echo "  - nodes_*.csv (node information)"
echo "  - coords_*.csv (3D coordinates)"
echo "  - coords_*.npy (coordinates)"
echo "  - distance_matrix_*.npy (shortest path distances)"
echo "  - graph_info_*.json (metadata)"
echo ""
echo "Next step: Generate random walks with"
echo "  ./01b_sequence_generation.sh"
echo ""
