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

# Variables N, K_EDGE, ITERS, TOPOLOGY_RULE are loaded from 00_config_env.sh
# They can be overridden via environment variables before running this script

echo ""
echo "Configuration:"
echo "  Topology prefix: ${TOPOLOGY_PREFIX}"
echo "  Topology rule: ${TOPOLOGY_RULE}"
echo "  Polygon n: ${N}"
echo "  K_edge: ${K_EDGE}"
echo "  Iters: ${ITERS}"
echo "  Output directory: ${GRAPH_DIR}"
echo "  Dataset: ${DATASET_NAME}"
echo ""

# Construct dataset name based on topology rule and parameters
# Note: The Python script will handle the actual dataset name construction
# This is just for display purposes
echo "Generating graph representation..."
python ../scripts/01a_graph_generation.py \
    --n ${N} \
    --K_edge ${K_EDGE} \
    --iters ${ITERS} \
    --topology "${TOPOLOGY_RULE}" \
    --prefix "${TOPOLOGY_PREFIX}" \
    --output_dir "${GRAPH_DIR}"

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
