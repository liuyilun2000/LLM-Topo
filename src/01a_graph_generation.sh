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

# Local configuration - polygon-based parameters
# These can be overridden via environment variables
N=${N:-2}                    # n in 2n-polygon
K_EDGE=${K_EDGE:-25}         # Points per edge
ITERS=${ITERS:-200}          # Relaxation iterations

# Topology gluing rule: use capital letters for reversed edges (A=a^-1, B=b^-1, etc.)
# Examples: "abAB" for torus, "abAb" for Klein bottle
TOPOLOGY_RULE=${TOPOLOGY_RULE:-"abAB"}

echo ""
echo "Configuration:"
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
python scripts/01a_graph_generation.py \
    --n ${N} \
    --K_edge ${K_EDGE} \
    --iters ${ITERS} \
    --topology "${TOPOLOGY_RULE}" \
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
