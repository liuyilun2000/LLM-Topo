#!/bin/bash

# Generate graph representation for topology analysis
#
# This script generates the graph structure (nodes, edges, adjacency matrix,
# coordinates, and distance matrix) for the configured topology and saves it
# to files for later use in random walk generation.

set -e

echo "=========================================="
echo "Graph Generation"
echo "=========================================="

# Load shared configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/00_config_env.sh"

# Local configuration
GRAPH_DIR="${GRAPH_DIR:-./${DATA_DIR}/graph}"
NEIGH=${NEIGH:-8}

echo ""
echo "Configuration:"
echo "  Topology: ${TOPOLOGY}"
echo "  Grid size: ${H}x${W}"
echo "  Dataset: ${DATASET_NAME}"
echo "  Neighborhood: ${NEIGH} (4 or 8)"
echo "  Output directory: ${GRAPH_DIR}"
echo ""

echo "Generating graph representation..."
python scripts/01a_graph_generation.py \
    --H ${H} --W ${W} --topology "${TOPOLOGY}" --neigh ${NEIGH} \
    --output_dir "${GRAPH_DIR}"

echo ""
echo "Graph generation complete!"
echo ""
echo "Graph files saved to: ${GRAPH_DIR}"
echo "  - A_${DATASET_NAME}_labeled.csv (adjacency matrix with labels)"
echo "  - A_${DATASET_NAME}.npy (adjacency matrix)"
echo "  - nodes_${DATASET_NAME}.csv (node information)"
echo "  - coords_${DATASET_NAME}.csv (3D coordinates)"
echo "  - coords_${DATASET_NAME}.npy (coordinates)"
echo "  - distance_matrix_${DATASET_NAME}.npy (shortest path distances)"
echo "  - graph_info_${DATASET_NAME}.json (metadata)"
echo ""
echo "Next step: Generate random walks with"
echo "  ./01b_sequence_generation.sh"
echo ""
