#!/bin/bash

# ============================================================================
# Graph Generation for Topological Manifolds
# ============================================================================
# This script generates graph structures from fundamental polygons with
# edge-gluing rules, representing topological surfaces.
#
# Approach:
#   1. Create a regular 2n-polygon (fundamental polygon)
#   2. Sample points uniformly inside (no fixed boundary points)
#   3. Use repulsive force evolution with periodic boundary conditions
#   4. Build Voronoi graph in tiled space to capture cross-boundary edges
#
# Output files (saved to GRAPH_DIR):
#   - A_{name}.npy, A_{name}_labeled.csv : Adjacency matrix
#   - nodes_{name}.csv : Node information with degrees
#   - coords_{name}.csv, coords_{name}.npy : 2D coordinates
#   - distance_matrix_{name}.npy : Shortest path distances
#   - graph_info_{name}.json : Metadata
#   - graph_visualization.png : Final graph visualization
#   - evolution_plots/ : Evolution snapshots
#
# Usage:
#   ./01a_graph_generation.sh                    # Use config from 00_config_env.sh
#   TOPOLOGY_RULE=abAB ./01a_graph_generation.sh # Override topology
#   N_TOTAL=500 ./01a_graph_generation.sh        # Override point count
# ============================================================================

set -e

echo ""
echo "=========================================="
echo "  Graph Generation (Topological Manifold)"
echo "=========================================="
echo ""

# Load shared configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/00_config_env.sh"

# ============================================================================
# Display Configuration
# ============================================================================

echo "Configuration:"
echo "  ─────────────────────────────────────────"
echo "  Topology"
echo "    Rule:           ${TOPOLOGY_RULE}"
echo "    Prefix:         ${TOPOLOGY_PREFIX}"
echo "    Polygon edges:  $(echo -n ${TOPOLOGY_RULE} | wc -c | tr -d ' ')"
echo ""
echo "  Evolution"
echo "    Points:         ${N_TOTAL}"
echo "    Iterations:     ${ITERS}"
echo "    Step size:      ${STEP_SIZE}"
echo "    Tiling layers:  ${TILING_LAYERS}"
echo ""
echo "  Visualization"
echo "    Plot interval:  every ${PLOT_INTERVAL} iterations"
echo ""
echo "  Output"
echo "    Dataset name:   ${DATASET_NAME}"
echo "    Graph dir:      ${GRAPH_DIR}"
echo "  ─────────────────────────────────────────"
echo ""

# Create output directory
mkdir -p "${GRAPH_DIR}"

# ============================================================================
# Build Command Arguments
# ============================================================================

CMD_ARGS=(
    --topology "${TOPOLOGY_RULE}"
    --prefix "${TOPOLOGY_PREFIX}"
    --N_total "${N_TOTAL}"
    --iters "${ITERS}"
    --step_size "${STEP_SIZE}"
    --tiling_layers "${TILING_LAYERS}"
    --plot_interval "${PLOT_INTERVAL}"
    --output_dir "${GRAPH_DIR}"
)

# Add optional seed if set
if [ -n "${SEED}" ]; then
    CMD_ARGS+=(--seed "${SEED}")
fi

# ============================================================================
# Run Graph Generation
# ============================================================================

echo "Starting graph generation..."
echo ""

python "${SCRIPT_DIR}/../scripts/01a_graph_generation.py" "${CMD_ARGS[@]}"

echo ""
echo "=========================================="
echo "  Graph Generation Complete!"
echo "=========================================="
echo ""
echo "Output files saved to: ${GRAPH_DIR}"
echo ""
echo "  Data files:"
echo "    - A_${DATASET_NAME}.npy              (adjacency matrix)"
echo "    - A_${DATASET_NAME}_labeled.csv      (adjacency with labels)"
echo "    - nodes_${DATASET_NAME}.csv          (node info: id, degree)"
echo "    - coords_${DATASET_NAME}.npy         (2D coordinates)"
echo "    - coords_${DATASET_NAME}.csv         (coordinates CSV)"
echo "    - distance_matrix_${DATASET_NAME}.npy (shortest paths)"
echo "    - graph_info_${DATASET_NAME}.json    (metadata)"
echo ""
echo "  Visualizations:"
echo "    - graph_visualization.png            (final graph view)"
echo "    - evolution_plots/                   (evolution snapshots)"
echo ""
echo "Next steps:"
echo "  1. View evolution: ls ${GRAPH_DIR}/evolution_plots/"
echo "  2. Check graph info: cat ${GRAPH_DIR}/graph_info_${DATASET_NAME}.json"
echo "  3. Generate random walks: ./01b_sequence_generation.sh"
echo ""
