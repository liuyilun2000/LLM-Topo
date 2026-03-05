#!/bin/bash

# ============================================================================
# Graph Generation for Topological Manifolds
# ============================================================================
# Generates a degree-regular graph on a topological surface (torus, Klein
# bottle, double torus, …) represented as a fundamental polygon with
# edge-gluing rules.
#
# Approach:
#   1. Create a regular 2n-polygon (fundamental polygon)
#   2. Sample N_TOTAL points uniformly inside
#   3. Evolve positions with degree-aware repulsion (live Voronoi each step):
#        - Recompute Voronoi graph every iteration (topology-aware via tiling)
#        - Baseline repulsion (scale=1) applied to all Voronoi neighbours
#        - Over-degree nodes: extra kick (scale=(degree/target)^DEGREE_POWER > 1)
#          applied only to the furthest (degree-target) Voronoi neighbours
#        - Under/at-target nodes: baseline repulsion only
#   4. Final graph = Voronoi neighbours at converged positions
#
# All parameters are read from 00_config_env.sh (sourced below). Override inline:
#   TOPOLOGY_RULE=abAB ./01a_graph_generation.sh
#   N_TOTAL=500 TARGET_DEGREE=6 ./01a_graph_generation.sh
#
# Output files (saved to GRAPH_DIR):
#   - A_{name}.npy, A_{name}_labeled.csv : Adjacency matrix
#   - nodes_{name}.csv                   : Node info (id, degree)
#   - coords_{name}.csv, coords_{name}.npy : 2D coordinates
#   - distance_matrix_{name}.npy         : Shortest-path distances
#   - graph_info_{name}.json             : Metadata (topology, degree stats)
#   - graph_visualization.pdf            : Final graph (domain + tiled view)
#   - evolution_plots/iter_NNN.pdf       : Per-step snapshots with degree histogram
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
echo "    Density factor: ${DENSITY_FACTOR}"
echo "    Max iters:      ${MAX_ITERS}   (stop when delta < ${CONVERGENCE_TOL})"
echo "    Lloyd step:     ${LLOYD_STEP_SIZE}   target_degree: ${TARGET_DEGREE}"
echo "    Noise:         ${NOISE_STRENGTH}  (decay: ${NOISE_DECAY_POWER})"
echo "    Tiling layers: ${TILING_LAYERS}"
echo ""
echo "  Visualization"
echo "    Plot interval:  every ${PLOT_INTERVAL} iterations"
echo "    Log interval:   every ${LOG_INTERVAL} iterations"
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
    --density_factor "${DENSITY_FACTOR}"
    --max_iters "${MAX_ITERS}"
    --convergence_tol "${CONVERGENCE_TOL}"
    --lloyd_step_size "${LLOYD_STEP_SIZE}"
    --target_degree "${TARGET_DEGREE}"
    --noise_strength "${NOISE_STRENGTH}"
    --noise_decay_power "${NOISE_DECAY_POWER}"
    --tiling_layers "${TILING_LAYERS}"
    --plot_interval "${PLOT_INTERVAL}"
    --log_interval "${LOG_INTERVAL}"
    --output_dir "${GRAPH_DIR}"
)
if [ -n "${N_TOTAL}" ]; then
    CMD_ARGS+=(--N_total "${N_TOTAL}")
fi

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
echo "    - graph_visualization.pdf            (final graph view)"
echo "    - evolution_plots/                   (evolution snapshots, PDF)"
echo ""
echo "Next steps:"
echo "  1. View evolution: ls ${GRAPH_DIR}/evolution_plots/"
echo "  2. Check graph info: cat ${GRAPH_DIR}/graph_info_${DATASET_NAME}.json"
echo "  3. Generate random walks: ./01b_sequence_generation.sh"
echo ""
