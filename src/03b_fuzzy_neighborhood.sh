#!/bin/bash

# Compute Fuzzy Neighborhood Distance Matrix for Persistent Homology
#
# This script computes fuzzy neighborhood distance matrices from PCA-reduced
# data. These distance matrices are suitable for persistent homology analysis
# as they capture local neighborhood relationships with controlled connectivity.
#
# How N_NEIGHBORS affects the final distance matrix:
#
# 1. Neighbor selection (Step 2): N_NEIGHBORS determines how many nearest neighbors
#    each point considers. Only these neighbors can have non-zero membership strength.
#
# 2. Local scales (Step 3): With target_entropy='auto', each point's local scale
#    σ_i is computed to achieve entropy = log2(N_NEIGHBORS). This means:
#    - Larger N_NEIGHBORS → larger σ_i → slower exponential decay
#    - Smaller N_NEIGHBORS → smaller σ_i → faster exponential decay
#
# 3. Membership computation (Step 4): Membership strength = exp(-dist / σ_i)
#    - Larger σ_i (from larger N_NEIGHBORS) → memberships extend to larger distances
#    - This creates more connections in the membership matrix
#
# 4. Distance matrix (Step 6): Final distance = -log(membership)
#    - More connections (from larger N_NEIGHBORS) → denser distance matrix
#    - Larger N_NEIGHBORS → typically smaller distances (more connections, higher memberships)
#    - Smaller N_NEIGHBORS → typically larger distances (fewer connections, lower memberships)
#
# Summary: N_NEIGHBORS controls the trade-off between local (small k) and global (large k)
# structure in the final distance matrix, affecting both sparsity and distance magnitudes.

set -e

echo "=========================================="
echo "Fuzzy Neighborhood Distance Matrix Computation"
echo "=========================================="

# Load shared configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/00_config_env.sh"

# Local configuration
PCA_DIR="${PCA_DIR:-./${WORK_DIR}/pca_result}"
OUTPUT_DIR="${OUTPUT_DIR:-./${WORK_DIR}/fuzzy_neighborhood}"

# Fuzzy neighborhood parameters
N_NEIGHBORS="${N_NEIGHBORS:-20}"
METRIC="${METRIC:-cosine}"
SYM_METHOD="${SYM_METHOD:-fuzzy_union}"
EPSILON="${EPSILON:-1e-10}"
SPARSITY_THRESHOLD="${SPARSITY_THRESHOLD:-0.0}"
TARGET_ENTROPY="${TARGET_ENTROPY:-auto}"

echo ""
echo "Configuration:"
echo "  PCA dir: $PCA_DIR"
echo "  Output dir: $OUTPUT_DIR"
echo "  n_neighbors: $N_NEIGHBORS"
echo "  metric: $METRIC"
echo "  sym_method: $SYM_METHOD"
echo "  epsilon: $EPSILON"
echo "  sparsity_threshold: $SPARSITY_THRESHOLD"
echo "  target_entropy: $TARGET_ENTROPY"
echo ""

# Check if PCA results exist
if [ ! -d "$PCA_DIR" ] || [ -z "$(ls -A $PCA_DIR/*_pca.npz 2>/dev/null)" ]; then
    echo "Error: PCA results not found in $PCA_DIR"
    echo "Please run ./03a_pca_analysis.sh first"
    exit 1
fi

echo "Computing fuzzy neighborhood distance matrices..."
python 03b_fuzzy_neighborhood.py \
    --pca_dir "$PCA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --n_neighbors "$N_NEIGHBORS" \
    --metric "$METRIC" \
    --sym_method "$SYM_METHOD" \
    --epsilon "$EPSILON" \
    --sparsity_threshold "$SPARSITY_THRESHOLD" \
    --target_entropy "$TARGET_ENTROPY"

echo ""
echo "=========================================="
echo "Fuzzy neighborhood computation complete!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files per representation:"
echo "  - {key}_fuzzy_dist.npz (fuzzy distance matrix)"
echo "  - {key}_fuzzy_info.json (computation information)"
echo ""
echo "Hyperparameter guidance:"
echo "  n_neighbors: [50, 2000] - Controls connectivity and scale"
echo "    * Typical values: 200-800 for medium-scale, 800-2000 for global structure"
echo "    * Larger k → more connections, captures more global structure"
echo "    * Smaller k → fewer connections, emphasizes local neighborhoods"
echo "  metric: euclidean | cosine | manhattan"
echo "  sym_method: fuzzy_union | average | max | min (fuzzy_union recommended)"
echo "  epsilon: [1e-10, 1e-5] - numerical stability constant"
echo "  sparsity_threshold: [0, 0.01] - optional sparsification"
echo "  target_entropy: auto | float - auto uses log2(n_neighbors)"
echo ""
echo "Next step: Apply UMAP dimensionality reduction:"
echo "  ./03c_umap_analysis.sh (with USE_FUZZY=true)"
echo ""

