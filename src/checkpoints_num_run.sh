echo ""
echo "Step 2.2: Representation Extraction..."
./02b_representation_extraction.sh

echo "" 
echo "✓ Stage 02 Complete: Model Training and Representation Extraction"
echo ""

# ============================================================================
# Stage 03: Dimensionality Reduction
# ============================================================================
echo ""
echo "============================================================================"
echo "STAGE 03: Dimensionality Reduction"
echo "============================================================================"
echo ""

echo "Step 3.1: PCA Analysis..."
./03a_pca_analysis.sh

echo ""
echo "Step 3.2: Fuzzy Neighborhood..."
./03b_fuzzy_neighborhood.sh

echo ""
echo "Step 3.3: UMAP Analysis..."
./03c_umap_analysis.sh

echo ""
echo "✓ Stage 03 Complete: Dimensionality Reduction"
echo ""

# ============================================================================
# Stage 04: Topology Analysis
# ============================================================================
echo ""
echo "============================================================================"
echo "STAGE 04: Topology Analysis"
echo "============================================================================"
echo ""

echo "Step 4.1: Topology Analysis (Persistent Homology)..."
./04a_topology_analysis.sh

echo ""
echo "Step 4.2: Persistence Barcode Visualization..."
./04b_persistence_barcode.sh

echo ""
echo "✓ Stage 04 Complete: Topology Analysis"
echo ""