#!/bin/bash

# ============================================================================
# Run Full Pipeline - Execute All Stages with Default Settings
# ============================================================================
# This script runs the complete pipeline from Stage 01 to Stage 04:
#   Stage 01: Data Generation (graph, sequences, dataset)
#   Stage 02: Model Training and Representation Extraction
#   Stage 03: Dimensionality Reduction (PCA, fuzzy neighborhood, UMAP)
#   Stage 04: Topology Analysis (persistent homology, barcodes)
#
# All stages use their default settings as defined in individual scripts.
# Configuration can be overridden via environment variables before running.
# ============================================================================

set -e

echo "============================================================================"
echo "Full Pipeline Execution"
echo "============================================================================"
echo ""

# Load shared configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/00_config_env.sh"

echo "Pipeline Configuration:"
echo "  Topology rule: ${TOPOLOGY_RULE}"
echo "  Total points: ${N_TOTAL}"
echo "  Iters: ${ITERS}"
echo "  Dataset: ${DATASET_NAME}"
echo "  Run name: ${RUN_NAME}"
echo ""
echo "All stages will run with default settings."
echo "To override settings, set environment variables before running this script."
echo ""
read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Pipeline cancelled."
    exit 0
fi

echo ""
echo "============================================================================"
echo "Starting Full Pipeline Execution"
echo "============================================================================"
echo ""

# Track start time
START_TIME=$(date +%s)

# ============================================================================
# Stage 01: Data Generation
# ============================================================================
echo ""
echo "============================================================================"
echo "STAGE 01: Data Generation"
echo "============================================================================"
echo ""

echo "Step 1.1: Graph Generation..."
./01a_graph_generation.sh

echo ""
echo "Step 1.2: Sequence Generation..."
./01b_sequence_generation.sh

echo ""
echo "Step 1.3: Dataset Preparation..."
./01c_dataset_preparation.sh

echo ""
echo "✓ Stage 01 Complete: Data Generation"
echo ""

# ============================================================================
# Stage 02: Model Training and Representation Extraction
# ============================================================================
echo ""
echo "============================================================================"
echo "STAGE 02: Model Training and Representation Extraction"
echo "============================================================================"
echo ""

echo "Step 2.1: Model Training..."
./02a_model_training.sh

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
./03d_umap_analysis.sh

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

# ============================================================================
# Pipeline Complete
# ============================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "============================================================================"
echo "Pipeline Execution Complete!"
echo "============================================================================"
echo ""
echo "Total execution time: ${MINUTES}m ${SECONDS}s"
echo ""
echo "Results Summary:"
echo "  Dataset: ${DATASET_NAME}"
echo "  Run: ${RUN_NAME}"
echo "  Results location: results/${DATASET_NAME}/${RUN_NAME}/"
echo ""
echo "Generated outputs:"
echo "  - Graph structures: results/${DATASET_NAME}/graph/"
echo "  - Sequences: results/${DATASET_NAME}/sequences/"
echo "  - Dataset: results/${DATASET_NAME}/dataset/"
echo "  - Trained model: results/${DATASET_NAME}/${RUN_NAME}/final_model/"
echo "  - Representations: results/${DATASET_NAME}/${RUN_NAME}/token_representations/"
echo "  - PCA results: results/${DATASET_NAME}/${RUN_NAME}/pca_result/"
echo "  - Fuzzy neighborhood: results/${DATASET_NAME}/${RUN_NAME}/fuzzy_neighborhood/"
echo "  - UMAP results: results/${DATASET_NAME}/${RUN_NAME}/umap_result_*/"
echo "  - Topology analysis: results/${DATASET_NAME}/${RUN_NAME}/topology_analysis/"
echo "  - Persistence barcodes: results/${DATASET_NAME}/${RUN_NAME}/persistence_barcode/"
echo ""
echo "============================================================================"

