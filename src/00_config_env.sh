#!/bin/bash

# ============================================================================
# Shared Configuration for TOPO Analysis Pipeline
# ============================================================================
# Source this file in each script: source 00_config_env.sh
#
# This file provides centralized configuration for:
#   - Manifold Geometry (N, K_EDGE, ITERS)
#   - Topology Gluing Rule
#   - Dataset naming
#   - Run naming
# ============================================================================

# Manifold Geometry dimensions
# Polygon-based parameters for repulsive force point distribution
export N=${N:-2}                    # n in 2n-polygon (2=Square, 3=Hexagon, etc.)
export K_EDGE=${K_EDGE:-25}         # Points per edge (density)
export ITERS=${ITERS:-200}          # Relaxation iterations

# Topology configuration
# Topology rule string (Gluing Rules):
#   IMPORTANT: Input format uses capital letters for reversed edges.
#   - Lowercase letters (a, b, c, ...) = normal edges
#   - Uppercase letters (A, B, C, ...) = reversed edges (mathematical notation: a^-1, b^-1, c^-1, ...)
#   
#   Examples (input the capital letter form directly):
#   - Torus (Standard)        : aba^-1b^-1  →  abAB
#   - Klein Bottle            : abab^-1      →  abAb
#   - Sphere                  : abb^-1a^-1   →  abBA
#   - Double torus            : aba^-1b^-1cdc^-1d^-1  →  abABcdCD
#   - Torus with 1 cross-cap   : aba^-1cdc^-1b^-1      →  abAcB
#   - custom                  : Any valid gluing string matching 2*N edges
export TOPOLOGY_RULE=${TOPOLOGY_RULE:-"abAB"}

# Use topology rule directly as directory name (no conversion needed)
export TOPOLOGY_DIR="$TOPOLOGY_RULE"

# Derived configuration - always recomputed from N, K_EDGE, ITERS, TOPOLOGY_RULE
# DATASET_NAME is automatically generated from topology rule and parameters
if [ -n "${N}" ] && [ -n "${K_EDGE}" ] && [ -n "${ITERS}" ] && [ -n "${TOPOLOGY_DIR}" ]; then
    export DATASET_NAME="${TOPOLOGY_DIR}_n${N}_k${K_EDGE}_iter${ITERS}"
fi

# Run configuration (can be overridden)
export RUN_NAME=${RUN_NAME:-"12M_llama"}

# Working directory structure:
#   - Data-related outputs (graphs, walks, prepared datasets): {DATASET_NAME}/
#   - Run-related outputs (models, representations, analyses): {DATASET_NAME}/{RUN_NAME}/
export WORK_DIR="results/${DATASET_NAME}/${RUN_NAME}"
export DATA_DIR="results/${DATASET_NAME}"
export DATASET_DIR="${DATASET_DIR:-${DATA_DIR}/dataset}"
export COMBINED_DATASET_DIR="${COMBINED_DATASET_DIR:-${DATA_DIR}/combined_dataset}"

# Graph generation parameters
export GRAPH_DIR="${GRAPH_DIR:-./${DATA_DIR}/graph}"
export SEQUENCE_DIR="${SEQUENCE_DIR:-./${DATA_DIR}/sequences}"


#export MODEL_DIR="${MODEL_DIR:-${WORK_DIR}/final_model}" # or checkpoints directory
export MODEL_DIR="${MODEL_DIR:-${WORK_DIR}/checkpoint-7900}"
export REPRESENTATION_DIR="${REPRESENTATION_DIR:-${MODEL_DIR}/token_representations}"
export SOURCE_TOKEN_REPRESENTATION_DIR="${SOURCE_TOKEN_REPRESENTATION_DIR:-${MODEL_DIR}/source_token_representations}"
export PCA_DIR="${PCA_DIR:-${MODEL_DIR}/pca_result}"
export FUZZY_NEIGHBORHOOD_DIR="${FUZZY_NEIGHBORHOOD_DIR:-${MODEL_DIR}/fuzzy_neighborhood}"
export UMAP_DIR="${UMAP_DIR:-${MODEL_DIR}/umap_result}"
export TOPOLOGY_ANALYSIS_DIR="${TOPOLOGY_ANALYSIS_DIR:-${MODEL_DIR}/topology_analysis}"
export PERSISTENCE_BARCODE_DIR="${PERSISTENCE_BARCODE_DIR:-${MODEL_DIR}/persistence_barcode}"

# Note: Individual scripts should set their own SCRIPT_DIR based on where they are
# and use relative paths from that location.

