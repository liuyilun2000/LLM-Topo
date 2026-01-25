#!/bin/bash

# ============================================================================
# Shared Configuration for TOPO Analysis Pipeline
# ============================================================================
# Source this file in each script: source 00_config_env.sh
#
# This file provides centralized configuration for:
#   - Topology Gluing Rule (N is automatically computed from TOPOLOGY_RULE)
#   - Manifold Geometry (N_TOTAL, DENSITY_FACTOR, ITERS, PLOT_INTERVAL)
#   - Dataset naming (DATASET_NAME is automatically generated)
#   - Run naming
# ============================================================================

# Topology configuration
# Topology rule string (Gluing Rules):
#   IMPORTANT: Input format uses capital letters for reversed edges.
#   - Lowercase letters (a, b, c, ...) = normal edges
#   - Uppercase letters (A, B, C, ...) = reversed edges (mathematical notation: a^-1, b^-1, c^-1, ...)
#   
#   Examples (input the capital letter form directly):
#   - Sphere                        : abb^-1a^-1   →  abBA (4 edges → n=2)
#   - Sphere (alternative)          : aa^-1bb^-1   →  aAbB (4 edges → n=2)
#   - Torus                         : aba^-1b^-1   →  abAB (4 edges → n=2)
#   - Double torus                  : aba^-1b^-1cdc^-1d^-1  →  abABcdCD (8 edges → n=4)
#   - Projective plane              : abab         →  abab (4 edges → n=2)
#   - Klein Bottle                  : aba^-1b      →  abAb (4 edges → n=2)
#   - Klein Bottle (alternative)    : abab^-1      →  abaB (4 edges → n=2)
#   - custom                  : Any valid gluing string (n = length/2)
export TOPOLOGY_RULE=${TOPOLOGY_RULE:-"abAB"}

export TOPOLOGY_PREFIX="torus"


# Note: N (n in 2n-polygon) is computed internally in 01a_graph_generation.py from TOPOLOGY_RULE
# It is not needed in bash scripts - only the Python code uses it for polygon construction

# Manifold Geometry dimensions
export N_TOTAL=${N_TOTAL:-800}      # Total number of points (all points can move freely, no fixed boundary)
export ITERS=${ITERS:-200}          # Relaxation iterations
export PLOT_INTERVAL=${PLOT_INTERVAL:-2}  # Interval for saving evolution plots (every N iterations)

# Derived configuration - always recomputed from N_TOTAL, ITERS, TOPOLOGY_RULE, TOPOLOGY_PREFIX
# DATASET_NAME is automatically generated with prefix: {PREFIX}_{TOPOLOGY_RULE}_N{N_TOTAL}_iter{ITERS}
# Note: N is only computed internally in 01a_graph_generation.py, not used in dataset naming
if [ -n "${N_TOTAL}" ] && [ -n "${ITERS}" ] && [ -n "${TOPOLOGY_RULE}" ] && [ -n "${TOPOLOGY_PREFIX}" ]; then
    export DATASET_NAME="${TOPOLOGY_PREFIX}_${TOPOLOGY_RULE}_N${N_TOTAL}_iter${ITERS}"
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

