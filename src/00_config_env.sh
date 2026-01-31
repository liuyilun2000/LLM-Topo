#!/bin/bash

# ============================================================================
# Shared Configuration for TOPO Analysis Pipeline
# ============================================================================
# Source this file in each script: source 00_config_env.sh
#
# This file provides centralized configuration for:
#   - Topology Gluing Rule (polygon size is auto-computed from rule length)
#   - Evolution Parameters (points, iterations, visualization)
#   - Tiling Configuration (layers for neighbor detection)
#   - Dataset naming (auto-generated from parameters)
#   - Directory structure
# ============================================================================

# ============================================================================
# TOPOLOGY CONFIGURATION
# ============================================================================
# Gluing rule notation:
#   - Lowercase letters (a, b, c, ...) = edges in forward direction
#   - Uppercase letters (A, B, C, ...) = reversed edges (A = a^-1)
#   - Each letter pair defines one gluing (edge identification)
#
# Common topologies:
#   Torus (genus 1):           abAB          (4-gon, 2 gluings)
#   Double Torus (genus 2):    abABcdCD      (8-gon, 4 gluings)
#   Triple Torus (genus 3):    abABcdCDefEF  (12-gon, 6 gluings)
#   Klein Bottle:              abAb or abaB  (4-gon, 2 gluings, non-orientable)
#   Projective Plane:          abab          (4-gon, 2 gluings, non-orientable)
#   Sphere:                    abBA or aAbB  (4-gon, 2 gluings)
#
# The polygon has 2n edges where n = len(rule) / 2
# Example: abABcdCD has 8 characters → 8-gon (octagon)
# ============================================================================

# Select topology (uncomment one or set via environment variable)
# export TOPOLOGY_RULE=${TOPOLOGY_RULE:-"abAB"}        # Torus (default)
# export TOPOLOGY_RULE=${TOPOLOGY_RULE:-"abABcdCD"}      # Double Torus (genus 2)
export TOPOLOGY_RULE=${TOPOLOGY_RULE:-"abABcdCDefEF"} # Triple Torus (genus 3)
# export TOPOLOGY_RULE=${TOPOLOGY_RULE:-"abAb"}        # Klein Bottle
# export TOPOLOGY_RULE=${TOPOLOGY_RULE:-"abab"}        # Projective Plane

# Topology prefix for dataset naming (auto-detected if not set)
# Will be automatically detected from TOPOLOGY_RULE if not specified
export TOPOLOGY_PREFIX=${TOPOLOGY_PREFIX:-"torus"}

# ============================================================================
# EVOLUTION PARAMETERS
# ============================================================================
# These control the repulsive force simulation that distributes points
# uniformly on the topological surface.

# Number of points to sample on the manifold
# More points = finer graph resolution but slower computation
# Recommended: 200-500 for testing, 800-2000 for production
export N_TOTAL=${N_TOTAL:-1200}

# Number of evolution iterations for point distribution
# More iterations = better uniformity but longer runtime
# Recommended: 100-200 for quick tests, 300-500 for production
export ITERS=${ITERS:-200}

# Step size for point movement during evolution
# Smaller = more stable but slower convergence
# Recommended: 0.03-0.1
export STEP_SIZE=${STEP_SIZE:-0.05}

# ============================================================================
# TILING CONFIGURATION
# ============================================================================
# Multi-layer tiling is used for:
#   1. Computing forces from neighbors across glued boundaries
#   2. Building Voronoi graph with cross-boundary edges
#
# More layers = more accurate neighbor detection but more memory
# Recommended by polygon type:
#   4-gon (torus, klein):     2-3 layers
#   8-gon (double torus):     3-4 layers
#   12-gon (triple torus):    4-5 layers
export TILING_LAYERS=${TILING_LAYERS:-1}

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================
# Control how often evolution plots are saved

# Interval for saving evolution plots (every N iterations)
# Set to 1 for maximum detail, higher for faster execution
# Recommended: 5-10 for detailed visualization, 25-50 for quick runs
export PLOT_INTERVAL=${PLOT_INTERVAL:-10}

# ============================================================================
# DERIVED CONFIGURATION (auto-computed)
# ============================================================================

# Auto-detect topology prefix if not set
if [ -z "${TOPOLOGY_PREFIX}" ]; then
    case "${TOPOLOGY_RULE}" in
        "abAB"|"abABcdCD"|"abABcdCDefEF")
            TOPOLOGY_PREFIX="torus"
            ;;
        "abAb"|"abaB")
            TOPOLOGY_PREFIX="klein"
            ;;
        "abBA"|"aAbB")
            TOPOLOGY_PREFIX="sphere"
            ;;
        "abab")
            TOPOLOGY_PREFIX="projective"
            ;;
        *)
            # Use first 8 chars of rule as prefix
            TOPOLOGY_PREFIX=$(echo "${TOPOLOGY_RULE}" | tr '[:upper:]' '[:lower:]' | cut -c1-8)
            ;;
    esac
    export TOPOLOGY_PREFIX
fi

# Dataset name: {prefix}_{rule}_N{points}_iter{iters}
if [ -n "${N_TOTAL}" ] && [ -n "${ITERS}" ] && [ -n "${TOPOLOGY_RULE}" ] && [ -n "${TOPOLOGY_PREFIX}" ]; then
    export DATASET_NAME="${TOPOLOGY_PREFIX}_${TOPOLOGY_RULE}_N${N_TOTAL}_iter${ITERS}"
fi

# ============================================================================
# DIRECTORY STRUCTURE
# ============================================================================
# All outputs organized under results/{DATASET_NAME}/

export RUN_NAME=${RUN_NAME:-"12M_llama"}

# Data directories (shared across runs)
export DATA_DIR="results/${DATASET_NAME}"
export GRAPH_DIR="${GRAPH_DIR:-${DATA_DIR}/graph}"
export SEQUENCE_DIR="${SEQUENCE_DIR:-${DATA_DIR}/sequences}"
export DATASET_DIR="${DATASET_DIR:-${DATA_DIR}/dataset}"
export COMBINED_DATASET_DIR="${COMBINED_DATASET_DIR:-${DATA_DIR}/combined_dataset}"

# Run-specific directories
export WORK_DIR="${DATA_DIR}/${RUN_NAME}"
export MODEL_DIR="${MODEL_DIR:-${WORK_DIR}/checkpoint-7900}"
export REPRESENTATION_DIR="${REPRESENTATION_DIR:-${MODEL_DIR}/token_representations}"
export SOURCE_TOKEN_REPRESENTATION_DIR="${SOURCE_TOKEN_REPRESENTATION_DIR:-${MODEL_DIR}/source_token_representations}"
export PCA_DIR="${PCA_DIR:-${MODEL_DIR}/pca_result}"
export FUZZY_NEIGHBORHOOD_DIR="${FUZZY_NEIGHBORHOOD_DIR:-${MODEL_DIR}/fuzzy_neighborhood}"
export UMAP_DIR="${UMAP_DIR:-${MODEL_DIR}/umap_result}"
export TOPOLOGY_ANALYSIS_DIR="${TOPOLOGY_ANALYSIS_DIR:-${MODEL_DIR}/topology_analysis}"
export PERSISTENCE_BARCODE_DIR="${PERSISTENCE_BARCODE_DIR:-${MODEL_DIR}/persistence_barcode}"

# ============================================================================
# PRINT CONFIGURATION SUMMARY (when sourced with verbose flag)
# ============================================================================
if [ "${VERBOSE_CONFIG:-0}" = "1" ]; then
    echo "============================================"
    echo "TOPO Configuration Summary"
    echo "============================================"
    echo "Topology:"
    echo "  Rule:        ${TOPOLOGY_RULE}"
    echo "  Prefix:      ${TOPOLOGY_PREFIX}"
    echo "  Polygon:     $(echo -n ${TOPOLOGY_RULE} | wc -c)-gon"
    echo ""
    echo "Evolution:"
    echo "  Points:      ${N_TOTAL}"
    echo "  Iterations:  ${ITERS}"
    echo "  Step size:   ${STEP_SIZE}"
    echo "  Tiling:      ${TILING_LAYERS} layers"
    echo ""
    echo "Visualization:"
    echo "  Plot every:  ${PLOT_INTERVAL} iterations"
    echo ""
    echo "Output:"
    echo "  Dataset:     ${DATASET_NAME}"
    echo "  Graph dir:   ${GRAPH_DIR}"
    echo "============================================"
fi
