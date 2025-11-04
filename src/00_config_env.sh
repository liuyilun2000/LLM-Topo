#!/bin/bash

# ============================================================================
# Shared Configuration for TOPO Analysis Pipeline
# ============================================================================
# Source this file in each script: source 00_config_env.sh
#
# This file provides centralized configuration for:
#   - Grid dimensions (H, W)
#   - Topology type
#   - Dataset naming
#   - Run naming
# ============================================================================

# Grid dimensions
export H=${H:-30}
export W=${W:-40}

# Topology configuration
# Available topology types:
#   - plane          : Plane (flat, no wrapping)
#   - cylinder_x     : Cylinder wrapping in x-direction
#   - cylinder_y     : Cylinder wrapping in y-direction
#   - mobius_x       : Möbius strip wrapping in x-direction
#   - mobius_y       : Möbius strip wrapping in y-direction
#   - torus          : Torus (both directions wrap)
#   - klein_x        : Klein bottle wrapping in x-direction
#   - klein_y        : Klein bottle wrapping in y-direction
#   - proj_plane     : Projective plane
#   - sphere_two     : Sphere with two hemispheres (A and B layers, reverse boundary gluing)
#   - hemisphere_n   : Northern hemisphere (open boundary)
#   - hemisphere_s   : Southern hemisphere (open boundary)
#   - sphere         : Sphere (single point S, all boundaries connect to S)
export TOPOLOGY=${TOPOLOGY:-"klein_y"}

# Derived configuration - always recomputed from H, W, TOPOLOGY
# DATASET_NAME is automatically generated from topology and dimensions
if [ -n "${H}" ] && [ -n "${W}" ] && [ -n "${TOPOLOGY}" ]; then
    export DATASET_NAME="${TOPOLOGY}_${H}x${W}"
fi

# Run configuration (can be overridden)
export RUN_NAME=${RUN_NAME:-"2M_llama"}

# Working directory structure:
#   - Data-related outputs (graphs, walks, prepared datasets): {DATASET_NAME}/
#   - Run-related outputs (models, representations, analyses): {DATASET_NAME}/{RUN_NAME}/
export WORK_DIR="results/${DATASET_NAME}/${RUN_NAME}"
export DATA_DIR="results/${DATASET_NAME}"

# Note: Individual scripts should set their own SCRIPT_DIR based on where they are
# and use relative paths from that location.

