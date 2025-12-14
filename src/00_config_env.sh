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
#   - plane          : Plane (flat, no wrapping) (H=height/Y, W=width/X)
#   - cylinder_x     : Cylinder wrapping in x-direction (H=height/Z-axis, W=circumference/angle)
#   - cylinder_y     : Cylinder wrapping in y-direction (H=circumference/angle, W=height/Z-axis)
#   - mobius_x       : Möbius strip wrapping in x-direction (H=width/across-strip, W=circumference/main-loop)
#   - mobius_y       : Möbius strip wrapping in y-direction (H=circumference/main-loop, W=width/across-strip)
#   - torus          : Torus (both directions wrap) (H=poloidal/tube-circumference, W=toroidal/major-radius)
#   - klein_x        : Klein bottle wrapping in x-direction (H=U-direction, W=V-direction) (H=toroidal/major-loop, W=poloidal/tube-with-twist)
#   - klein_y        : Klein bottle wrapping in y-direction (H=V-direction, W=U-direction) (H=poloidal/tube-with-twist, W=toroidal/major-loop)
#   - proj_plane     : Projective plane (H=U-parameter, W=V-parameter)
#   - sphere_two     : Sphere with two hemispheres (A and B layers, reverse boundary gluing) (H=square-grid-Y, W=square-grid-X)
#   - hemisphere_n   : Northern hemisphere (open boundary) (H=latitude/radial-to-pole, W=longitude/azimuthal)
#   - hemisphere_s   : Southern hemisphere (open boundary) (H=latitude/radial-to-pole, W=longitude/azimuthal)
#   - sphere         : Sphere (single point S, all boundaries connect to S) (H≈latitude, W≈longitude, interior points only)
export TOPOLOGY=${TOPOLOGY:-"torus"}

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
export DATASET_DIR="${DATASET_DIR:-${DATA_DIR}/dataset}"
export COMBINED_DATASET_DIR="${COMBINED_DATASET_DIR:-${DATA_DIR}/combined_dataset}"

# Graph generation parameters
export GRAPH_DIR="${GRAPH_DIR:-./${DATA_DIR}/graph}"
export NEIGH=${NEIGH:-8}


#export MODEL_DIR="${MODEL_DIR:-${WORK_DIR}/final_model}" # or checkpoints directory
export MODEL_DIR="${MODEL_DIR:-${WORK_DIR}/final_model}"
export REPRESENTATION_DIR="${REPRESENTATION_DIR:-${MODEL_DIR}/token_representations}"
export SOURCE_TOKEN_REPRESENTATION_DIR="${SOURCE_TOKEN_REPRESENTATION_DIR:-${MODEL_DIR}/source_token_representations}"
export PCA_DIR="${PCA_DIR:-${MODEL_DIR}/pca_result}"
export FUZZY_NEIGHBORHOOD_DIR="${FUZZY_NEIGHBORHOOD_DIR:-${MODEL_DIR}/fuzzy_neighborhood}"
export UMAP_DIR="${UMAP_DIR:-${MODEL_DIR}/umap_result}"
export TOPOLOGY_ANALYSIS_DIR="${TOPOLOGY_ANALYSIS_DIR:-${MODEL_DIR}/topology_analysis}"
export PERSISTENCE_BARCODE_DIR="${PERSISTENCE_BARCODE_DIR:-${MODEL_DIR}/persistence_barcode}"

# Note: Individual scripts should set their own SCRIPT_DIR based on where they are
# and use relative paths from that location.

