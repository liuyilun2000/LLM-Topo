#!/bin/bash

# =============================================================================
# Shared Configuration for TOPO Analysis Pipeline
# =============================================================================
# Source in each script:  source 00_config_env.sh
#
# Structure:
#   1. Graph construction (01a)     – topology, evolution, tiling, visualization
#   2. Sequences & dataset (01b,01c) – walks, train split
#   3. Graph analysis (01d–01g)      – graph UMAP, Ripser, barcode
#   4. Model (02a, 02b)             – training, representation extraction
#   5. Representation analysis (03a–03d) – PCA, fuzzy neighborhood, UMAP
#   6. Topology on model (04a, 04b) – input mode, data type, barcode
#   7. Derived & paths              – DATASET_NAME, directory structure
# =============================================================================

# =============================================================================
# 1. GRAPH CONSTRUCTION (01a)
# =============================================================================

# --- Topology ---
# Gluing rule: lowercase = forward edge, uppercase = reverse (A = a^-1).
# Examples: abAB (torus), abABcdCD (double torus), abABcdCDefEF (triple torus),
#           abAb (Klein), abab (projective), abBA (sphere).
export TOPOLOGY_RULE=${TOPOLOGY_RULE:-"abAB"}
# export TOPOLOGY_RULE=${TOPOLOGY_RULE:-"abABcdCD"}
# export TOPOLOGY_RULE=${TOPOLOGY_RULE:-"abABcdCDefEF"}
export TOPOLOGY_PREFIX=${TOPOLOGY_PREFIX:-"torus"}

# --- Evolution (repulsive force on manifold) ---
export N_TOTAL=${N_TOTAL:-1200}
export ITERS=${ITERS:-200}
export STEP_SIZE=${STEP_SIZE:-0.05}

# --- Tiling (cross-boundary neighbors; more layers = more memory) ---
export TILING_LAYERS=${TILING_LAYERS:-1}

# --- Visualization & randomness ---
export PLOT_INTERVAL=${PLOT_INTERVAL:-20}
export SEED=${SEED:-42}

# =============================================================================
# 2. SEQUENCES & DATASET (01b, 01c)
# =============================================================================

# --- Random walks (01b) ---
export MAX_LENGTH=${MAX_LENGTH:-128}
export MAX_SEQS=${MAX_SEQS:-1200000}
export MIN_VISITS_PER_NODE=${MIN_VISITS_PER_NODE:-10000000000}
export NO_REPEAT_WINDOW=${NO_REPEAT_WINDOW:-32}
export RESTART_PROB=${RESTART_PROB:-0}
export TEMPERATURE=${TEMPERATURE:-1}

# --- Dataset prep (01c) ---
export TRAIN_SPLIT=${TRAIN_SPLIT:-0.95}

# =============================================================================
# 3. GRAPH ANALYSIS (01d, 01e, 01f, 01g)
# =============================================================================

# --- Graph UMAP (01d: 3D viz, 01e: higher-dim for topology) ---
export MATRIX_TYPE=${MATRIX_TYPE:-auto}
export GRAPH_UMAP_N_COMPONENTS_VIS=${GRAPH_UMAP_N_COMPONENTS_VIS:-3}
export GRAPH_UMAP_N_COMPONENTS_TOPOLOGY=${GRAPH_UMAP_N_COMPONENTS_TOPOLOGY:-6}
export GRAPH_UMAP_MIN_DIST=${GRAPH_UMAP_MIN_DIST:-0.2}
export GRAPH_UMAP_N_NEIGHBORS=${GRAPH_UMAP_N_NEIGHBORS:-20}
export GRAPH_UMAP_RANDOM_STATE=${GRAPH_UMAP_RANDOM_STATE:-100}
export TRAJECTORY_IDX=${TRAJECTORY_IDX:-128}

# --- Ripser (01f, 04a) ---
export RIPSER_THRESH=${RIPSER_THRESH:-}
export RIPSER_MAXDIM=${RIPSER_MAXDIM:-2}
export RIPSER_COEFF=${RIPSER_COEFF:-3}

# --- Persistence barcode (01g, 04b) ---
export MAX_BARS=${MAX_BARS:-20}
export TOP_K=${TOP_K:-4}

# =============================================================================
# 4. MODEL TRAINING & EXTRACTION (02a, 02b)
# =============================================================================

# --- Training (02a); CONFIG default set in 02a script (uses RUN_NAME) ---
export EPOCHS=${EPOCHS:-1}
export BATCH_SIZE=${BATCH_SIZE:-10}
export GRAD_ACCUM=${GRAD_ACCUM:-1}
export LEARNING_RATE=${LEARNING_RATE:-5e-4}
export SAVE_STEPS=${SAVE_STEPS:-1000}
export EVAL_STEPS=${EVAL_STEPS:-20}
export LOGGING_STEPS=${LOGGING_STEPS:-1}
export SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-all}
export USE_CPU=${USE_CPU:-false}

# --- Representation extraction (02b) ---
# representations: after_block, ffn_gate, ffn_up, etc.
export REPRESENTATIONS=${REPRESENTATIONS:-after_block}
export REPRESENTATION_UPSEAMPLE=${REPRESENTATION_UPSEAMPLE:-true}
export REPRESENTATION_UPSEAMPLE_N=${REPRESENTATION_UPSEAMPLE_N:-100}
export REPRESENTATION_UPSEAMPLE_SEED=${REPRESENTATION_UPSEAMPLE_SEED:-42}

# =============================================================================
# 5. REPRESENTATION ANALYSIS (03a, 03b, 03c, 03d)
# =============================================================================

# --- PCA (03a); use one of the two ---
export PCA_N_COMPONENTS=${PCA_N_COMPONENTS:-}
export PCA_VARIANCE=${PCA_VARIANCE:-0.95}

# --- Fuzzy neighborhood (03b) ---
export N_NEIGHBORS=${N_NEIGHBORS:-20}
export METRIC=${METRIC:-cosine}
export SYM_METHOD=${SYM_METHOD:-fuzzy_union}
export EPSILON=${EPSILON:-1e-10}
export SPARSITY_THRESHOLD=${SPARSITY_THRESHOLD:-0.0}
export TARGET_ENTROPY=${TARGET_ENTROPY:-auto}

# --- UMAP on model (03c: vis 2D/3D, 03d: topology e.g. 10D) ---
export USE_FUZZY=${USE_FUZZY:-true}
export UMAP_N_COMPONENTS_VIS=${UMAP_N_COMPONENTS_VIS:-3}
export UMAP_N_COMPONENTS_TOPOLOGY=${UMAP_N_COMPONENTS_TOPOLOGY:-6}
export UMAP_MIN_DIST=${UMAP_MIN_DIST:-0.2}
export UMAP_N_NEIGHBORS=${UMAP_N_NEIGHBORS:-20}
export UMAP_METRIC=${UMAP_METRIC:-cosine}
export UMAP_RANDOM_STATE=${UMAP_RANDOM_STATE:-42}
export SAVE_UMAP_RESULT=${SAVE_UMAP_RESULT:-true}
export GENERATE_VISUALIZATIONS=${GENERATE_VISUALIZATIONS:-true}

# =============================================================================
# 6. TOPOLOGY ANALYSIS ON MODEL (04a)
# =============================================================================
export INPUT_MODE=${INPUT_MODE:-data_representation}
export DATA_TYPE=${DATA_TYPE:-auto}

# =============================================================================
# 7. DERIVED CONFIGURATION
# =============================================================================

# Topology prefix from rule (if not set)
if [ -z "${TOPOLOGY_PREFIX}" ]; then
    case "${TOPOLOGY_RULE}" in
        "abAB"|"abABcdCD"|"abABcdCDefEF") TOPOLOGY_PREFIX="torus" ;;
        "abAb"|"abaB")                     TOPOLOGY_PREFIX="klein" ;;
        "abBA"|"aAbB")                     TOPOLOGY_PREFIX="sphere" ;;
        "abab")                            TOPOLOGY_PREFIX="projective" ;;
        *) TOPOLOGY_PREFIX=$(echo "${TOPOLOGY_RULE}" | tr '[:upper:]' '[:lower:]' | cut -c1-8) ;;
    esac
    export TOPOLOGY_PREFIX
fi

# Dataset name: {prefix}_{rule}_N{points}_iter{iters}
if [ -n "${N_TOTAL}" ] && [ -n "${ITERS}" ] && [ -n "${TOPOLOGY_RULE}" ] && [ -n "${TOPOLOGY_PREFIX}" ]; then
    export DATASET_NAME="${TOPOLOGY_PREFIX}_${TOPOLOGY_RULE}_N${N_TOTAL}_iter${ITERS}"
fi

# =============================================================================
# 8. DIRECTORY STRUCTURE
# =============================================================================
# Base: results/{DATASET_NAME}/ ; run-specific: results/{DATASET_NAME}/{RUN_NAME}/

export RUN_NAME=${RUN_NAME:-"12M_llama"}
# Model config JSON (by run name); set CONFIG to override. Must include "architectures": ["LlamaForCausalLM"] or ["MambaForCausalLM"].
export CONFIG="${CONFIG:-configs/config_${RUN_NAME}.json}"

export DATA_DIR="results/${DATASET_NAME}"
export GRAPH_DIR="${GRAPH_DIR:-${DATA_DIR}/graph}"
export SEQUENCE_DIR="${SEQUENCE_DIR:-${DATA_DIR}/sequences}"
export DATASET_DIR="${DATASET_DIR:-${DATA_DIR}/dataset}"
export COMBINED_DATASET_DIR="${COMBINED_DATASET_DIR:-${DATA_DIR}/combined_dataset}"

export WORK_DIR="${DATA_DIR}/${RUN_NAME}"
export MODEL_DIR="${MODEL_DIR:-${WORK_DIR}/final_model}"

# UMAP result dirs (folder name = umap_result_{N}d from config dimension)
export UMAP_VIS_RESULT_DIR="${UMAP_VIS_RESULT_DIR:-${MODEL_DIR}/umap_result_${UMAP_N_COMPONENTS_VIS}d}"
export TOPOLOGY_UMAP_RESULT_DIR="${TOPOLOGY_UMAP_RESULT_DIR:-${MODEL_DIR}/umap_result_${UMAP_N_COMPONENTS_TOPOLOGY}d}"
export GRAPH_UMAP_TOPOLOGY_RESULT_DIR="${GRAPH_UMAP_TOPOLOGY_RESULT_DIR:-${DATA_DIR}/graph_umap_result_${GRAPH_UMAP_N_COMPONENTS_TOPOLOGY}d}"

export REPRESENTATION_DIR="${REPRESENTATION_DIR:-${MODEL_DIR}/token_representations}"
export SOURCE_TOKEN_REPRESENTATION_DIR="${SOURCE_TOKEN_REPRESENTATION_DIR:-${MODEL_DIR}/source_token_representations}"
export PCA_DIR="${PCA_DIR:-${MODEL_DIR}/pca_result}"
export FUZZY_NEIGHBORHOOD_DIR="${FUZZY_NEIGHBORHOOD_DIR:-${MODEL_DIR}/fuzzy_neighborhood}"
export UMAP_DIR="${UMAP_DIR:-${MODEL_DIR}/umap_result}"
export TOPOLOGY_ANALYSIS_DIR="${TOPOLOGY_ANALYSIS_DIR:-${MODEL_DIR}/topology_analysis}"
export PERSISTENCE_BARCODE_DIR="${PERSISTENCE_BARCODE_DIR:-${MODEL_DIR}/persistence_barcode}"

# =============================================================================
# 9. VERBOSE SUMMARY (VERBOSE_CONFIG=1)
# =============================================================================
if [ "${VERBOSE_CONFIG:-0}" = "1" ]; then
    echo "============================================"
    echo "TOPO Configuration Summary"
    echo "============================================"
    echo "Topology:      ${TOPOLOGY_RULE} (${TOPOLOGY_PREFIX}, $(echo -n ${TOPOLOGY_RULE} | wc -c)-gon)"
    echo "Evolution:     N=${N_TOTAL} iter=${ITERS} step=${STEP_SIZE} layers=${TILING_LAYERS}"
    echo "Plot/seed:     interval=${PLOT_INTERVAL} seed=${SEED}"
    echo "Dataset:       ${DATASET_NAME}"
    echo "Paths:         DATA_DIR=${DATA_DIR} MODEL_DIR=${MODEL_DIR}"
    echo "============================================"
fi
