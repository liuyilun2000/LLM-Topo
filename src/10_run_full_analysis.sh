#!/bin/bash

# ============================================================================
# Run Full Analysis Pipeline for Multiple Checkpoints (Parallel Execution)
# ============================================================================
#
# This script runs the full analysis pipeline (12b, 03a, 03b, 03c_visualize, 03d_analysis, 04a, 04b)
# for multiple specified checkpoints in parallel.
#
# Usage:
#   ./10_run_full_analysis.sh [CHECKPOINT1] [CHECKPOINT2] ... [CHECKPOINTN]
#
# Examples:
#   # Run for specific checkpoints
#   ./10_run_full_analysis.sh checkpoint-400 checkpoint-800 checkpoint-1200 final_model
#
#   # Run for checkpoints 400, 800, 1200, 1600, 2000, and final_model
#   ./10_run_full_analysis.sh 400 800 1200 1600 2000 final_model
#
#   # Run for all checkpoints found in WORK_DIR (if no arguments provided)
#   ./10_run_full_analysis.sh
#
# Environment Variables:
#   - WORK_DIR: Base directory containing checkpoints (default: from 00_config_env.sh)
#   - MAX_PARALLEL: Maximum number of parallel jobs (default: number of CPUs)
#   - SKIP_EXISTING: Skip checkpoints that already have all outputs (default: false)
#
# ============================================================================

set -e

# Load shared configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/00_config_env.sh"

# Configuration
# Determine max parallel jobs: check SLURM first, then nproc
if [ -n "$SLURM_NTASKS" ]; then
    DEFAULT_MAX_PARALLEL="$SLURM_NTASKS"
elif [ -n "$SLURM_CPUS_PER_TASK" ]; then
    DEFAULT_MAX_PARALLEL="$SLURM_CPUS_PER_TASK"
elif [ -n "$SLURM_CPUS_ON_NODE" ]; then
    DEFAULT_MAX_PARALLEL="$SLURM_CPUS_ON_NODE"
else
    DEFAULT_MAX_PARALLEL=$(nproc)
fi
MAX_PARALLEL="${MAX_PARALLEL:-$DEFAULT_MAX_PARALLEL}"
SKIP_EXISTING="${SKIP_EXISTING:-false}"


CHECKPOINTS=(checkpoint-100 checkpoint-200 checkpoint-400 checkpoint-800)
CHECKPOINTS=(checkpoint-2800)

CHECKPOINTS=(checkpoint-1200 checkpoint-2000 checkpoint-3200 checkpoint-4800)

CHECKPOINTS=(checkpoint-100 checkpoint-400 checkpoint-800 checkpoint-1000 checkpoint-1400 checkpoint-1600 checkpoint-1800 checkpoint-2400 checkpoint-2800 checkpoint-3600 checkpoint-4000 checkpoint-5600 checkpoint-6400 checkpoint-8000)

# Parse checkpoint numbers/names from arguments
# If CHECKPOINTS is already set (hardcoded in script), use it
# Otherwise, use command-line arguments or auto-detect
if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    # CHECKPOINTS not pre-populated, check command-line arguments
    if [ $# -eq 0 ]; then
        # No arguments: find all checkpoints in WORK_DIR
        echo "No checkpoints specified. Finding all checkpoints in $WORK_DIR..."
        if [ ! -d "$WORK_DIR" ]; then
            echo "Error: WORK_DIR does not exist: $WORK_DIR"
            exit 1
        fi
        
        for item in "$WORK_DIR"/*; do
            if [ -d "$item" ]; then
                checkpoint_name=$(basename "$item")
                # Check if it looks like a checkpoint directory
                if [[ "$checkpoint_name" =~ ^checkpoint-[0-9]+$ ]] || [ "$checkpoint_name" = "final_model" ]; then
                    CHECKPOINTS+=("$checkpoint_name")
                fi
            fi
        done
        
        if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
            echo "Error: No checkpoints found in $WORK_DIR"
            exit 1
        fi
        
        echo "Found ${#CHECKPOINTS[@]} checkpoints: ${CHECKPOINTS[*]}"
    else
        # Arguments provided: convert numbers to checkpoint-XXX format
        for arg in "$@"; do
            if [[ "$arg" =~ ^[0-9]+$ ]]; then
                # Numeric argument: convert to checkpoint-XXX
                CHECKPOINTS+=("checkpoint-$arg")
            elif [[ "$arg" =~ ^checkpoint-[0-9]+$ ]] || [ "$arg" = "final_model" ]; then
                # Already in correct format
                CHECKPOINTS+=("$arg")
            else
                echo "Warning: Invalid checkpoint format: $arg (expected number or checkpoint-XXX or final_model)"
            fi
        done
    fi
else
    # CHECKPOINTS already set (hardcoded in script), use it
    echo "Using pre-configured checkpoints: ${CHECKPOINTS[*]}"
fi

if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo "Error: No valid checkpoints specified"
    exit 1
fi

echo ""
echo "=========================================="
echo "Full Analysis Pipeline for Multiple Checkpoints"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Work directory: $WORK_DIR"
echo "  Checkpoints to process: ${#CHECKPOINTS[@]}"
echo "    ${CHECKPOINTS[*]}"
echo "  Max parallel jobs: $MAX_PARALLEL"
echo "  Skip existing: $SKIP_EXISTING"
echo ""

# Function to check if all outputs exist for a checkpoint
check_outputs_exist() {
    local checkpoint_dir="$1"
    local model_dir="$WORK_DIR/$checkpoint_dir"
    
    # Check for required directories/files
    [ -d "$model_dir/pca_result" ] && \
    [ -d "$model_dir/fuzzy_neighborhood" ] && \
    [ -d "$model_dir/umap_result"* ] && \
    [ -d "$model_dir/topology_analysis" ] && \
    [ -d "$model_dir/persistence_barcode" ]
}

# Function to run analysis for a single checkpoint
run_checkpoint_analysis() {
    local checkpoint_name="$1"
    local checkpoint_dir="$WORK_DIR/$checkpoint_name"
    local log_file="$WORK_DIR/${checkpoint_name}_analysis.log"
    
    echo "[$checkpoint_name] Starting analysis at $(date)"
    
    # Check if checkpoint directory exists
    if [ ! -d "$checkpoint_dir" ]; then
        echo "[$checkpoint_name] ERROR: Checkpoint directory does not exist: $checkpoint_dir" | tee -a "$log_file"
        return 1
    fi
    
    # Check if we should skip
    if [ "$SKIP_EXISTING" = "true" ] && check_outputs_exist "$checkpoint_name"; then
        echo "[$checkpoint_name] Skipping: All outputs already exist" | tee -a "$log_file"
        return 0
    fi
    
    # Set MODEL_DIR for this checkpoint
    export MODEL_DIR="$checkpoint_dir"
    export REPRESENTATION_DIR="${MODEL_DIR}/token_representations"
    export PCA_DIR="${MODEL_DIR}/pca_result"
    export FUZZY_NEIGHBORHOOD_DIR="${MODEL_DIR}/fuzzy_neighborhood"
    export UMAP_DIR="${MODEL_DIR}/umap_result"
    export TOPOLOGY_ANALYSIS_DIR="${MODEL_DIR}/topology_analysis"
    export PERSISTENCE_BARCODE_DIR="${MODEL_DIR}/persistence_barcode"
    
    # Run all analysis scripts in sequence
    {
        echo "[$checkpoint_name] ========================================"
        echo "[$checkpoint_name] Running analysis for: $checkpoint_name"
        echo "[$checkpoint_name] ========================================"
        echo ""
        
        # Step 0: Source Token Extraction (if needed)
        if [ ! -f "$REPRESENTATION_DIR/token_representations.npz" ]; then
            echo "[$checkpoint_name] Step 0/7: Source Token Extraction..."
            cd "$SCRIPT_DIR"
            if ! ./12b_source_token_extraction.sh >> "$log_file" 2>&1; then
                echo "[$checkpoint_name] ERROR: Source token extraction failed" | tee -a "$log_file"
                echo "[$checkpoint_name] Token representations not found. Please ensure:" | tee -a "$log_file"
                echo "[$checkpoint_name]   - Model exists at $MODEL_DIR" | tee -a "$log_file"
                echo "[$checkpoint_name]   - Combined dataset exists (for 12b)" | tee -a "$log_file"
                echo "[$checkpoint_name]   - Or run ./02b_representation_extraction.sh manually" | tee -a "$log_file"
                return 1
            fi
            echo "[$checkpoint_name] ✓ Source Token Extraction complete"
        else
            echo "[$checkpoint_name] Step 0/7: Source Token Extraction (skipped - already exists)"
        fi
        
        # Step 1: PCA Analysis
        echo "[$checkpoint_name] Step 1/7: PCA Analysis..."
        cd "$SCRIPT_DIR"
        if ! ./03a_pca_analysis.sh >> "$log_file" 2>&1; then
            echo "[$checkpoint_name] ERROR: PCA analysis failed" | tee -a "$log_file"
            return 1
        fi
        echo "[$checkpoint_name] ✓ PCA Analysis complete"
        
        # Step 2: Fuzzy Neighborhood
        echo "[$checkpoint_name] Step 2/7: Fuzzy Neighborhood..."
        if ! ./03b_fuzzy_neighborhood.sh >> "$log_file" 2>&1; then
            echo "[$checkpoint_name] ERROR: Fuzzy neighborhood computation failed" | tee -a "$log_file"
            return 1
        fi
        echo "[$checkpoint_name] ✓ Fuzzy Neighborhood complete"
        
        # Step 3: UMAP Visualization (3D)
        echo "[$checkpoint_name] Step 3/7: UMAP Visualization..."
        if ! ./03c_umap_visualize.sh >> "$log_file" 2>&1; then
            echo "[$checkpoint_name] ERROR: UMAP visualization failed" | tee -a "$log_file"
            return 1
        fi
        echo "[$checkpoint_name] ✓ UMAP Visualization complete"
        
        # Step 4: UMAP Analysis (6D for topology)
        echo "[$checkpoint_name] Step 4/7: UMAP Analysis..."
        if ! ./03d_umap_analysis.sh >> "$log_file" 2>&1; then
            echo "[$checkpoint_name] ERROR: UMAP analysis failed" | tee -a "$log_file"
            return 1
        fi
        echo "[$checkpoint_name] ✓ UMAP Analysis complete"
        
        # Step 5: Topology Analysis
        echo "[$checkpoint_name] Step 5/7: Topology Analysis..."
        if ! ./04a_topology_analysis.sh >> "$log_file" 2>&1; then
            echo "[$checkpoint_name] ERROR: Topology analysis failed" | tee -a "$log_file"
            return 1
        fi
        echo "[$checkpoint_name] ✓ Topology Analysis complete"
        
        # Step 6: Persistence Barcode
        echo "[$checkpoint_name] Step 6/7: Persistence Barcode..."
        if ! ./04b_persistence_barcode.sh >> "$log_file" 2>&1; then
            echo "[$checkpoint_name] ERROR: Persistence barcode generation failed" | tee -a "$log_file"
            return 1
        fi
        echo "[$checkpoint_name] ✓ Persistence Barcode complete"
        
        echo ""
        echo "[$checkpoint_name] ========================================"
        echo "[$checkpoint_name] ✓ All analysis steps complete for $checkpoint_name"
        echo "[$checkpoint_name] ========================================"
        echo "[$checkpoint_name] Completed at $(date)"
        
    } | tee -a "$log_file"
    
    return 0
}

# Export function for parallel execution
export -f run_checkpoint_analysis
export SCRIPT_DIR
export WORK_DIR
export SKIP_EXISTING

# Run checkpoints in parallel
echo "Starting parallel execution..."
echo ""

# Track failed checkpoints
FAILED_CHECKPOINTS=()

# Launch jobs in parallel
echo "Launching ${#CHECKPOINTS[@]} checkpoint analysis jobs (max $MAX_PARALLEL parallel)..."
echo ""

# Use GNU parallel if available, otherwise use simple background jobs with job control
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for job management"
    printf '%s\n' "${CHECKPOINTS[@]}" | parallel -j "$MAX_PARALLEL" run_checkpoint_analysis {}
else
    echo "Using background jobs (GNU parallel not available, install for better job management)"
    
    # Track running jobs using parallel arrays
    local -a RUNNING_PIDS=()
    local -a RUNNING_CHECKPOINTS=()
    
    for checkpoint in "${CHECKPOINTS[@]}"; do
        # Wait if we've reached max parallel jobs
        while [ ${#RUNNING_PIDS[@]} -ge "$MAX_PARALLEL" ]; do
            # Check for finished jobs
            local new_pids=()
            local new_checkpoints=()
            for i in "${!RUNNING_PIDS[@]}"; do
                local pid="${RUNNING_PIDS[$i]}"
                if kill -0 "$pid" 2>/dev/null; then
                    # Job still running
                    new_pids+=("$pid")
                    new_checkpoints+=("${RUNNING_CHECKPOINTS[$i]}")
                else
                    # Job finished, wait for it and get exit code
                    wait "$pid"
                    local exit_code=$?
                    local finished_checkpoint="${RUNNING_CHECKPOINTS[$i]}"
                    if [ $exit_code -eq 0 ]; then
                        echo "[MASTER] ✓ Completed: $finished_checkpoint"
                    else
                        echo "[MASTER] ✗ Failed: $finished_checkpoint"
                        FAILED_CHECKPOINTS+=("$finished_checkpoint")
                    fi
                fi
            done
            RUNNING_PIDS=("${new_pids[@]}")
            RUNNING_CHECKPOINTS=("${new_checkpoints[@]}")
            sleep 1
        done
        
        # Launch new job
        echo "[MASTER] Launching: $checkpoint"
        run_checkpoint_analysis "$checkpoint" &
        local pid=$!
        RUNNING_PIDS+=("$pid")
        RUNNING_CHECKPOINTS+=("$checkpoint")
    done
    
    # Wait for all remaining jobs
    for i in "${!RUNNING_PIDS[@]}"; do
        wait "${RUNNING_PIDS[$i]}"
        local exit_code=$?
        local checkpoint="${RUNNING_CHECKPOINTS[$i]}"
        if [ $exit_code -eq 0 ]; then
            echo "[MASTER] ✓ Completed: $checkpoint"
        else
            echo "[MASTER] ✗ Failed: $checkpoint"
            FAILED_CHECKPOINTS+=("$checkpoint")
        fi
    done
fi

echo ""
echo "=========================================="
echo "All checkpoint analyses completed!"
echo "=========================================="
echo ""

# Summary
SUCCESS_COUNT=0
FAILED_COUNT=0

for checkpoint in "${CHECKPOINTS[@]}"; do
    if check_outputs_exist "$checkpoint"; then
        ((SUCCESS_COUNT++))
        echo "✓ $checkpoint: All outputs generated"
    else
        ((FAILED_COUNT++))
        echo "✗ $checkpoint: Missing outputs (check log: ${WORK_DIR}/${checkpoint}_analysis.log)"
    fi
done

echo ""
echo "Summary:"
echo "  Successful: $SUCCESS_COUNT"
echo "  Failed: $FAILED_COUNT"
echo "  Total: ${#CHECKPOINTS[@]}"
echo ""

if [ $FAILED_COUNT -gt 0 ]; then
    echo "Failed checkpoints:"
    for checkpoint in "${CHECKPOINTS[@]}"; do
        if ! check_outputs_exist "$checkpoint"; then
            echo "  - $checkpoint (log: ${WORK_DIR}/${checkpoint}_analysis.log)"
        fi
    done
    echo ""
    exit 1
fi

echo "All checkpoints processed successfully!"
echo ""
