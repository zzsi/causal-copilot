#!/bin/bash

# Get the directory containing this script
SCRIPT_DIR="algorithm/tests"

# Set number of parallel jobs
N_JOBS=2

# Define GPU devices to use in rotation
GPU_DEVICES=(4 5 6 7) # Modify this array with your available GPU IDs

# Define CPU cores to use and calculate cores per job
# Detect total available CPU cores
TOTAL_CPU_CORES=$(nproc)
echo "Total available CPU cores: $TOTAL_CPU_CORES"

# Calculate cores per job (divide available cores by number of jobs)
CORES_PER_JOB=$((TOTAL_CPU_CORES / N_JOBS))
if [ $CORES_PER_JOB -lt 1 ]; then
    CORES_PER_JOB=1
    echo "Warning: More jobs than CPU cores. Setting minimum 1 core per job."
fi
echo "Assigning $CORES_PER_JOB core(s) per job"

# Get list of algorithms from config
ALGORITHMS=(PC FCI CDNOD FGES XGES GRaSP GES
            DirectLiNGAM # GOLEM 
            NOTEARSLinear BAMB IAMBnPC MBOR InterIAMB # HITONMB
            #Hybrid CALM NOTEARSNonlinear CORL
)

# Define algorithm hyperparameter combinations for linear and nonlinear settings
# Format: "algorithm_name:hyperparameter_json"
ALGORITHM_HYPERPARAMS=(
    # PC algorithm - nonlinear variants
    "PC:{\"indep_test\":\"rcit_cpu\"}"
    "PC:{\"indep_test\":\"fastkci_cpu\"}"
    "PC:{\"indep_test\":\"kci_cpu\"}"
    
    # # FCI algorithm - nonlinear variants
    "FCI:{\"indep_test\":\"rcit\"}"
    "FCI:{\"indep_test\":\"fastkci\"}"
    "FCI:{\"indep_test\":\"kci\"}"
    
    # # CDNOD algorithm - nonlinear variants
    "CDNOD:{\"indep_test\":\"rcit\"}"
    "CDNOD:{\"indep_test\":\"fastkci\"}"
    "CDNOD:{\"indep_test\":\"kci_cpu\"}"
    
    # # DirectLiNGAM - kernel-based nonlinear variant
    "DirectLiNGAM:{\"measure\":\"kernel\",\"gpu\":false}"
    "DirectLiNGAM:{\"measure\":\"kernel\",\"gpu\":true}"
    
    # # Markov blanket algorithms - nonlinear independence tests
    "BAMB:{\"indep_test\":\"rcit\"}"
    "BAMB:{\"indep_test\":\"fastkci\"}"
    "BAMB:{\"indep_test\":\"kci\"}"
    
    "IAMBnPC:{\"indep_test\":\"rcit\"}"
    "IAMBnPC:{\"indep_test\":\"fastkci\"}"
    "IAMBnPC:{\"indep_test\":\"kci\"}"
    
    "MBOR:{\"indep_test\":\"rcit\"}"
    "MBOR:{\"indep_test\":\"fastkci\"}"
    "MBOR:{\"indep_test\":\"kci\"}"
    
    "InterIAMB:{\"indep_test\":\"rcit\"}"
    "InterIAMB:{\"indep_test\":\"fastkci\"}"
    "InterIAMB:{\"indep_test\":\"kci\"}"

    # Linear methods first
    
    # GRaSP algorithm - linear score-based
    "GRaSP:{\"score_func\":\"local_score_BIC\"}"

    # GES algorithm - linear score-based
    "GES:{\"score_func\":\"local_score_BIC\"}"
    
    # # PC algorithm - linear variant
    "PC:{\"indep_test\":\"fisherz_cpu\"}"
    "PC:{\"indep_test\":\"fisherz_gpu\"}"
    "PC:{\"indep_test\":\"cmiknn_gpu\"}"
    
    # # FCI algorithm - linear variant
    "FCI:{\"indep_test\":\"fisherz\"}"
    
    # # # CDNOD algorithm - linear variant
    "CDNOD:{\"indep_test\":\"fisherz_gpu\"}"
    "CDNOD:{\"indep_test\":\"cmiknn_gpu\"}"
    "CDNOD:{\"indep_test\":\"fisherz_cpu\"}"
    
    # # # FGES algorithm
    "FGES:{\"sparsity\":10}"
    "FGES:{\"sparsity\":5}"
    "FGES:{\"sparsity\":2}"
    "FGES:{\"sparsity\":1}"
    
    # # # XGES algorithm
    "XGES:{\"alpha\":0.5}"
    "XGES:{\"alpha\":1}"
    "XGES:{\"alpha\":2}"
    "XGES:{\"alpha\":4}"
    
    # # DirectLiNGAM algorithm - linear non-Gaussian method
    "DirectLiNGAM:{\"measure\":\"pwling\",\"gpu\":false}"
    "DirectLiNGAM:{\"measure\":\"pwling\",\"gpu\":true}"
    
    # NOTEARSLinear algorithm - gradient-based optimization
    "NOTEARSLinear:{\"lambda1\":0.01,\"loss_type\":\"l2\"}"
    
    # Markov blanket algorithms - linear independence tests
    "BAMB:{\"indep_test\":\"fisherz\"}"
    "IAMBnPC:{\"indep_test\":\"fisherz\"}"
    "MBOR:{\"indep_test\":\"fisherz\"}"
    "InterIAMB:{\"indep_test\":\"fisherz\"}"
    
    # Nonlinear methods

    # GES algorithm - nonlinear score
    # "GES:{\"score_func\":\"local_score_marginal_general\"}"
    
    # GRaSP algorithm - nonlinear score
    # "GRaSP:{\"score_func\":\"local_score_marginal_general\"}"
)

# ALGORITHMS=(AcceleratedPC)

# (CDNOD AcceleratedPC AcceleratedLiNGAM FCI GES GOLEM GRaSP IAMBnPC InterIAMB MBOR NOTEARSLinear PC XGES)

# Create a temporary directory for parallel job logs
LOG_DIR="$SCRIPT_DIR/benchmark_logs_v3/$(date +"%Y%m%d_%H%M%S")"
mkdir -p "$LOG_DIR"

# Create a file to track GPU assignment across processes
GPU_COUNTER_FILE="${LOG_DIR}/gpu_counter"
echo "0" > "$GPU_COUNTER_FILE"

# Create a file to track CPU core assignment across processes
CPU_COUNTER_FILE="${LOG_DIR}/cpu_counter"
echo "0" > "$CPU_COUNTER_FILE"

# Function to get the next GPU device in rotation using a file-based counter
get_next_gpu() {
    # Use flock to ensure atomic read-modify-write operation
    (
        flock -x 200
        gpu_counter=$(cat "$GPU_COUNTER_FILE")
        gpu=${GPU_DEVICES[$gpu_counter]}
        # Increment counter and wrap around
        gpu_counter=$(( (gpu_counter + 1) % ${#GPU_DEVICES[@]} ))
        echo "$gpu_counter" > "$GPU_COUNTER_FILE"
        echo "$gpu"
    ) 200>"${LOG_DIR}/gpu_lock"
}

# Function to get the next set of CPU cores for a job
get_next_cpu_cores() {
    (
        flock -x 201
        cpu_counter=$(cat "$CPU_COUNTER_FILE")
        # Calculate core range for this job
        start_core=$((cpu_counter * CORES_PER_JOB))
        end_core=$((start_core + CORES_PER_JOB - 1))
        # Ensure we don't exceed available cores
        if [ $end_core -ge $TOTAL_CPU_CORES ]; then
            end_core=$((TOTAL_CPU_CORES - 1))
        fi
        # Increment counter and wrap around
        cpu_counter=$(( (cpu_counter + 1) % N_JOBS ))
        echo "$cpu_counter" > "$CPU_COUNTER_FILE"
        # Create CPU mask
        cpu_list=""
        for (( core=start_core; core<=end_core; core++ )); do
            if [ -z "$cpu_list" ]; then
                cpu_list="$core"
            else
                cpu_list="$cpu_list,$core"
            fi
        done
        echo "$cpu_list"
    ) 201>"${LOG_DIR}/cpu_lock"
}

# Function to run benchmark for a single algorithm with logging
run_benchmark() {
    local algo=$1
    local params=$2
    local timestamp
    timestamp=$(date +"%Y%m%d_%H%M%S")
    local log_file="${LOG_DIR}/${algo}_${timestamp}.log"
    
    # Get the next GPU device in rotation
    local gpu_device=$(get_next_gpu)
    
    # Get the next set of CPU cores
    local cpu_cores=$(get_next_cpu_cores)
    
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] Starting benchmark for algorithm: $algo on GPU $gpu_device with CPU cores $cpu_cores" | tee -a "$log_file"
    
    # Run the benchmark with params if provided
    if [ -n "$params" ]; then
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] Using hyperparameters: $params" | tee -a "$log_file"
        # Use taskset to set CPU affinity and CUDA_VISIBLE_DEVICES for GPU
        CUDA_VISIBLE_DEVICES=$gpu_device taskset -c $cpu_cores python "$SCRIPT_DIR/benchmarking.py" --algorithm "$algo" --params "$params" >> "$log_file" 2>&1
    else
        CUDA_VISIBLE_DEVICES=$gpu_device taskset -c $cpu_cores python "$SCRIPT_DIR/benchmarking.py" --algorithm "$algo" >> "$log_file" 2>&1
    fi
    
    if [ $? -eq 0 ]; then
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] Benchmark for algorithm $algo completed successfully." | tee -a "$log_file"
    else
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] Benchmark for algorithm $algo encountered an error." | tee -a "$log_file"
    fi
}

# Function to run jobs in parallel with dynamic scheduling
run_parallel() {
    local items=("$@")
    local total_jobs=${#items[@]}
    local next_job=0
    local active_pids=()
    
    # Start initial batch of jobs up to N_JOBS
    while [ $next_job -lt $total_jobs ] && [ ${#active_pids[@]} -lt $N_JOBS ]; do
        if [ -n "${items[$next_job]}" ]; then
            eval "${items[$next_job]}" &
            active_pids+=($!)
        fi
        next_job=$((next_job + 1))
    done
    
    # Monitor running jobs and start new ones as slots become available
    while [ ${#active_pids[@]} -gt 0 ]; do
        # Check for completed jobs
        for i in "${!active_pids[@]}"; do
            if ! kill -0 ${active_pids[$i]} 2>/dev/null; then
                # Remove completed job from active list
                unset active_pids[$i]
                # Reindex array to remove gaps
                active_pids=("${active_pids[@]}")
                
                # Start a new job if available
                if [ $next_job -lt $total_jobs ]; then
                    if [ -n "${items[$next_job]}" ]; then
                        eval "${items[$next_job]}" &
                        active_pids+=($!)
                    fi
                    next_job=$((next_job + 1))
                fi
            fi
        done
        
        # Short sleep to prevent CPU hogging
        sleep 0.5
    done
    
    # Wait for any remaining background processes
    wait
}

export -f run_benchmark
export -f get_next_gpu
export -f get_next_cpu_cores
export GPU_COUNTER_FILE
export CPU_COUNTER_FILE
export LOG_DIR
export SCRIPT_DIR
export GPU_DEVICES
export CORES_PER_JOB
export TOTAL_CPU_CORES
export N_JOBS

echo "[$(date +"%Y-%m-%d %H:%M:%S")] Starting all benchmarks..."

# Check if we should run with hyperparameters
if [ "$1" == "--with-params" ]; then
    echo "Running benchmarks with specified hyperparameters..."
    
    # Create a temporary array of commands
    COMMANDS=()
    for combo in "${ALGORITHM_HYPERPARAMS[@]}"; do
        algo=${combo%%:*}
        params=${combo#*:}
        COMMANDS+=("run_benchmark \"$algo\" '$params'")
    done
    
    # Check if sequential flag is provided
    if [ "$2" == "--sequential" ]; then
        echo "Running hyperparameter benchmarks sequentially..."
        for cmd in "${COMMANDS[@]}"; do
            eval "$cmd"
        done
    else
        echo "Running hyperparameter benchmarks in parallel..."
        run_parallel "${COMMANDS[@]}"
    fi
    
# Otherwise run standard benchmarks
else
    # Check if sequential flag is provided
    if [ "$1" == "--sequential" ]; then
        echo "Running benchmarks sequentially..."
        for algo in "${ALGORITHMS[@]}"; do
            run_benchmark "$algo" ""
        done
    else
        echo "Running benchmarks in parallel..."
        # Create commands for parallel execution
        COMMANDS=()
        for algo in "${ALGORITHMS[@]}"; do
            COMMANDS+=("run_benchmark \"$algo\" \"\"")
        done
        run_parallel "${COMMANDS[@]}"
    fi
fi

echo "[$(date +"%Y-%m-%d %H:%M:%S")] All benchmarks completed. Logs available in $LOG_DIR/"
