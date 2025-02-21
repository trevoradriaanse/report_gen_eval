#!/bin/bash

# Ensure script stops if any command fails
set -e

.  /opt/anaconda3/etc/profile.d/conda.sh

conda activate report_eval_orion

# Define the inputs and options
DATA_DIR="/home/hltcoe/tadriaanse/SCALE/SCALE2025/report_gen_eval/data"
RESULTS_DIR="/home/hltcoe/tadriaanse/SCALE/SCALE2025/report_gen_eval/results/runs"
PYTHON_SCRIPT="/home/hltcoe/tadriaanse/SCALE/SCALE2025/report_gen_eval/run_report_gen_eval.py"

# Configurable input files
REPORTS_FILE="dev_reports.jsonl"
NUGGETS_FILE="dev_nuggets.jsonl"

# Additional options
BATCH_SIZE=1
PROVIDERS=("together")
#PROVIDERS=("openai" "together" "huggingface")
NUM_RUNS=5  # Set the number of identical runs

# Ensure results directory exists
mkdir -p "$RESULTS_DIR"

# Loop through each combination of inputs and options
for provider in "${PROVIDERS[@]}"; do
    # Run multiple identical experiments
    for run_num in $(seq 1 "$NUM_RUNS"); do
        # Define output results file for each run
        RESULTS_FILE="${RESULTS_DIR}/results_${REPORTS_FILE}_${NUGGETS_FILE}_batch${BATCH_SIZE}_${provider}_run${run_num}_old_prompts_seed.txt"

        # Run the Python script
        echo "Running experiment $run_num with:"
        echo "  Report: $REPORTS_FILE"
        echo "  Nugget: $NUGGETS_FILE"
        echo "  Batch size: $BATCH_SIZE"
        echo "  Provider: $provider"
        echo "  Output file: $RESULTS_FILE"

        python "$PYTHON_SCRIPT" \
            "$DATA_DIR/$REPORTS_FILE" \
            "$DATA_DIR/$NUGGETS_FILE" \
            "$RESULTS_FILE" \
            --batch-size "$BATCH_SIZE" \
            -p "$provider"
    done
done