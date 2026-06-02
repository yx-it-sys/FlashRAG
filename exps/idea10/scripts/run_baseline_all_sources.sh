#!/usr/bin/env bash
# set -euo pipefail

# If this script is sourced, avoid `exit` killing the interactive shell.
is_sourced=0
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    is_sourced=1
fi

safe_exit() {
    local code="${1:-0}"
    if [[ "${is_sourced}" -eq 1 ]]; then
        return "${code}"
    fi
    exit "${code}"
}

# Use REFAMB_* names so stale variables from sourced EAO wrappers cannot hijack this baseline run.
MODEL="${REFAMB_MODEL:-gpt}"
RUNNER="${REFAMB_RUNNER:-/home/you/FlashRAG/exps/idea10/run_refamb_by_source.py}"
TRAIN_PATH="${REFAMB_TRAIN_PATH:-/home/you/FlashRAG/exps/idea10/data/datasets/RefAmb/task_balanced_analysis_subset.jsonl}"
LOG_DIR="${REFAMB_LOG_DIR:-/home/you/FlashRAG/exps/idea10/data/result/run_logs/refamb_${MODEL}_$(date +%Y%m%d_%H%M%S)}"
read -r -a SOURCES <<< "${REFAMB_SOURCES:-oven}"
SLEEP_BETWEEN="${REFAMB_SLEEP_BETWEEN:-10}"
CONTINUE_ON_ERROR="${REFAMB_CONTINUE_ON_ERROR:-1}"
GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-}}"

if [[ -z "${GPU_ID}" ]]; then
    echo "GPU_ID or CUDA_VISIBLE_DEVICES must be set to an absolute CUDA device id, e.g. 1."
    safe_exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

mkdir -p "${LOG_DIR}"

echo "Model: ${MODEL}"
echo "Runner: ${RUNNER}"
echo "Train path: ${TRAIN_PATH}"
echo "Sources: ${SOURCES[*]}"
echo "Log dir: ${LOG_DIR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo

for source in "${SOURCES[@]}"; do
    log_path="${LOG_DIR}/${source}.log"
    echo "[$(date '+%F %T')] Start source=${source}. Log: ${log_path}"

    set +e
    python "${RUNNER}" \
        --model "${MODEL}" \
        --source "${source}" \
        --train-path "${TRAIN_PATH}" \
        2>&1 | tee "${log_path}"
    status="${PIPESTATUS[0]}"
    set -e

    if [[ "${status}" -ne 0 ]]; then
        echo "[$(date '+%F %T')] source=${source} failed with exit code ${status}." | tee -a "${log_path}"
        if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
            echo "Stop because CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR}."
            safe_exit "${status}"
        fi
        echo "Continue to next source because CONTINUE_ON_ERROR=1."
    else
        echo "[$(date '+%F %T')] Finished source=${source}."
    fi

    echo "Waiting ${SLEEP_BETWEEN}s for the Python process to exit cleanly and release GPU memory..."
    sleep "${SLEEP_BETWEEN}"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi || true
    fi
    echo
done

echo "[$(date '+%F %T')] All requested sources finished. Logs: ${LOG_DIR}"
