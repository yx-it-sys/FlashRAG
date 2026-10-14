#!/usr/bin/env bash
set -euo pipefail

RUNNER="${RUNNER:-/home/you/FlashRAG/exps/idea10/run_refamb_mm_by_source.py}"
SOURCE="${SOURCE:-mcsearch crag}"
MODEL="${MODEL:-qwen2_5_7b}"
TRAIN_PATH="${TRAIN_PATH:-/home/you/FlashRAG/exps/idea10/data/datasets/RefAmb/task_balanced.jsonl}"
LIMIT="${LIMIT:-1500}"
GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-}}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-10}"
POST_RUN_CLEANUP_SLEEP="${POST_RUN_CLEANUP_SLEEP:-5}"
DISABLE_AUTO_HISTORY_RESUME="${DISABLE_AUTO_HISTORY_RESUME:-1}"
read -r -a SOURCES <<< "${SOURCE}"

cleanup_runtime_processes() {
    pkill -f vllm >/dev/null 2>&1 || true
    pkill -f api_server >/dev/null 2>&1 || true
    pkill -f openai.api_server >/dev/null 2>&1 || true
    sleep "${POST_RUN_CLEANUP_SLEEP}"
}

if [[ -z "${GPU_ID}" ]]; then
    echo "GPU_ID or CUDA_VISIBLE_DEVICES must be set to an absolute CUDA device id, e.g. 1."
    exit 1
fi
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "Runner: ${RUNNER}"
echo "Sources: ${SOURCES[*]}"
echo "Model: ${MODEL}"
echo "Train path: ${TRAIN_PATH}"
echo "Limit: ${LIMIT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Post-run cleanup sleep: ${POST_RUN_CLEANUP_SLEEP}"
echo "Disable auto history resume: ${DISABLE_AUTO_HISTORY_RESUME}"
echo

if [[ -n "${LIMIT}" ]]; then
    if ! [[ "${LIMIT}" =~ ^[0-9]+$ ]]; then
        echo "LIMIT must be a non-negative integer, got: ${LIMIT}"
        exit 1
    fi
fi

for ((i=0; i<${#SOURCES[@]}; i++)); do
    source="${SOURCES[i]}"

    cmd=(
        python "${RUNNER}"
        --source "${source}"
        --model "${MODEL}"
        --train-path "${TRAIN_PATH}"
    )
    if [[ -n "${LIMIT}" ]]; then
        cmd+=(--limit "${LIMIT}")
    fi
    if [[ "${DISABLE_AUTO_HISTORY_RESUME}" == "1" ]]; then
        cmd+=(--disable-auto-history-resume)
    fi

    echo "[$(date '+%F %T')] Start source=${source} model=${MODEL}"
    "${cmd[@]}"
    echo "[$(date '+%F %T')] Finished source=${source} model=${MODEL}"

    echo "[$(date '+%F %T')] Cleaning runtime processes before next stage..."
    cleanup_runtime_processes
    if (( i + 1 < ${#SOURCES[@]} )); then
        echo "Waiting ${SLEEP_BETWEEN}s before next stage..."
        sleep "${SLEEP_BETWEEN}"
    fi
done

echo "[$(date '+%F %T')] Finished all requested sources."
