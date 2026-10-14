#!/usr/bin/env bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VLLM_CPU_OMP_THREADS=1

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

RUNNER="${RUNNER:-/home/you/FlashRAG/exps/idea10/run_amb_rewrite_by_source.py}"
TRAIN_PATH="${REFAMB_TRAIN_PATH:-/home/you/FlashRAG/exps/idea10/data/datasets/RefAmb/task_balanced.jsonl}"
read -r -a SOURCES <<< "${REFAMB_SOURCES:-mcsearch oven infoseek crag}"
LIMIT="${REFAMB_LIMIT:-1500}"
SLEEP_BETWEEN="${REFAMB_SLEEP_BETWEEN:-10}"
CONTINUE_ON_ERROR="${REFAMB_CONTINUE_ON_ERROR:-1}"
GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-}}"
DISABLE_AUTO_HISTORY_RESUME="${REFAMB_DISABLE_AUTO_HISTORY_RESUME:-1}"
DISABLE_EVAL="${REFAMB_DISABLE_EVAL:-0}"
read -r -a HISTORY_ROOTS <<< "${REFAMB_HISTORY_ROOTS:-}"
OUTPUT_DIR="${REFAMB_OUTPUT_DIR:-}"

DEFAULT_MODELS="qwen3_vl_32b"
if [[ -n "${REFAMB_MODELS:-}" ]]; then
    read -r -a MODELS <<< "${REFAMB_MODELS}"
else
    read -r -a MODELS <<< "${DEFAULT_MODELS}"
fi

LOG_DIR="${REFAMB_LOG_DIR:-/home/you/FlashRAG/exps/idea10/data/result/run_logs/amb_rewrite_$(date +%Y%m%d_%H%M%S)}"

if [[ -z "${GPU_ID}" ]]; then
    echo "GPU_ID or CUDA_VISIBLE_DEVICES must be set to an absolute CUDA device id, e.g. 1."
    safe_exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export TRAJECTORY_FACT_MODEL_NAME="${TRAJECTORY_FACT_MODEL_NAME:-Qwen/Qwen3-VL-32B-Instruct}"
export TRAJECTORY_FACT_MODEL_PATH="${TRAJECTORY_FACT_MODEL_PATH:-}"
export TRAJECTORY_UTILITY_BACKEND="${TRAJECTORY_UTILITY_BACKEND:-api}"
export TRAJECTORY_UTILITY_MODEL_NAME="${TRAJECTORY_UTILITY_MODEL_NAME:-Qwen/Qwen3-VL-32B-Instruct}"
export TRAJECTORY_UTILITY_MODEL_PATH="${TRAJECTORY_UTILITY_MODEL_PATH:-}"
export TRAJECTORY_UTILITY_API_KEY_ENV="${TRAJECTORY_UTILITY_API_KEY_ENV:-SILICONFLOW_API_KEY}"
export TRAJECTORY_UTILITY_BASE_URL="${TRAJECTORY_UTILITY_BASE_URL:-https://api.siliconflow.cn/v1}"
mkdir -p "${LOG_DIR}"

if [[ -z "${SILICONFLOW_API_KEY:-}" ]]; then
    echo "SILICONFLOW_API_KEY must be set when using the default Qwen-VL-32B API configuration."
    safe_exit 1
fi

echo "Models: ${MODELS[*]}"
echo "Runner: ${RUNNER}"
echo "Train path: ${TRAIN_PATH}"
echo "Sources: ${SOURCES[*]}"
echo "Limit: ${LIMIT:-<unset>}"
echo "Log dir: ${LOG_DIR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Trajectory fact model: ${TRAJECTORY_FACT_MODEL_NAME}"
echo "Trajectory fact model path: ${TRAJECTORY_FACT_MODEL_PATH:-<empty>}"
echo "Trajectory utility backend: ${TRAJECTORY_UTILITY_BACKEND}"
echo "Trajectory utility model: ${TRAJECTORY_UTILITY_MODEL_NAME}"
echo "Trajectory utility model path: ${TRAJECTORY_UTILITY_MODEL_PATH:-<empty>}"
echo "Trajectory utility base url: ${TRAJECTORY_UTILITY_BASE_URL}"
echo "Trajectory utility api key env: ${TRAJECTORY_UTILITY_API_KEY_ENV}"
echo "Disable auto history resume: ${DISABLE_AUTO_HISTORY_RESUME}"
echo "Disable eval: ${DISABLE_EVAL}"
echo "Extra history roots: ${HISTORY_ROOTS[*]:-<unset>}"
echo "Fixed output dir: ${OUTPUT_DIR:-<unset>}"
echo

if [[ -n "${LIMIT}" ]]; then
    if ! [[ "${LIMIT}" =~ ^[0-9]+$ ]]; then
        echo "REFAMB_LIMIT must be a non-negative integer or empty, got: ${LIMIT}"
        safe_exit 1
    fi
    if [[ "${LIMIT}" -le 0 ]]; then
        echo "REFAMB_LIMIT must be positive when set, got: ${LIMIT}"
        safe_exit 1
    fi
fi

for model in "${MODELS[@]}"; do
    model_log_dir="${LOG_DIR}/${model}"
    mkdir -p "${model_log_dir}"
    echo "============================================================"
    echo "Model: ${model}"
    echo "Model log dir: ${model_log_dir}"
    echo "============================================================"

    for source in "${SOURCES[@]}"; do
        log_path="${model_log_dir}/${source}.log"
        echo "[$(date '+%F %T')] Start model=${model} source=${source}. Log: ${log_path}"

        cmd=(
            python "${RUNNER}"
            --model "${model}"
            --source "${source}"
            --train-path "${TRAIN_PATH}"
        )
        if [[ -n "${LIMIT}" ]]; then
            cmd+=(--limit "${LIMIT}")
        fi
        cmd+=(--disable-resume --disable-incremental)
        if [[ "${DISABLE_AUTO_HISTORY_RESUME}" == "1" ]]; then
            cmd+=(--disable-auto-history-resume)
        fi
        if [[ "${DISABLE_EVAL}" == "1" ]]; then
            cmd+=(--disable-eval)
        fi
        if [[ -n "${OUTPUT_DIR}" ]]; then
            cmd+=(--output-dir "${OUTPUT_DIR}")
        fi
        for history_root in "${HISTORY_ROOTS[@]}"; do
            if [[ -n "${history_root}" ]]; then
                cmd+=(--history-root "${history_root}")
            fi
        done

        set +e
        "${cmd[@]}" 2>&1 | tee "${log_path}"
        status="${PIPESTATUS[0]}"
        set -e

        if [[ "${status}" -ne 0 ]]; then
            echo "[$(date '+%F %T')] model=${model} source=${source} failed with exit code ${status}." | tee -a "${log_path}"
            if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
                echo "Stop because CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR}."
                safe_exit "${status}"
            fi
            echo "Continue to next source because CONTINUE_ON_ERROR=1."
        else
            echo "[$(date '+%F %T')] Finished model=${model} source=${source}."
        fi

        echo "Waiting ${SLEEP_BETWEEN}s for the Python process to exit cleanly and release GPU memory..."
        sleep "${SLEEP_BETWEEN}"
        if command -v nvidia-smi >/dev/null 2>&1; then
            nvidia-smi || true
        fi
        echo
    done
done

echo "[$(date '+%F %T')] All requested sources finished. Logs: ${LOG_DIR}"
