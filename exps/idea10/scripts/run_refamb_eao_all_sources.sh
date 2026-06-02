#!/usr/bin/env bash
set -euo pipefail

RUNNER="${RUNNER:-/home/you/FlashRAG/exps/idea10/run_refamb_eao_by_source.py}"
TRAIN_PATH="${TRAIN_PATH:-/home/you/FlashRAG/exps/idea10/data/datasets/RefAmb/task_balanced_analysis_subset.jsonl}"
SOURCES=(${SOURCES:-mcsearch crag oven infoseek})
LIMIT="${LIMIT:-1500}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-10}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"

if [[ -n "${MODELS:-}" ]]; then
    read -r -a MODELS <<< "${MODELS}"
else
    MODELS=("GPT-5.1")
fi
declare -A CONFIG_ROOT_MAP=(
    ["GPT-5.1"]="/home/you/FlashRAG/exps/idea10/configs/configs_refamb/gpt/enhanced"
    # ["InternVL3.5-8B"]="/home/you/FlashRAG/exps/idea10/configs/configs_refamb/intervl3_5_8b"
    # ["Qwen3-vl-4B"]="/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen3_vl_4b"
    # ["Qwen3-vl-8b"]="/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen3_vl_8b"
)

GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-}}"
CONFIG_GROUP="${CONFIG_GROUP:-enhanced}"
LOG_ROOT="${LOG_ROOT:-/home/you/FlashRAG/exps/idea10/data/result/run_logs/refamb_eao_$(date +%Y%m%d_%H%M%S)}"

if [[ -z "${GPU_ID}" ]]; then
    echo "GPU_ID or CUDA_VISIBLE_DEVICES must be set to an absolute CUDA device id, e.g. 1."
    exit 1
fi
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

mkdir -p "${LOG_ROOT}"

echo "Runner: ${RUNNER}"
echo "Train path: ${TRAIN_PATH}"
echo "Sources: ${SOURCES[*]}"
echo "Models: ${MODELS[*]}"
echo "Limit: ${LIMIT:-<unset>}"
echo "Log root: ${LOG_ROOT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Config group: ${CONFIG_GROUP}"
echo

source_count="${#SOURCES[@]}"
limit_per_source=()
if [[ -n "${LIMIT}" ]]; then
    if ! [[ "${LIMIT}" =~ ^[0-9]+$ ]]; then
        echo "LIMIT must be a non-negative integer or empty, got: ${LIMIT}"
        exit 1
    fi
    if [[ "${LIMIT}" -lt 0 ]]; then
        echo "LIMIT must be non-negative, got: ${LIMIT}"
        exit 1
    fi

    base_limit=$((LIMIT / source_count))
    remainder=$((LIMIT % source_count))
    for ((i = 0; i < source_count; i++)); do
        per_source=$base_limit
        if [[ "${i}" -lt "${remainder}" ]]; then
            per_source=$((per_source + 1))
        fi
        limit_per_source+=("${per_source}")
    done
fi

for model in "${MODELS[@]}"; do
    config_root="${CONFIG_ROOT_MAP[${model}]:-}"
    if [[ -z "${config_root}" ]]; then
        echo "No config root mapping found for model: ${model}"
        exit 1
    fi
    if [[ -d "${config_root}/enhanced" ]]; then
        config_source_dir="${config_root}/enhanced"
    else
        config_source_dir="${config_root}"
    fi
    if [[ ! -d "${config_source_dir}" ]]; then
        echo "Config source directory does not exist: ${config_source_dir}"
        exit 1
    fi

    snapshot_suffix="${CONFIG_GROUP}_snapshot_$(date +%Y%m%d_%H%M%S)_$$"
    config_runtime_dir_base="${CONFIG_RUNTIME_DIR_BASE:-/tmp/refamb_config_snapshots/${model}}"
    config_runtime_dir="${config_runtime_dir_base}/${snapshot_suffix}"
    if [[ -e "${config_runtime_dir}" ]]; then
        echo "Unexpected existing config snapshot path: ${config_runtime_dir}"
        exit 1
    fi

    mkdir -p "${config_runtime_dir_base}"
    cp -a "${config_source_dir}" "${config_runtime_dir}"

    model_log_dir="${LOG_ROOT}/${model}"
    mkdir -p "${model_log_dir}"

    echo "============================================================"
    echo "Model: ${model}"
    echo "Config source: ${config_source_dir}"
    echo "Config snapshot dir: ${config_runtime_dir}"
    echo "Model log dir: ${model_log_dir}"
    echo "============================================================"

    for idx in "${!SOURCES[@]}"; do
        source="${SOURCES[$idx]}"
        source_limit=""
        if [[ -n "${LIMIT}" ]]; then
            source_limit="${limit_per_source[$idx]}"
            if [[ "${source_limit}" -le 0 ]]; then
                echo "[$(date '+%F %T')] Skip source=${source} because assigned limit is 0."
                echo
                continue
            fi
        fi

        log_path="${model_log_dir}/${source}.log"
        if [[ -n "${source_limit}" ]]; then
            echo "[$(date '+%F %T')] Start model=${model} source=${source} limit=${source_limit}. Log: ${log_path}"
        else
            echo "[$(date '+%F %T')] Start model=${model} source=${source}. Log: ${log_path}"
        fi

        cmd=(
            python "${RUNNER}"
            --model "${model}"
            --source "${source}"
            --train-path "${TRAIN_PATH}"
            --config-dir "${config_runtime_dir}"
        )
        if [[ -n "${source_limit}" ]]; then
            cmd+=(--limit "${source_limit}")
        fi

        set +e
        "${cmd[@]}" 2>&1 | tee "${log_path}"
        status="${PIPESTATUS[0]}"
        set -e

        if [[ "${status}" -ne 0 ]]; then
            echo "[$(date '+%F %T')] model=${model} source=${source} failed with exit code ${status}." | tee -a "${log_path}"
            if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
                echo "Stop because CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR}."
                exit "${status}"
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

    echo "[$(date '+%F %T')] Finished model=${model}. Logs: ${model_log_dir}"
    echo
done

echo "[$(date '+%F %T')] All requested models and sources finished. Logs: ${LOG_ROOT}"
