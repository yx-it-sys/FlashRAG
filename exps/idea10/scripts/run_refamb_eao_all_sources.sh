#!/usr/bin/env bash
set -euo pipefail

RUNNER="${RUNNER:-/home/you/FlashRAG/exps/idea10/run_refamb_eao_by_source.py}"
SOURCE="${SOURCE:-mcsearch crag}"
MODEL="${MODEL:-qwen2_5_7b}"
TRAIN_PATH="${TRAIN_PATH:-/home/you/FlashRAG/exps/idea10/data/datasets/RefAmb/task_balanced.jsonl}"
LIMIT="${LIMIT:-1500}"
GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-}}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-10}"
POST_RUN_CLEANUP_SLEEP="${POST_RUN_CLEANUP_SLEEP:-5}"
DISABLE_AUTO_HISTORY_RESUME="${DISABLE_AUTO_HISTORY_RESUME:-1}"
CONFIG_SNAPSHOT_ROOT="${CONFIG_SNAPSHOT_ROOT:-/tmp/refamb_config_snapshots}"
DEFAULT_CONFIG_SWEEP_ROOT=""
DEFAULT_SWEEP_CONFIGS="${SWEEP_CONFIGS:-}"
if [[ "${MODEL}" == "qwen2_5_7b" ]]; then
    DEFAULT_CONFIG_SWEEP_ROOT="/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen2_5_7b/enhanced"
    DEFAULT_SWEEP_CONFIGS="${DEFAULT_SWEEP_CONFIGS:-sim/enhanced_thr_090}"
elif [[ "${MODEL}" == "gpt" ]]; then
    DEFAULT_CONFIG_SWEEP_ROOT="/home/you/FlashRAG/exps/idea10/configs/configs_refamb/gpt"
    DEFAULT_SWEEP_CONFIGS="${DEFAULT_SWEEP_CONFIGS:-enhanced}"
fi
CONFIG_SWEEP_ROOT="${CONFIG_SWEEP_ROOT:-${DEFAULT_CONFIG_SWEEP_ROOT}}"
SWEEP_CONFIGS="${SWEEP_CONFIGS:-${DEFAULT_SWEEP_CONFIGS}}"
read -r -a SOURCES <<< "${SOURCE}"

resolve_config_dir() {
    local sweep_root="$1"
    local config_spec="$2"

    if [[ -z "${config_spec}" ]]; then
        return 1
    fi

    if [[ -d "${config_spec}" ]]; then
        printf '%s\n' "${config_spec}"
        return 0
    fi

    if [[ -n "${sweep_root}" && -d "${sweep_root}/${config_spec}" ]]; then
        printf '%s\n' "${sweep_root}/${config_spec}"
        return 0
    fi

    return 1
}

cleanup_runtime_processes() {
    pkill -f vllm >/dev/null 2>&1 || true
    pkill -f api_server >/dev/null 2>&1 || true
    pkill -f openai.api_server >/dev/null 2>&1 || true
    sleep "${POST_RUN_CLEANUP_SLEEP}"
}

create_config_snapshot() {
    local config_dir="$1"
    local config_spec="$2"
    local safe_label="${config_spec//\//_}"
    local snapshot_dir="${CONFIG_SNAPSHOT_ROOT}/${MODEL}/${safe_label}_$(date +%Y%m%d_%H%M%S)_$$"

    mkdir -p "${CONFIG_SNAPSHOT_ROOT}/${MODEL}"
    cp -a "${config_dir}" "${snapshot_dir}"
    printf '%s\n' "${snapshot_dir}"
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
echo "Config sweep root: ${CONFIG_SWEEP_ROOT:-<disabled>}"
echo "Config snapshot root: ${CONFIG_SNAPSHOT_ROOT}"
echo "Sweep configs: ${SWEEP_CONFIGS:-<unset>}"
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

if [[ -z "${CONFIG_SWEEP_ROOT}" ]]; then
    echo "CONFIG_SWEEP_ROOT is empty. Set it explicitly or use MODEL=qwen2_5_7b."
    exit 1
fi

if [[ -z "${SWEEP_CONFIGS}" ]]; then
    echo "SWEEP_CONFIGS must be set explicitly when CONFIG_SWEEP_ROOT is enabled."
    exit 1
fi

read -r -a SWEEP_CONFIG_ARRAY <<< "${SWEEP_CONFIGS}"

for ((config_idx=0; config_idx<${#SWEEP_CONFIG_ARRAY[@]}; config_idx++)); do
    config_spec="${SWEEP_CONFIG_ARRAY[config_idx]}"
    config_dir="$(resolve_config_dir "${CONFIG_SWEEP_ROOT}" "${config_spec}")" || {
        echo "Config spec not found: ${config_spec}"
        exit 1
    }
    snapshot_config_dir="$(create_config_snapshot "${config_dir}" "${config_spec}")"
    sweep_label="$(basename "${config_dir}")"

    echo "[$(date '+%F %T')] Start sweep config=${config_spec} (${config_idx} / ${#SWEEP_CONFIG_ARRAY[@]})"
    echo "[$(date '+%F %T')] Config snapshot dir=${snapshot_config_dir}"

    for ((i=0; i<${#SOURCES[@]}; i++)); do
        source="${SOURCES[i]}"
        config_path="${snapshot_config_dir}/config_stage_${source}.yaml"
        if [[ ! -f "${config_path}" ]]; then
            echo "Missing config file for source=${source}: ${config_path}"
            exit 1
        fi

        cmd=(
            python "${RUNNER}"
            --source "${source}"
            --model "${MODEL}"
            --train-path "${TRAIN_PATH}"
            --config-path "${config_path}"
            --save-note-suffix "${sweep_label}"
        )
        if [[ -n "${LIMIT}" ]]; then
            cmd+=(--limit "${LIMIT}")
        fi
        if [[ "${DISABLE_AUTO_HISTORY_RESUME}" == "1" ]]; then
            cmd+=(--disable-auto-history-resume)
        fi

        echo "[$(date '+%F %T')] Start source=${source} config=${config_path}"
        "${cmd[@]}"
        echo "[$(date '+%F %T')] Finished source=${source} config=${config_path}"

        echo "[$(date '+%F %T')] Cleaning runtime processes before next stage..."
        cleanup_runtime_processes
        if (( i + 1 < ${#SOURCES[@]} )); then
            echo "Waiting ${SLEEP_BETWEEN}s before next stage..."
            sleep "${SLEEP_BETWEEN}"
        fi
    done

    echo "[$(date '+%F %T')] Finished sweep config=${config_spec}"
    if (( config_idx + 1 < ${#SWEEP_CONFIG_ARRAY[@]} )); then
        echo "Waiting ${SLEEP_BETWEEN}s before next hyperparameter config..."
        sleep "${SLEEP_BETWEEN}"
    fi
done

echo "[$(date '+%F %T')] Finished all requested sources."
