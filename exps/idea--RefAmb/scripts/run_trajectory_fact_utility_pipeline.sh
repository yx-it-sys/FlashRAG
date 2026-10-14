#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/you/FlashRAG/exps/idea10"
CUDA_VISIBLE_DEVICES_VALUE="0"
STAGE_COOLDOWN_SECONDS="10"

INPUT_JSONL="/home/you/FlashRAG/exps/idea10/data/result/cross-model-rte/label/human/TQS/human_label_150.trajectories.jsonl"

FACT_MODEL_NAME="Qwen2.5-7B-Instruct"
FACT_MODEL_PATH="/mnt/data/you/modelscope/Qwen2.5-7B-Instruct"
FACT_OUTPUT_DIR="/home/you/FlashRAG/exps/idea10/data/result/cross-model-rte/label/human/TQS/qwen25_fact_stage"

UTILITY_BACKEND="api"
UTILITY_MODEL_NAME="Qwen/Qwen3-VL-32B-Instruct"
UTILITY_MODEL_PATH=""
UTILITY_OUTPUT_DIR="/home/you/FlashRAG/exps/idea10/data/result/cross-model-rte/label/human/TQS/qwen25_fact_qwen3vl32b_utility_eval"

export TRAJECTORY_FACT_INPUT="${INPUT_JSONL}"
export TRAJECTORY_FACT_MODEL_NAME="${FACT_MODEL_NAME}"
export TRAJECTORY_FACT_MODEL_PATH="${FACT_MODEL_PATH}"
export TRAJECTORY_FACT_OUTPUT_DIR="${FACT_OUTPUT_DIR}"

export TRAJECTORY_UTILITY_INPUT="${FACT_OUTPUT_DIR}/trajectory_facts.jsonl"
export TRAJECTORY_UTILITY_BACKEND="${UTILITY_BACKEND}"
export TRAJECTORY_UTILITY_MODEL_NAME="${UTILITY_MODEL_NAME}"
export TRAJECTORY_UTILITY_MODEL_PATH="${UTILITY_MODEL_PATH}"
export TRAJECTORY_UTILITY_OUTPUT_DIR="${UTILITY_OUTPUT_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

FACT_OUTPUT_JSONL="${FACT_OUTPUT_DIR}/trajectory_facts.jsonl"
FACT_OUTPUT_SUMMARY="${FACT_OUTPUT_DIR}/trajectory_facts_summary.json"

fact_stage_is_complete() {
  local input_path="$1"
  local output_path="$2"
  local summary_path="$3"
  local expected_model_name="$4"
  local expected_model_path="$5"
  if [[ ! -f "${input_path}" || ! -f "${output_path}" || ! -f "${summary_path}" ]]; then
    return 1
  fi
  local input_count
  local output_count
  local summary_model_name
  local summary_model_path
  input_count="$(wc -l < "${input_path}" | tr -d '[:space:]')"
  output_count="$(wc -l < "${output_path}" | tr -d '[:space:]')"
  summary_model_name="$(python - <<PY
import json
from pathlib import Path
path = Path(${summary_path@Q})
data = json.loads(path.read_text(encoding="utf-8"))
print(data.get("meta", {}).get("local_fact_model", ""))
PY
)"
  summary_model_path="$(python - <<PY
import json
from pathlib import Path
path = Path(${summary_path@Q})
data = json.loads(path.read_text(encoding="utf-8"))
print(data.get("meta", {}).get("local_fact_model_path", ""))
PY
)"
  [[ -n "${input_count}" && -n "${output_count}" && "${input_count}" == "${output_count}" \
     && "${summary_model_name}" == "${expected_model_name}" \
     && "${summary_model_path}" == "${expected_model_path}" ]]
}

if fact_stage_is_complete \
  "${INPUT_JSONL}" \
  "${FACT_OUTPUT_JSONL}" \
  "${FACT_OUTPUT_SUMMARY}" \
  "${TRAJECTORY_FACT_MODEL_NAME}" \
  "${TRAJECTORY_FACT_MODEL_PATH}"; then
  echo "[Pipeline] Stage 1 already complete, skipping"
  echo "[Pipeline] existing_facts_output=${FACT_OUTPUT_JSONL}"
else
  echo "[Pipeline] Stage 1: extract facts"
  echo "[Pipeline] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "[Pipeline] fact_model=${TRAJECTORY_FACT_MODEL_NAME}"
  echo "[Pipeline] fact_model_path=${TRAJECTORY_FACT_MODEL_PATH}"
  python "${ROOT_DIR}/scripts/extract_trajectory_facts.py"
fi

echo "[Pipeline] Stage 1 process exited"
if [[ "${STAGE_COOLDOWN_SECONDS}" != "0" ]]; then
  echo "[Pipeline] Cooling down ${STAGE_COOLDOWN_SECONDS}s before Stage 2"
  sleep "${STAGE_COOLDOWN_SECONDS}"
fi

echo "[Pipeline] Stage 2: compute utility and RTE"
echo "[Pipeline] VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD}"
echo "[Pipeline] utility_backend=${TRAJECTORY_UTILITY_BACKEND}"
echo "[Pipeline] utility_model=${TRAJECTORY_UTILITY_MODEL_NAME}"
echo "[Pipeline] utility_model_path=${TRAJECTORY_UTILITY_MODEL_PATH}"
python "${ROOT_DIR}/scripts/compute_trajectory_utility_rte.py"

echo "[Pipeline] Done"
echo "[Pipeline] facts_output=${FACT_OUTPUT_DIR}/trajectory_facts.jsonl"
echo "[Pipeline] final_output=${UTILITY_OUTPUT_DIR}/trajectory_quality_samples.jsonl"
