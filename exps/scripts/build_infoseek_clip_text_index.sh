#!/usr/bin/env bash
set -euo pipefail

CORPUS_PATH="/home/you/FlashRAG/exps/idea10/data/datasets/infoseek_val/wiki_100k_flatten.jsonl"
SAVE_DIR="/home/you/FlashRAG/exps/idea10/data/datasets/infoseek_val/indexes"
MODEL_PATH="/mnt/data/you/modelscope/clip-vit-large-patch14"

mkdir -p "${SAVE_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -m flashrag.retriever.index_builder \
  --retrieval_method clip \
  --model_path "${MODEL_PATH}" \
  --corpus_path "${CORPUS_PATH}" \
  --save_dir "${SAVE_DIR}" \
  --use_fp16 \
  --max_length 256 \
  --batch_size 512 \
  --faiss_type Flat \
  --index_modal text
