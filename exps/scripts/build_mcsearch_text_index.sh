#!/usr/bin/env bash

python /home/you/FlashRAG/exps/scripts/prepare_mcsearch_text_corpus.py

CUDA_VISIBLE_DEVICES=1 python -m flashrag.retriever.index_builder \
    --retrieval_method bge \
    --model_path /mnt/data/you/modelscope/bge-large-en-v1.5 \
    --corpus_path /home/you/FlashRAG/exps/idea10/data/datasets/MC-Search/corpus/all_docs.jsonl \
    --save_dir /home/you/FlashRAG/exps/idea10/data/datasets/MC-Search/indexes \
    --use_fp16 \
    --max_length 512 \
    --batch_size 256 \
    --pooling_method cls \
    --faiss_type Flat
