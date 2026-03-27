CUDA_VISIBLE_DEVICES=1 python -m flashrag.retriever.index_builder \
    --retrieval_method clip \
    --model_path /mnt/data/you/modelscope/clip-vit-large-patch14 \
    --corpus_path /home/you/FlashRAG/exps/idea10/data/datasets/mcsearch/corpus/image_corpus.parquet \
    --save_dir /home/you/FlashRAG/exps/idea10/data/datasets/mcsearch/image_indexes \
    --use_fp16 \
    --max_length 256 \
    --batch_size 512 \
    --faiss_type Flat \
    --index_modal image
