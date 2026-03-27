TMP_CORPUS=/tmp/mcsearch_clip_text_corpus.jsonl

python - <<'PY'
import json
from pathlib import Path

src = Path("/mnt/data/you/datasets/mcsearch/corpus/all_docs.jsonl")
dst = Path("/tmp/mcsearch_clip_text_corpus.jsonl")

with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
    for line in fin:
        if not line.strip():
            continue
        obj = json.loads(line)
        obj["text"] = obj["contents"]
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
PY

CUDA_VISIBLE_DEVICES=1 python -m flashrag.retriever.index_builder \
    --retrieval_method clip \
    --model_path /mnt/data/you/modelscope/clip-vit-large-patch14 \
    --corpus_path ${TMP_CORPUS}\
    --save_dir /mnt/data/you/datasets/mcsearch \
    --use_fp16 \
    --max_length 256 \
    --batch_size 512 \
    --pooling_method cls \
    --faiss_type Flat \
    --save_embedding \
    --index_modal text
