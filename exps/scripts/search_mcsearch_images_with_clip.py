import argparse
import json
from pathlib import Path

from flashrag.retriever.retriever import MultiModalRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search MC-Search image corpus with a CLIP image query.")
    parser.add_argument("--query-image", type=Path, required=True, help="Path to the query image.")
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=Path("/home/you/FlashRAG/exps/idea10/data/datasets/mcsearch/corpus/image_corpus.parquet"),
        help="Path to the image corpus parquet.",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=Path("/home/you/FlashRAG/exps/idea10/data/datasets/mcsearch/indexes/clip_Flat_image.index"),
        help="Path to the CLIP image index.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/mnt/data/you/modelscope/clip-vit-large-patch14"),
        help="Path to the local CLIP model.",
    )
    parser.add_argument(
        "--all-image-infos",
        type=Path,
        default=Path("/home/you/FlashRAG/exps/idea10/data/datasets/mcsearch/corpus/all_image_infos.json"),
        help="Optional title fallback source.",
    )
    parser.add_argument("--topk", type=int, default=5, help="Number of retrieved results.")
    parser.add_argument("--batch-size", type=int, default=32, help="Retriever batch size.")
    parser.add_argument("--faiss-gpu", action="store_true", help="Load FAISS index onto GPU.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save retrieval results as JSON.",
    )
    return parser.parse_args()


def load_title_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(item["image_id"]): str(item.get("title", "")) for item in data}


def build_retriever(args: argparse.Namespace) -> MultiModalRetriever:
    config = {
        "retrieval_method": "clip",
        "retrieval_model_path": str(args.model_path),
        "index_path": None,
        "corpus_path": str(args.corpus_path),
        "multimodal_index_path_dict": {
            "text": None,
            "image": str(args.index_path),
        },
        "retrieval_topk": args.topk,
        "retrieval_batch_size": args.batch_size,
        "faiss_gpu": args.faiss_gpu,
        "save_retrieval_cache": False,
        "use_retrieval_cache": False,
        "retrieval_cache_path": None,
        "use_reranker": False,
        "save_dir": "/tmp",
        "silent": False,
        "silent_retrieval": False,
    }
    return MultiModalRetriever(config)


def main() -> None:
    args = parse_args()
    retriever = build_retriever(args)
    title_map = load_title_map(args.all_image_infos)

    docs, scores = retriever.search(str(args.query_image), target_modal="image", return_score=True)

    results = []
    for rank, (doc, score) in enumerate(zip(docs, scores), start=1):
        image_id = str(doc.get("id", ""))
        title = str(doc.get("title", "")).strip() or title_map.get(image_id, "")
        raw_image = doc.get("image")
        if isinstance(raw_image, str):
            image_field = raw_image
        else:
            image_field = None

        item = {
            "rank": rank,
            "score": float(score),
            "id": image_id,
            "title": title,
            "image": image_field,
            "contents": doc.get("contents", ""),
        }
        results.append(item)
        print(json.dumps(item, ensure_ascii=False))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
