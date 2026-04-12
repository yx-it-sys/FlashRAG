#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from flashrag.retriever.encoder import Encoder


DEFAULT_CONFIG = Path(
    "/home/you/FlashRAG/exps/idea10/data/result/crag_mm_2026_03_31_14_06_experiment/config.yaml"
)
DEFAULT_OUTPUT = Path(
    "/home/you/FlashRAG/exps/idea10/data/result/crag_mm_2026_03_31_14_06_experiment/negation_similarity_test.json"
)
DEFAULT_BATCH_SIZE = 16

DEFAULT_PAIRS = [
    {
        "id": "capital_france",
        "sentence": "Paris is the capital of France.",
        "negated": "Paris is not the capital of France.",
    },
    {
        "id": "cat_mat",
        "sentence": "The cat is sitting on the mat.",
        "negated": "The cat is not sitting on the mat.",
    },
    {
        "id": "water_boils",
        "sentence": "Water boils at 100 degrees Celsius.",
        "negated": "Water does not boil at 100 degrees Celsius.",
    },
    {
        "id": "owns_brand",
        "sentence": "General Motors owns Chevrolet.",
        "negated": "General Motors does not own Chevrolet.",
    },
    {
        "id": "building_style",
        "sentence": "The New York Stock Exchange building is in the Beaux-Arts style.",
        "negated": "The New York Stock Exchange building is not in the Beaux-Arts style.",
    },
    {
        "id": "plant_blooming",
        "sentence": "Magnolia grandiflora typically blooms from late spring to summer.",
        "negated": "Magnolia grandiflora does not typically bloom from late spring to summer.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test whether a dense encoder assigns high cosine similarity to a sentence and its negated variant."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pairs", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser.parse_args()


def load_encoder_from_config(config_path: Path) -> Encoder:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    retriever_cfg = config.get("text_retriever_config") or {}
    retrieval_method = retriever_cfg.get("retrieval_method") or config.get("retrieval_method")
    retrieval_model_path = retriever_cfg.get("retrieval_model_path") or config.get("retrieval_model_path")
    pooling_method = retriever_cfg.get("retrieval_pooling_method") or config.get(
        "retrieval_pooling_method", "mean"
    )
    max_length = retriever_cfg.get("retrieval_query_max_length") or config.get(
        "retrieval_query_max_length", 64
    )
    use_fp16 = retriever_cfg.get("retrieval_use_fp16")
    if use_fp16 is None:
        use_fp16 = config.get("retrieval_use_fp16", True)
    instruction = retriever_cfg.get("instruction")
    if instruction is None:
        instruction = config.get("instruction")

    if not retrieval_method or not retrieval_model_path:
        raise ValueError(f"Invalid text retriever config in {config_path}")

    return Encoder(
        model_name=retrieval_method,
        model_path=retrieval_model_path,
        pooling_method=pooling_method,
        max_length=max_length,
        use_fp16=use_fp16,
        instruction=instruction,
        silent=True,
    )


def load_pairs(path: Path | None) -> list[dict]:
    if path is None:
        return list(DEFAULT_PAIRS)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Pairs file must be a JSON list.")
    pairs = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Pair #{idx} must be an object.")
        sentence = item.get("sentence")
        negated = item.get("negated")
        if not isinstance(sentence, str) or not isinstance(negated, str):
            raise ValueError(f"Pair #{idx} must contain string fields `sentence` and `negated`.")
        pair_id = item.get("id", f"pair_{idx}")
        pairs.append({"id": pair_id, "sentence": sentence, "negated": negated})
    return pairs


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def main() -> None:
    args = parse_args()
    pairs = load_pairs(args.pairs)
    encoder = load_encoder_from_config(args.config)

    texts = []
    for pair in pairs:
        texts.append(pair["sentence"])
        texts.append(pair["negated"])

    embeddings = encoder.encode(texts, batch_size=args.batch_size, is_query=True)

    results = []
    for idx, pair in enumerate(pairs):
        sent_emb = embeddings[idx * 2]
        neg_emb = embeddings[idx * 2 + 1]
        similarity = cosine_similarity(sent_emb, neg_emb)
        results.append(
            {
                "id": pair["id"],
                "sentence": pair["sentence"],
                "negated": pair["negated"],
                "cosine_similarity": similarity,
            }
        )

    summary = {
        "config": str(args.config),
        "num_pairs": len(results),
        "mean_cosine_similarity": float(np.mean([item["cosine_similarity"] for item in results])) if results else 0.0,
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Encoder config: {args.config}")
    print(f"Output: {args.output}")
    print("")
    print(f"{'id':<20} {'cosine':>10}")
    print("-" * 32)
    for item in results:
        print(f"{item['id']:<20} {item['cosine_similarity']:>10.4f}")
    print("-" * 32)
    print(f"{'mean':<20} {summary['mean_cosine_similarity']:>10.4f}")


if __name__ == "__main__":
    main()
