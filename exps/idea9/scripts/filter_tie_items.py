#!/usr/bin/env python3
import argparse
import json
from typing import Dict, Generator, List, Set


def read_jsonl(file_path: str) -> Generator[dict, None, None]:
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def extract_title(doc_text: str) -> str:
    if not isinstance(doc_text, str):
        return ""
    parts = doc_text.split(" - ", 1)
    return parts[0].strip()


def load_ground_truth_mapping(gt_file_path: str) -> Dict[str, str]:
    gt_mapping: Dict[str, str] = {}
    for row in read_jsonl(gt_file_path):
        data_id = row.get("data_id")
        entity_text = row.get("entity_text")
        if data_id and entity_text:
            gt_mapping[str(data_id)] = str(entity_text).strip()
    return gt_mapping


def load_target_data_ids(judgment_file_path: str) -> Set[str]:
    """Keep only data_ids where has_match == 1 and is_first is False."""
    target_ids: Set[str] = set()
    for row in read_jsonl(judgment_file_path):
        data_id = row.get("data_id")
        has_match = row.get("has_match", 0)
        is_first = row.get("is_first", True)
        if data_id is None:
            continue
        if int(has_match) == 1 and bool(is_first) is False:
            target_ids.add(str(data_id))
    return target_ids


def normalize_docs(docs: List[dict]) -> List[dict]:
    normalized: List[dict] = []
    for doc in docs:
        rank = doc.get("rank")
        vote_count = doc.get("vote_count")
        text = doc.get("text", "")
        try:
            rank_val = int(rank)
        except (TypeError, ValueError):
            continue
        try:
            score_val = float(vote_count)
        except (TypeError, ValueError):
            score_val = 0.0
        normalized.append({
            "rank": rank_val,
            "vote_count": score_val,
            "text": str(text),
        })
    return sorted(normalized, key=lambda d: d["rank"])


def find_tie_items(
    merged_file_path: str,
    judgment_file_path: str,
    gt_file_path: str,
) -> List[dict]:
    gt_mapping = load_ground_truth_mapping(gt_file_path)
    target_ids = load_target_data_ids(judgment_file_path)

    tie_items: List[dict] = []
    eps = 1e-9

    for item in read_jsonl(merged_file_path):
        data_id = str(item.get("data_id", ""))
        if not data_id or data_id not in target_ids:
            continue

        gt_title = gt_mapping.get(data_id)
        if not gt_title:
            continue

        docs = item.get("docs", [])
        if not isinstance(docs, list):
            continue
        ranked_docs = normalize_docs(docs)
        if len(ranked_docs) < 2:
            continue

        rank1_doc = ranked_docs[0]
        rank1_score = rank1_doc["vote_count"]

        # Generalized requirement:
        # GT may appear at any rank m (>1). Keep item if any GT-ranked doc ties with rank1 score.
        gt_tie_docs = []
        for doc in ranked_docs[1:]:
            doc_title = extract_title(doc["text"])
            if doc_title != gt_title:
                continue
            if abs(doc["vote_count"] - rank1_score) <= eps:
                gt_tie_docs.append(doc)

        if gt_tie_docs:
            first_gt_tie_doc = min(gt_tie_docs, key=lambda d: d["rank"])
            tie_items.append({
                "data_id": data_id,
                "image_id": item.get("image_id"),
                "text_query": item.get("text_query"),
                "clue": item.get("clue"),
                "ground_truth_title": gt_title,
                "rank1_title": extract_title(rank1_doc["text"]),
                "rank1_score": rank1_score,
                "rank1_text": rank1_doc["text"],
                "first_gt_tie_rank": first_gt_tie_doc["rank"],
                "first_gt_tie_score": first_gt_tie_doc["vote_count"],
                "first_gt_tie_text": first_gt_tie_doc["text"],
                "gt_tie_ranks": [doc["rank"] for doc in gt_tie_docs],
                "gt_tie_count": len(gt_tie_docs),
                "docs": item.get("docs", []),
            })

    return tie_items


def write_jsonl(items: List[dict], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Filter tie items: has_match==1 and is_first==false, then keep items where "
            "GT appears at any non-top rank and ties with rank1 score."
        )
    )
    parser.add_argument(
        "--merged",
        default="/home/you/FlashRAG/exps/idea9/data/result/infoseek_v9/debug_cluster/merged_naive_vote_logs.jsonl",
        help="Path to merged_naive_vote_logs.jsonl",
    )
    parser.add_argument(
        "--judgment",
        default="/home/you/FlashRAG/exps/idea9/data/result/infoseek_v9/debug_cluster/judgment_results.jsonl",
        help="Path to judgment_results.jsonl",
    )
    parser.add_argument(
        "--ground-truth",
        default="/mnt/data/you/datasets/infoseek_val/original_corpus/infoseek_val_withkb.jsonl",
        help="Path to infoseek_val_withkb.jsonl",
    )
    parser.add_argument(
        "--output",
        default="/home/you/FlashRAG/exps/idea9/data/result/infoseek_v9/debug_cluster/tie.jsonl",
        help="Path to output tie.jsonl",
    )
    args = parser.parse_args()

    tie_items = find_tie_items(args.merged, args.judgment, args.ground_truth)
    write_jsonl(tie_items, args.output)

    print(f"Found {len(tie_items)} tie items.")
    print(f"Output written to: {args.output}")


if __name__ == "__main__":
    main()
