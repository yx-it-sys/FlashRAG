#!/usr/bin/env python3
"""
Merge naive_vote_logs.jsonl by data_id and judge ranking performance.

This script:
1. Merges document retrieval logs by grouping all documents for each data_id
2. Judges ranking performance by checking if ground truth entities are retrieved
   and whether they're ranked first
"""

import json
from collections import defaultdict
from typing import Dict, Generator


def read_jsonl(file_path: str) -> Generator[dict, None, None]:
    """
    Generator to read JSONL files line-by-line.

    Adapted from flashrag/retriever/utils.py

    Args:
        file_path: Path to the JSONL file

    Yields:
        dict: Parsed JSON object from each line
    """
    with open(file_path, "r") as f:
        while True:
            new_line = f.readline()
            if not new_line:
                return
            new_item = json.loads(new_line)
            yield new_item


def merge_vote_logs(input_path: str, output_path: str) -> None:
    """
    Merge document logs by grouping all documents for each data_id.

    Args:
        input_path: Path to naive_vote_logs.jsonl
        output_path: Path to write merged_naive_vote_logs.jsonl
    """
    # Aggregate entries by data_id
    merged_data: Dict[str, dict] = defaultdict(dict)

    for entry in read_jsonl(input_path):
        data_id = entry["data_id"]

        # Initialize data_id entry if not exists
        if not merged_data[data_id]:
            merged_data[data_id] = {
                "data_id": data_id,
                "image_id": entry.get("image_id"),
                "text_query": entry.get("text_query", ""),
                # Some pipelines (e.g. my_reranking_v1) do not emit clue.
                "clue": entry.get("clue", []),
                "docs": []
            }

        # Support both legacy dict docs and plain-string docs.
        raw_doc = entry.get("doc", "")
        if isinstance(raw_doc, dict):
            doc_text = str(raw_doc.get("text", ""))
            doc_id = raw_doc.get("id") or extract_entity_text(doc_text)
        else:
            doc_text = str(raw_doc)
            # For string docs, derive a stable id from the entity prefix.
            doc_id = extract_entity_text(doc_text)

        doc_info = {
            "id": doc_id,
            "text": doc_text,
            "vote_count": entry.get("vote_count", 0.0),
            "rank": entry.get("rank", 0)
        }
        merged_data[data_id]["docs"].append(doc_info)

    # Write merged output
    with open(output_path, "w") as f:
        for data_id in sorted(merged_data.keys()):
            f.write(json.dumps(merged_data[data_id], ensure_ascii=False) + "\n")

    print(f"Merged {len(merged_data)} unique data_ids to {output_path}")


def load_ground_truth(ground_truth_path: str) -> Dict[str, str]:
    """
    Load ground truth data into a lookup dictionary.

    Args:
        ground_truth_path: Path to infoseek_val_withkb.jsonl

    Returns:
        dict: Mapping from data_id to entity_text
    """
    ground_truth = {}

    for entry in read_jsonl(ground_truth_path):
        data_id = entry["data_id"]
        entity_text = entry["entity_text"]
        ground_truth[data_id] = entity_text

    return ground_truth


def extract_entity_text(doc_text: str) -> str:
    """
    Extract entity text from document text.
    The entity text is the part before the first ' - ' in the text.

    Args:
        doc_text: Document text content

    Returns:
        str: Entity text extracted from the beginning of the text
    """
    if not doc_text:
        return ""

    # Split on the first occurrence of ' - '
    parts = doc_text.split(' - ', 1)
    entity_text = parts[0].strip()
    return entity_text


def judge_ranking(merged_path: str, ground_truth_path: str, output_path: str) -> None:
    """
    Judge ranking performance by checking if ground truth entities are retrieved.

    Args:
        merged_path: Path to merged_naive_vote_logs.jsonl
        ground_truth_path: Path to infoseek_val_withkb.jsonl
        output_path: Path to write judgment_results.jsonl
    """
    # Load ground truth (maps data_id -> entity_text)
    ground_truth = load_ground_truth(ground_truth_path)

    # Statistics
    total_count = 0
    has_match_count = 0
    is_first_count = 0

    # Process merged data and write judgments
    with open(output_path, "w") as f_out:
        for entry in read_jsonl(merged_path):
            data_id = entry["data_id"]
            docs = entry.get("docs", [])

            # Initialize judgment
            has_match = 0
            is_first = False

            # Get expected entity_text from ground truth
            expected_entity_text = ground_truth.get(data_id)

            if expected_entity_text:
                # Check if any doc text contains the expected entity_text
                for doc in docs:
                    doc_entity_text = extract_entity_text(doc["text"])
                    if expected_entity_text in doc_entity_text:
                        # print(f"Match found for data_id {data_id}:\n expected:\n'{expected_entity_text}'\nin doc\n'{doc_entity_text}'\n")
                        has_match = 1
                        # Check if it's ranked first
                        is_first = (doc["rank"] == 1)
                        break

            # Write judgment result
            result = {
                "data_id": data_id,
                "has_match": has_match,
                "is_first": is_first
            }
            f_out.write(json.dumps(result) + "\n")

            # Update statistics
            total_count += 1
            if has_match:
                has_match_count += 1
            if is_first:
                is_first_count += 1

    # Print statistics
    print(f"\n=== Judgment Statistics ===")
    print(f"Total data_ids processed: {total_count}")
    print(f"Has match: {has_match_count} ({has_match_count / total_count * 100:.2f}%)")
    print(f"Ranked first: {is_first_count} ({is_first_count / total_count * 100:.2f}%)")
    print(f"\nResults written to {output_path}")


def main():
    """Main function to orchestrate merging and judgment tasks."""

    # Define paths
    naive_vote_logs_path = "/home/you/FlashRAG/exps/idea9/data/result/infoseek_v9/my_ranking_v1/naive_vote_logs.jsonl"
    merged_output_path = "/home/you/FlashRAG/exps/idea9/data/result/infoseek_v9/my_ranking_v1/merged_naive_vote_logs.jsonl"
    ground_truth_path = "/home/you/FlashRAG/exps/idea9/data/datasets/infoseek_val/original_corpus/infoseek_val_withkb.jsonl"
    judgment_output_path = "/home/you/FlashRAG/exps/idea9/data/result/infoseek_v9/my_ranking_v1/judgment_results.jsonl"

    # Task 1: Merge naive_vote_logs.jsonl by data_id
    print("=" * 50)
    print("Task 1: Merging naive_vote_logs.jsonl by data_id")
    print("=" * 50)
    merge_vote_logs(naive_vote_logs_path, merged_output_path)

    # Task 2: Judge ranking performance
    print("\n" + "=" * 50)
    print("Task 2: Judging ranking performance")
    print("=" * 50)
    judge_ranking(merged_output_path, ground_truth_path, judgment_output_path)

    print("\n" + "=" * 50)
    print("All tasks completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
