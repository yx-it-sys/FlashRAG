#!/usr/bin/env python3
import json
from pathlib import Path


ORIGINAL_PATH = Path(
    "/home/you/FlashRAG/exps/idea10/data/result/crag_mm_2026_03_31_14_06_experiment/label/trajectory_annotation.llm_labeled.rewrite.jsonl"
)
REVIEW_PATH = Path(
    "/home/you/FlashRAG/exps/idea10/data/result/crag_mm_2026_03_31_14_06_experiment/label/trajectory_annotation.llm_labeled.rewrite.review_minimal.jsonl"
)
OUTPUT_PATH = Path(
    "/home/you/FlashRAG/exps/idea10/data/result/crag_mm_2026_03_31_14_06_experiment/label/trajectory_annotation.llm_labeled.rewrite.merged.jsonl"
)


def load_review_map() -> dict[tuple[str, int], str]:
    rewrite_map: dict[tuple[str, int], str] = {}
    kept_annotation_ids: set[str] = set()
    with REVIEW_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            annotation_id = item["annotation_id"]
            kept_annotation_ids.add(annotation_id)
            for step in item.get("query_steps", []):
                key = (annotation_id, int(step["step_index"]))
                rewrite_map[key] = step.get("rewrite_query", "")
    return rewrite_map, kept_annotation_ids


def main() -> None:
    rewrite_map, kept_annotation_ids = load_review_map()
    updated = 0
    kept = 0
    removed = 0

    with ORIGINAL_PATH.open("r", encoding="utf-8") as src, OUTPUT_PATH.open("w", encoding="utf-8") as out:
        for line in src:
            item = json.loads(line)
            annotation_id = item.get("annotation_id")
            if annotation_id not in kept_annotation_ids:
                removed += 1
                continue
            for step in item.get("query_steps", []):
                key = (annotation_id, int(step.get("step_index", -1)))
                if key in rewrite_map:
                    step["rewrite_query"] = rewrite_map[key]
                    updated += 1
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Saved {OUTPUT_PATH}")
    print(f"Updated {updated} rewrite queries")
    print(f"Kept {kept} items")
    print(f"Removed {removed} items")


if __name__ == "__main__":
    main()
