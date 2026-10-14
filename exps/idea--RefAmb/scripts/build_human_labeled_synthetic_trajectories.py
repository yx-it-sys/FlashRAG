#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DEFAULT_INPUT = Path(
    "/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_Qwen2.5-vl-7B/"
    "label_studio/utility_reasoning_sample_120/project-20-at-2026-05-06-07-28-bd179bb6.readable.jsonl"
)
DEFAULT_OUTPUT = DEFAULT_INPUT.parent / "project-20-at-2026-05-06-07-28-bd179bb6.synthetic_trajectories.jsonl"


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_retrieval_content(facts: list[str]) -> str:
    if not facts:
        return "Doc1:\nNo supporting facts were extracted."
    docs = []
    for idx, fact in enumerate(facts, start=1):
        docs.append(f"Doc{idx}:\n{fact}")
    return "\n\n".join(docs)


def build_sample(row: dict, index: int) -> dict:
    query = str(row.get("query") or "").strip()
    reasoning = str(row.get("reasoning") or "").strip()
    facts = row.get("facts") or []
    if not isinstance(facts, list):
        facts = []
    retrieval_content = build_retrieval_content([str(f).strip() for f in facts if str(f).strip()])
    sample_id = f"human_labeled_{index:04d}"

    return {
        "id": sample_id,
        "question": query,
        "final_answer": "",
        "status": "synthetic_from_human_label_export",
        "trajectory": [
            {
                "action": "sub-question",
                "content": f"{query}\n</Sub-Question>",
            },
            {
                "action": "search",
                "content": f"Text Retrieval: {query}</Search>",
            },
            {
                "action": "text_retrieval_result",
                "mode": "text_retrieval",
                "query": query,
                "content": retrieval_content,
            },
            {
                "action": "thought",
                "content": reasoning,
            },
        ],
        "_source": row.get("source"),
        "_raw_task_id": row.get("task_id"),
        "_llm_label": row.get("llm_label"),
        "_human_label": row.get("human_label"),
        "_human_label_raw": row.get("human_label_raw"),
        "_human_comment": row.get("human_comment"),
        "_cache_path": row.get("cache_path"),
        "_normalized_query": normalize_whitespace(query),
        "_normalized_reasoning": normalize_whitespace(reasoning),
        "_facts_count": len(facts),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert human-labeled utility export into a synthetic trajectory jsonl for evaluate_trajectory_quality.py."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    samples = [build_sample(row, idx) for idx, row in enumerate(rows, start=1)]
    write_jsonl(args.output, samples)
    print(f"Wrote {len(samples)} synthetic trajectories to {args.output}")


if __name__ == "__main__":
    main()
