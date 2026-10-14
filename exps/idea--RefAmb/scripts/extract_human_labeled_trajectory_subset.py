#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_MANIFEST = Path(
    "/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_Qwen2.5-vl-7B/"
    "label_studio/utility_reasoning_sample_120/utility_reasoning_sample_120_manifest.json"
)
DEFAULT_OUTPUT = DEFAULT_MANIFEST.parent / "utility_reasoning_sample_120.trajectories.jsonl"

SOURCE_TO_TRAJECTORY = {
    "mcsearch": Path(
        "/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_Qwen2.5-vl-7B/"
        "RefAmb_2026_05_01_11_26_refamb_mcsearch_stage/omnisearch_trajectories.jsonl"
    ),
    "oven": Path(
        "/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_Qwen2.5-vl-7B/"
        "RefAmb_2026_05_01_13_28_refamb_oven_stage/omnisearch_trajectories.jsonl"
    ),
    "infoseek": Path(
        "/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_Qwen2.5-vl-7B/"
        "RefAmb_2026_05_02_13_50_refamb_infoseek_stage/omnisearch_trajectories.jsonl"
    ),
}


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def parse_search_query(content: str) -> str:
    text = content or ""
    match = re.search(r"Text Retrieval:\s*(.*?)(?:</Search>|$)", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return normalize_whitespace(match.group(1))
    return normalize_whitespace(text)


def parse_retrieval_action(content: str) -> str | None:
    text = content or ""
    if "text retrieval" in text.lower():
        return "text_retrieval_result"
    if "image retrieval" in text.lower():
        return "image_retrieval_result"
    return None


def is_search_instruction(content: str) -> bool:
    text = content or ""
    return bool(re.search(r"(text|image)\s+retrieval\s*:", text, flags=re.IGNORECASE))


def build_iterations(trajectory: list[dict]) -> list[dict]:
    iterations = []
    latest_sub_question = ""
    pending_search: dict | None = None
    open_iteration: dict | None = None

    def finalize_open_iteration(reaction_index: int, reaction_action: str | None, reaction_text: str) -> None:
        nonlocal open_iteration
        if open_iteration is None:
            return
        open_iteration["iteration_index"] = len(iterations) + 1
        open_iteration["reaction_index"] = reaction_index
        open_iteration["reaction_action"] = reaction_action
        open_iteration["reaction_text"] = reaction_text
        iterations.append(open_iteration)
        open_iteration = None

    for idx, step in enumerate(trajectory):
        action = step.get("action")

        if action == "sub-question":
            latest_sub_question = normalize_whitespace(step.get("content", ""))
            latest_sub_question = re.sub(r"</[^>]+>\s*$", "", latest_sub_question).strip()
            continue

        if action == "assistant_response":
            continue

        if action == "search":
            retrieval_action = None
            query = parse_search_query(step.get("content", ""))
            retrieval_index = idx
            retrieval_content = step.get("content", "") or ""
            retrieval_mode = step.get("mode")
            reaction_index = None
            reaction_action = None
            reaction_text = ""
            for later_idx in range(idx + 1, len(trajectory)):
                later = trajectory[later_idx]
                later_action = later.get("action")
                if later_action in {
                    "text_retrieval_result",
                    "image_retrieval_result",
                    "no_retrieval_result",
                } and retrieval_action is None:
                    retrieval_action = later_action
                    retrieval_index = later_idx
                    retrieval_content = later.get("content", "") or ""
                    retrieval_mode = later.get("mode")
                    if not query:
                        query = normalize_whitespace(str(later.get("query", "") or ""))
                    continue
                if later_action in {"thought", "final_answer", "assistant_response"}:
                    reaction_index = later_idx
                    reaction_action = later_action
                    reaction_text = later.get("content", "")
                    break
                if later_action == "search":
                    break
            if retrieval_action is None:
                retrieval_action = parse_retrieval_action(step.get("content", ""))
                if retrieval_action is None:
                    continue
            evidence_source = "explicit_retrieval_result"
            if retrieval_index == idx and is_search_instruction(step.get("content", "")):
                retrieval_content = ""
                evidence_source = "reaction_text_fallback"

            sub_question = normalize_whitespace(latest_sub_question) or query
            iterations.append(
                {
                    "iteration_index": len(iterations) + 1,
                    "search_index": idx,
                    "retrieval_index": retrieval_index,
                    "retrieval_action": retrieval_action,
                    "mode": retrieval_mode,
                    "query": query,
                    "sub_question": sub_question,
                    "retrieval_content": retrieval_content,
                    "evidence_source": evidence_source,
                    "reaction_index": reaction_index,
                    "reaction_action": reaction_action,
                    "reaction_text": reaction_text,
                }
            )
            continue

        if action in {"retrieval_result", "text_retrieval_result", "image_retrieval_result", "no_retrieval_result"}:
            if pending_search is None:
                continue
            pending_search["retrieval_index"] = idx
            pending_search["retrieval_action"] = action
            pending_search["retrieval_content"] = step.get("content", "") or ""
            pending_search["mode"] = step.get("mode") or pending_search["mode"]
            if not pending_search["query"]:
                pending_search["query"] = normalize_whitespace(str(step.get("query", "") or ""))
            open_iteration = pending_search
            pending_search = None
            continue

        if action == "assistant_response" and open_iteration is not None:
            finalize_open_iteration(idx, action, step.get("content", ""))

    return iterations


def read_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_index(samples: list[dict]) -> dict[tuple[str, str], list[dict]]:
    index: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for sample in samples:
        for iteration in build_iterations(sample.get("trajectory", [])):
            key = (
                normalize_whitespace(iteration.get("query", "")),
                normalize_whitespace(iteration.get("reaction_text", "")),
            )
            if key[0] and key[1]:
                index[key].append(sample)
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract original trajectory samples corresponding to human-labeled utility items."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    sampled_items = manifest.get("sampled_items", [])
    if not isinstance(sampled_items, list) or not sampled_items:
        raise ValueError(f"No sampled_items found in {args.manifest}")

    source_indexes = {
        source: build_index(read_jsonl(path))
        for source, path in SOURCE_TO_TRAJECTORY.items()
    }

    output_rows: list[dict] = []
    seen_ids: set[str] = set()
    source_match_counts: Counter[str] = Counter()

    for item in sampled_items:
        source = item["source"]
        query = normalize_whitespace(item["query"])
        reasoning = normalize_whitespace(item["reasoning"])
        candidates = source_indexes.get(source, {}).get((query, reasoning), [])
        if len(candidates) != 1:
            raise ValueError(
                f"Expected exactly one trajectory match for source={source!r}, "
                f"query={query!r}, found {len(candidates)}"
            )
        sample = candidates[0]
        sample_id = str(sample.get("id"))
        if sample_id in seen_ids:
            continue
        seen_ids.add(sample_id)
        source_match_counts[source] += 1
        enriched = dict(sample)
        enriched["_human_label_source"] = source
        enriched["_human_label_query"] = item["query"]
        enriched["_human_label_reasoning"] = item["reasoning"]
        enriched["_human_label_raw_task_id"] = item["task_id"]
        enriched["_human_label"] = item.get("human_label")
        output_rows.append(enriched)

    write_jsonl(args.output, output_rows)
    print(f"Wrote {len(output_rows)} trajectories to {args.output}")
    print(f"Per-source counts: {dict(source_match_counts)}")


if __name__ == "__main__":
    main()
