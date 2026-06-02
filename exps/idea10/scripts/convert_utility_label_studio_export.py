#!/usr/bin/env python3
import json
from pathlib import Path


INPUT_PATH = Path(
    "/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_Qwen2.5-vl-7B/label_studio/utility_reasoning_sample_120/project-20-at-2026-05-06-07-28-bd179bb6.json"
)
OUTPUT_PATH = INPUT_PATH.with_suffix(".readable.jsonl")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, items: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def extract_first_annotation(task: dict) -> dict | None:
    annotations = task.get("annotations")
    if not isinstance(annotations, list) or not annotations:
        return None
    annotation = annotations[0]
    return annotation if isinstance(annotation, dict) else None


def extract_choice(annotation: dict | None) -> str | None:
    if not annotation:
        return None
    for result in annotation.get("result", []):
        if not isinstance(result, dict):
            continue
        value = result.get("value")
        if not isinstance(value, dict):
            continue
        choices = value.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, str):
                return choice
    return None


def extract_textarea(annotation: dict | None) -> str:
    if not annotation:
        return ""
    for result in annotation.get("result", []):
        if not isinstance(result, dict):
            continue
        value = result.get("value")
        if not isinstance(value, dict):
            continue
        texts = value.get("text")
        if isinstance(texts, list) and texts:
            text = texts[0]
            if isinstance(text, str):
                return text
    return ""


def extract_cache_input(cache_key: str) -> dict:
    try:
        data = json.loads(cache_key)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def normalize_human_label(choice: str | None) -> str | None:
    mapping = {
        "No Contribution": "no_contribution",
        "Partial Contribution": "partial_contribution",
        "Full Contribution": "full_contribution",
        "no_contribution": "no_contribution",
        "partial_contribution": "partial_contribution",
        "full_contribution": "full_contribution",
    }
    return mapping.get(choice)


def convert_task(task: dict) -> dict:
    data = task.get("data", {})
    annotation = extract_first_annotation(task)
    cache_input = extract_cache_input(data.get("cache_key", ""))
    human_choice = extract_choice(annotation)
    return {
        "task_id": data.get("task_id"),
        "source": data.get("source"),
        "query": data.get("query") or cache_input.get("query"),
        "reasoning": cache_input.get("reasoning"),
        "facts": cache_input.get("facts", []),
        "llm_label": data.get("llm_label"),
        "llm_reason": data.get("llm_reason", ""),
        "human_label": normalize_human_label(human_choice),
        "human_label_raw": human_choice,
        "human_comment": extract_textarea(annotation),
        "cache_path": data.get("cache_path"),
        "cache_key": data.get("cache_key"),
    }


def main() -> None:
    tasks = read_json(INPUT_PATH)
    if not isinstance(tasks, list):
        raise ValueError(f"Expected list in {INPUT_PATH}")
    readable_items = [convert_task(task) for task in tasks]
    write_jsonl(OUTPUT_PATH, readable_items)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
