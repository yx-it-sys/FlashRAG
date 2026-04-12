#!/usr/bin/env python3
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from string import Formatter
import tomllib
from openai import OpenAI

for k in [
    "http_proxy", "https_proxy",
    "HTTP_PROXY", "HTTPS_PROXY",
    "all_proxy", "ALL_PROXY"
]:
    os.environ.pop(k, None)

INPUT_PATH = Path(
    "/home/you/FlashRAG/exps/idea10/data/result/crag_mm_2026_03_31_14_06_experiment/label/trajectory_annotation.jsonl"
)
OUTPUT_PATH = Path(
    "/home/you/FlashRAG/exps/idea10/data/result/crag_mm_2026_03_31_14_06_experiment/label/deepseek/trajectory_annotation.llm_labeled.jsonl"
)
PROMPT_PATH = Path("/home/you/FlashRAG/exps/idea10/prompts/label_entitiy_ambiguouty.toml")
MODEL = "Pro/deepseek-ai/DeepSeek-R1"
BASE_URL = "https://api.siliconflow.cn/v1"
WORKERS = 2
TIMEOUT = 120.0
RETRIES = 2
VALID_LABELS = {
    "Object Identification",
    "Description",
    "Indirect Entity Ambiguity",
    "No Object Involved",
}
LABEL_ALIASES = {
    "Normal ambiguouty": "Indirect Entity Ambiguity",
}


def render_progress(completed: int, total: int, width: int = 32) -> str:
    if total <= 0:
        return "[--------------------------------] 0/0"
    filled = int(width * completed / total)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {completed}/{total}"


def read_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(path: Path, items: list[dict]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def load_system_prompt(path: Path) -> str:
    with path.open("rb") as f:
        return tomllib.load(f)["system_prompt"]


def collect_keys(template: str) -> set[str]:
    keys = set()
    for _, field_name, _, _ in Formatter().parse(template):
        if field_name:
            keys.add(field_name)
    return keys


def render_prompt(template: str, sample: dict, query_step: dict) -> str:
    values = {
        "question": sample.get("question", ""),
        "sub_question": query_step.get("sub_question", ""),
        "text_query": query_step.get("text_query", ""),
    }
    required = collect_keys(template)
    missing = [key for key in required if key not in values]
    if missing:
        raise KeyError(f"Missing prompt variables: {missing}")
    return template.format(**values)


def is_labeled(query_step: dict) -> bool:
    llm_label = query_step.get("llm_label")
    if not isinstance(llm_label, dict):
        return False
    entity = llm_label.get("entity_ambiguous")
    level = LABEL_ALIASES.get(llm_label.get("ambiguity_level"), llm_label.get("ambiguity_level"))
    if entity == "No":
        return True
    if entity == "Yes" and level in VALID_LABELS:
        return True
    return False


def extract_json_block(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in model output: {text!r}")
    return json.loads(text[start : end + 1])


def normalize_result(data: dict) -> dict:
    entity = data.get("entity_ambiguous")
    level = LABEL_ALIASES.get(data.get("ambiguity_level"), data.get("ambiguity_level"))
    remark = data.get("remark", "")

    if entity not in {"Yes", "No"}:
        raise ValueError(f"Invalid entity_ambiguous: {entity!r}")
    if entity == "No":
        level = None
    elif level not in VALID_LABELS:
        raise ValueError(f"Invalid ambiguity_level: {level!r}")
    if remark is None:
        remark = ""
    if not isinstance(remark, str):
        remark = str(remark)

    return {
        "entity_ambiguous": entity,
        "ambiguity_level": level,
        "remark": remark,
    }


def normalize_query_key(query: str) -> str | None:
    if not isinstance(query, str):
        return None
    query = query.strip()
    return query or None


def build_existing_label_cache(items: list[dict]) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    for sample in items:
        for query_step in sample.get("query_steps", []):
            if not is_labeled(query_step):
                continue
            query_key = normalize_query_key(query_step.get("text_query", ""))
            if query_key is None:
                continue
            cache[query_key] = dict(query_step["llm_label"])
    return cache


def annotate_query(client: OpenAI, model: str, system_prompt: str, sample_id: str, query_index: int) -> dict:
    last_error = None
    for attempt in range(RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}],
                stream=False,
                timeout=TIMEOUT,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
            print(f"\n[LLM RAW] annotation_id={sample_id} step_index={query_index}")
            print(content, flush=True)
            return normalize_result(extract_json_block(content))
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < RETRIES:
                time.sleep(min(2 ** attempt, 4))
    raise RuntimeError(f"Failed sample_id={sample_id} query_index={query_index}: {last_error}") from last_error


def main() -> None:
    api_key = os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set SILICONFLOW_API_KEY or OPENAI_API_KEY before running.")

    source_items = read_jsonl(INPUT_PATH)
    if OUTPUT_PATH.exists():
        items = read_jsonl(OUTPUT_PATH)
        if len(items) != len(source_items):
            raise ValueError(f"Existing output length mismatch: {len(items)} vs {len(source_items)}")
        print(f"Resuming from existing output: {OUTPUT_PATH}")
    else:
        items = source_items
        print(f"Starting new labeling run from: {INPUT_PATH}")

    system_template = load_system_prompt(PROMPT_PATH)
    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    lock = threading.Lock()
    futures = {}
    future_targets = {}
    pending = 0
    completed = 0
    cache_hits = 0
    dedup_hits = 0
    label_cache = build_existing_label_cache(items)

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        for sample_idx, sample in enumerate(items):
            for step_idx, query_step in enumerate(sample.get("query_steps", [])):
                if is_labeled(query_step):
                    continue
                query_key = normalize_query_key(query_step.get("text_query", ""))
                if query_key is not None and query_key in label_cache:
                    query_step["llm_label"] = dict(label_cache[query_key])
                    cache_hits += 1
                    continue
                prompt = render_prompt(system_template, sample, query_step)
                if query_key is not None and query_key in futures:
                    future_targets[futures[query_key]].append((sample_idx, step_idx))
                    dedup_hits += 1
                    continue
                future = executor.submit(
                    annotate_query,
                    client,
                    MODEL,
                    prompt,
                    sample.get("annotation_id", ""),
                    query_step.get("step_index", step_idx + 1),
                )
                if query_key is not None:
                    futures[query_key] = future
                future_targets[future] = [(sample_idx, step_idx)]
                pending += 1

        print(f"Pending queries: {pending}")
        print(
            f"Skipped by existing label cache: {cache_hits}; "
            f"deduplicated in current run: {dedup_hits}"
        )
        print(render_progress(0, pending), flush=True)

        if cache_hits > 0:
            write_jsonl(OUTPUT_PATH, items)

        for future in as_completed(future_targets):
            result = future.result()
            with lock:
                for sample_idx, step_idx in future_targets[future]:
                    query_step = items[sample_idx]["query_steps"][step_idx]
                    query_step["llm_label"] = dict(result)
                    query_key = normalize_query_key(query_step.get("text_query", ""))
                    if query_key is not None:
                        label_cache[query_key] = dict(result)
                completed += 1
                write_jsonl(OUTPUT_PATH, items)
                sys.stdout.write("\r" + render_progress(completed, pending))
                sys.stdout.flush()

    if pending == 0 and not OUTPUT_PATH.exists():
        write_jsonl(OUTPUT_PATH, items)
    elif pending > 0:
        print()
    print(f"Done. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
