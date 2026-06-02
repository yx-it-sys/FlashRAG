#!/usr/bin/env python3
import argparse
import json
import os
import io
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path
from string import Formatter
import tomllib

from openai import OpenAI

for k in [
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
]:
    os.environ.pop(k, None)

DEFAULT_DATASET_PATH = Path(
    "/home/you/FlashRAG/exps/idea10/data/datasets/RefAmb/task_balanced_analysis_subset.jsonl"
)
DEFAULT_PROMPT_PATH = Path(
    "/home/you/FlashRAG/exps/idea10/prompts/rewrite_entity_ambiguous_query.toml"
)
ROOT_RESULT_DIR = Path(
    "/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_Qwen3-vl-8b"
)
SOURCE_DIRS = [
    ROOT_RESULT_DIR / "RefAmb_2026_05_23_11_40_refamb_oven_qwen3_vl_8b_stage",
    ROOT_RESULT_DIR / "RefAmb_2026_05_23_12_10_refamb_infoseek_qwen3_vl_8b_stage",
    ROOT_RESULT_DIR / "RefAmb_2026_05_23_12_53_refamb_mcsearch_qwen3_vl_8b_stage",
    ROOT_RESULT_DIR / "RefAmb_2026_05_23_14_07_refamb_crag_qwen3_vl_8b_stage",
]
MULTI_INPUT_REL_PATH = Path(
    "label/deepseek/omnisearch_trajectories.entity_ambiguity_labeled.jsonl"
)
MULTI_OUTPUT_REL_PATH = Path(
    "label/deepseek/omnisearch_trajectories.entity_ambiguity_labeled.rewrite.jsonl"
)
MODEL = "Pro/deepseek-ai/DeepSeek-V3"
BASE_URL = "https://api.siliconflow.cn/v1"
WORKERS = 2
TIMEOUT = 120.0
RETRIES = 2
PRINT_LOCK = threading.Lock()


class TeeStream(io.TextIOBase):
    def __init__(self, *streams: io.TextIOBase) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def default_log_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".rewrite.log")


def setup_logging(
    log_path: Path,
) -> tuple[io.TextIOBase, io.TextIOBase, io.TextIOBase]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_file)
    sys.stderr = TeeStream(original_stderr, log_file)
    print(f"[Logging] Appending stdout/stderr to {log_path}", flush=True)
    return log_file, original_stdout, original_stderr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use an LLM to rewrite entity-ambiguous queries and attach "
            '`rewrite_query` to each ambiguous query step.'
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=WORKERS)
    return parser.parse_args()


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
    path.parent.mkdir(parents=True, exist_ok=True)
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


def answers_to_text(sample: dict) -> str:
    answers = sample.get("Answers")
    if isinstance(answers, list):
        return "\n".join(str(x) for x in answers)
    if isinstance(answers, str):
        return answers
    raw_answers = sample.get("answer")
    if isinstance(raw_answers, list):
        return "\n".join(str(x) for x in raw_answers)
    if raw_answers is None:
        return ""
    return str(raw_answers)


def oracle_entities_to_text(sample: dict) -> str:
    oracle_entity = sample.get("oracle_entity")
    if isinstance(oracle_entity, list):
        return "\n".join(str(x) for x in oracle_entity)
    if oracle_entity is None:
        return ""
    return str(oracle_entity)


def render_prompt(template: str, sample: dict, query_step: dict, text_query: str) -> str:
    llm_label = query_step.get("llm_label") or {}
    values = {
        "question": sample.get("question", ""),
        "text_query": text_query,
        "ambiguity_level": llm_label.get("ambiguity_level", ""),
        "answers": answers_to_text(sample),
        "oracle_entity": oracle_entities_to_text(sample),
    }
    required = collect_keys(template)
    missing = [key for key in required if key not in values]
    if missing:
        raise KeyError(f"Missing prompt variables: {missing}")
    return template.format(**values)


def needs_rewrite(query_step: dict) -> bool:
    llm_label = query_step.get("llm_label")
    if not isinstance(llm_label, dict):
        return False
    if llm_label.get("entity_ambiguous") != "Yes":
        return False
    rewrite_query = query_step.get("rewrite_query")
    return not isinstance(rewrite_query, str) or not rewrite_query.strip()


def normalize_query_key(query: str) -> str | None:
    if not isinstance(query, str):
        return None
    query = query.strip()
    return query or None


def extract_text_retrieval_query_from_search(content: str) -> str:
    if not isinstance(content, str):
        return ""
    match = re.search(r'Text Retrieval[:\s"]*(.*?)(?=(?:<|$))', content, re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip().strip('"')


def iter_rewrite_targets(sample: dict):
    query_steps = sample.get("query_steps")
    if isinstance(query_steps, list):
        for step_idx, query_step in enumerate(query_steps):
            if not isinstance(query_step, dict):
                continue
            text_query = normalize_query_key(query_step.get("text_query", ""))
            if not text_query or not needs_rewrite(query_step):
                continue
            yield {
                "container": "query_steps",
                "index": step_idx,
                "target": query_step,
                "step_id": query_step.get("step_index", step_idx + 1),
                "text_query": text_query,
            }
        return

    trajectory = sample.get("trajectory")
    if not isinstance(trajectory, list):
        return

    for step_idx, step in enumerate(trajectory):
        if not isinstance(step, dict):
            continue
        if step.get("action") != "search":
            continue
        text_query = normalize_query_key(
            extract_text_retrieval_query_from_search(step.get("content", ""))
        )
        if not text_query or not needs_rewrite(step):
            continue
        yield {
            "container": "trajectory",
            "index": step_idx,
            "target": step,
            "step_id": step_idx + 1,
            "text_query": text_query,
        }


def first_rewrite_target(sample: dict) -> dict | None:
    for target_info in iter_rewrite_targets(sample):
        return target_info
    return None


def sample_id_of(sample: dict) -> str:
    sample_id = sample.get("id")
    if isinstance(sample_id, str) and sample_id.strip():
        return sample_id
    sample_id = sample.get("annotation_id")
    if isinstance(sample_id, str) and sample_id.strip():
        return sample_id
    return ""


def build_dataset_index(items: list[dict]) -> dict[str, dict]:
    dataset_by_id: dict[str, dict] = {}
    for item in items:
        item_id = item.get("id")
        if isinstance(item_id, str) and item_id.strip():
            dataset_by_id[item_id] = item
    return dataset_by_id


def merge_dataset_fields(source_items: list[dict], dataset_by_id: dict[str, dict]) -> list[dict]:
    merged_items: list[dict] = []
    matched = 0
    for source_item in source_items:
        merged_item = dict(source_item)
        source_id = sample_id_of(source_item)
        dataset_item = dataset_by_id.get(source_id)
        if dataset_item is not None:
            matched += 1
            for key, value in dataset_item.items():
                if key in {"trajectory", "query_steps", "llm_label", "rewrite_query"}:
                    continue
                if key not in merged_item:
                    merged_item[key] = value
        merged_items.append(merged_item)
    print(
        f"Merged original dataset fields for {matched}/{len(source_items)} labeled items.",
        flush=True,
    )
    return merged_items


def get_step_container(sample: dict, container: str) -> list[dict]:
    if container == "query_steps":
        steps = sample.get("query_steps")
    elif container == "trajectory":
        steps = sample.get("trajectory")
    else:
        raise ValueError(f"Unsupported container: {container}")
    if not isinstance(steps, list):
        raise ValueError(f"Sample is missing expected list container: {container}")
    return steps


def extract_json_block(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in model output: {text!r}")
    return json.loads(text[start : end + 1])


def normalize_result(data: dict, fallback_query: str) -> str:
    rewritten_query = data.get("rewritten_query", fallback_query)
    if rewritten_query is None:
        rewritten_query = fallback_query
    if not isinstance(rewritten_query, str):
        rewritten_query = str(rewritten_query)
    rewritten_query = rewritten_query.strip()
    if not rewritten_query:
        rewritten_query = fallback_query
    return rewritten_query


def rewrite_query_with_llm(
    client: OpenAI,
    model: str,
    prompt: str,
    sample_id: str,
    query_index: int,
    fallback_query: str,
) -> str:
    last_error = None
    for attempt in range(RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a query rewriting assistant.",
                    },
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                timeout=TIMEOUT,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
            with PRINT_LOCK:
                print(
                    f"\n[REWRITE BLOCK] sample_id={sample_id} step_index={query_index}",
                    flush=True,
                )
                print("[BEFORE]", flush=True)
                print(fallback_query, flush=True)
                print("[LLM RAW]", flush=True)
                print(content, flush=True)
            return normalize_result(extract_json_block(content), fallback_query)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < RETRIES:
                time.sleep(min(2**attempt, 4))
    raise RuntimeError(
        f"Failed sample_id={sample_id} query_index={query_index}: {last_error}"
    ) from last_error


def build_existing_rewrite_cache(items: list[dict]) -> dict[str, str]:
    cache: dict[str, str] = {}
    for sample in items:
        target_info = first_rewrite_target(sample)
        if target_info is None:
            continue
        query_step = target_info["target"]
        text_query = target_info["text_query"]
        rewrite_query = query_step.get("rewrite_query")
        if not isinstance(rewrite_query, str) or not rewrite_query.strip():
            continue
        cache[text_query] = rewrite_query.strip()
    return cache


def build_multi_source_paths() -> list[tuple[Path, Path, Path]]:
    runs: list[tuple[Path, Path, Path]] = []
    for source_dir in SOURCE_DIRS:
        input_path = source_dir / MULTI_INPUT_REL_PATH
        output_path = source_dir / MULTI_OUTPUT_REL_PATH
        log_path = default_log_path(output_path)
        runs.append((input_path, output_path, log_path))
    return runs


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("SILICONFLOW_API_KEY")
    if not api_key:
        raise ValueError("Set SILICONFLOW_API_KEY before running.")

    runs = build_multi_source_paths()
    for run_idx, (input_path, output_path, log_path) in enumerate(runs, start=1):
        print(f"\n===== [{run_idx}/{len(runs)}] Entity Rewrite =====", flush=True)
        print(f"[Run] input={input_path}", flush=True)
        print(f"[Run] output={output_path}", flush=True)
        print(f"[Run] log={log_path}", flush=True)

        if not input_path.exists() and not output_path.exists():
            raise FileNotFoundError(f"Neither input nor output exists for run: {input_path}")

        log_file, original_stdout, original_stderr = setup_logging(log_path)
        try:
            source_items = read_jsonl(input_path)
            dataset_items = read_jsonl(args.dataset)
            dataset_by_id = build_dataset_index(dataset_items)
            if output_path.exists():
                items = read_jsonl(output_path)
                if len(items) != len(source_items):
                    raise ValueError(
                        f"Existing output length mismatch: {len(items)} vs {len(source_items)}"
                    )
                items = merge_dataset_fields(items, dataset_by_id)
                print(f"Resuming from existing output: {output_path}")
            else:
                items = merge_dataset_fields(source_items, dataset_by_id)
                print(f"Starting new rewrite run from: {input_path}")

            system_template = load_system_prompt(args.prompt)
            client = OpenAI(api_key=api_key, base_url=BASE_URL)

            lock = threading.Lock()
            futures: dict[str, object] = {}
            future_targets: dict = defaultdict(list)
            pending = 0
            completed = 0
            failed = 0
            cache_hits = 0
            dedup_hits = 0
            rewrite_cache = build_existing_rewrite_cache(items)

            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                for sample_idx, sample in enumerate(items):
                    target_info = first_rewrite_target(sample)
                    if target_info is None:
                        continue
                    step_idx = target_info["index"]
                    query_step = target_info["target"]
                    text_query = target_info["text_query"]
                    if text_query in rewrite_cache:
                        query_step["rewrite_query"] = rewrite_cache[text_query]
                        cache_hits += 1
                        continue
                    prompt = render_prompt(system_template, sample, query_step, text_query)
                    dedup_key = text_query
                    if dedup_key in futures:
                        future_targets[futures[dedup_key]].append(
                            (sample_idx, target_info["container"], step_idx)
                        )
                        dedup_hits += 1
                        continue
                    future = executor.submit(
                        rewrite_query_with_llm,
                        client,
                        MODEL,
                        prompt,
                        sample_id_of(sample),
                        target_info["step_id"],
                        text_query,
                    )
                    futures[dedup_key] = future
                    future_targets[future].append(
                        (sample_idx, target_info["container"], step_idx)
                    )
                    pending += 1

                print(f"Pending queries: {pending}")
                print(
                    f"Skipped by existing rewrite cache: {cache_hits}; "
                    f"deduplicated in current run: {dedup_hits}"
                )
                print(render_progress(0, pending), flush=True)

                write_jsonl(output_path, items)

                for future in as_completed(future_targets):
                    targets = future_targets[future]
                    with lock:
                        try:
                            rewritten_query = future.result()
                        except Exception as exc:  # noqa: BLE001
                            failed += 1
                            error_message = str(exc)
                            print(
                                f"\n[Rewrite Failed] {error_message}",
                                file=sys.stderr,
                                flush=True,
                            )
                            for sample_idx, container, step_idx in targets:
                                query_step = get_step_container(
                                    items[sample_idx], container
                                )[step_idx]
                                query_step["rewrite_error"] = error_message
                            completed += 1
                            write_jsonl(output_path, items)
                            sys.stdout.write("\r" + render_progress(completed, pending))
                            sys.stdout.flush()
                            continue

                        for sample_idx, container, step_idx in targets:
                            query_step = get_step_container(items[sample_idx], container)[
                                step_idx
                            ]
                            query_step["rewrite_query"] = rewritten_query
                            query_step.pop("rewrite_error", None)
                            text_query = normalize_query_key(
                                query_step.get("text_query", "")
                                if "text_query" in query_step
                                else extract_text_retrieval_query_from_search(
                                    query_step.get("content", "")
                                )
                            )
                            if text_query:
                                rewrite_cache[text_query] = rewritten_query
                        completed += 1
                        write_jsonl(output_path, items)
                        sys.stdout.write("\r" + render_progress(completed, pending))
                        sys.stdout.flush()

            if pending == 0:
                write_jsonl(output_path, items)
            else:
                print()
            print(f"Done. Output: {output_path}; failed rewrites: {failed}")
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_file.close()


if __name__ == "__main__":
    main()
