#!/usr/bin/env python3
import argparse
import json
import os
import io
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

DEFAULT_INPUT_PATH = Path(
    "/home/you/FlashRAG/exps/idea10/data/result/crag_mm_2026_03_31_14_06_experiment/label/trajectory_annotation.llm_labeled.jsonl"
)
DEFAULT_OUTPUT_PATH = "/home/you/FlashRAG/exps/idea10/data/result/crag_mm_2026_03_31_14_06_experiment/label/trajectory_annotation.llm_labeled.rewrite_deepseek.jsonl"
DEFAULT_PROMPT_PATH = Path(
    "/home/you/FlashRAG/exps/idea10/prompts/rewrite_entity_ambiguous_query.toml"
)
MODEL = "Pro/deepseek-ai/DeepSeek-R1"
BASE_URL = "https://api.siliconflow.cn/v1"
WORKERS = 2
TIMEOUT = 120.0
RETRIES = 2


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
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
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
        return "\n".join(answers)
    if isinstance(answers, str):
        return answers
    raw_answers = sample.get("answer")
    if isinstance(raw_answers, list):
        return "\n".join(str(x) for x in raw_answers)
    if raw_answers is None:
        return ""
    return str(raw_answers)


def render_prompt(template: str, sample: dict, query_step: dict) -> str:
    llm_label = query_step.get("llm_label") or {}
    values = {
        "question": sample.get("question", ""),
        "text_query": query_step.get("text_query", ""),
        "ambiguity_level": llm_label.get("ambiguity_level", ""),
        "answers": answers_to_text(sample),
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
            print(f"\n[LLM RAW] annotation_id={sample_id} step_index={query_index}")
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
        for query_step in sample.get("query_steps", []):
            text_query = query_step.get("text_query", "")
            rewrite_query = query_step.get("rewrite_query")
            if not isinstance(text_query, str) or not text_query:
                continue
            if not isinstance(rewrite_query, str) or not rewrite_query.strip():
                continue
            cache[text_query] = rewrite_query.strip()
    return cache


def main() -> None:
    args = parse_args()
    log_path = args.log or default_log_path(args.output)
    log_file, original_stdout, original_stderr = setup_logging(log_path)
    api_key = os.environ.get("SILICONFLOW_API_KEY")
    try:
        if not api_key:
            raise ValueError("Set SILICONFLOW_API_KEY before running.")

        source_items = read_jsonl(args.input)
        if args.output.exists():
            items = read_jsonl(args.output)
            if len(items) != len(source_items):
                raise ValueError(
                    f"Existing output length mismatch: {len(items)} vs {len(source_items)}"
                )
            print(f"Resuming from existing output: {args.output}")
        else:
            items = source_items
            print(f"Starting new rewrite run from: {args.input}")

        system_template = load_system_prompt(args.prompt)
        client = OpenAI(api_key=api_key, base_url=BASE_URL)

        lock = threading.Lock()
        futures = {}
        future_targets: dict = defaultdict(list)
        pending = 0
        completed = 0
        failed = 0
        cache_hits = 0
        dedup_hits = 0
        rewrite_cache = build_existing_rewrite_cache(items)

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for sample_idx, sample in enumerate(items):
                for step_idx, query_step in enumerate(sample.get("query_steps", [])):
                    if not needs_rewrite(query_step):
                        continue
                    text_query = query_step.get("text_query", "")
                    if isinstance(text_query, str) and text_query in rewrite_cache:
                        query_step["rewrite_query"] = rewrite_cache[text_query]
                        cache_hits += 1
                        continue
                    prompt = render_prompt(system_template, sample, query_step)
                    dedup_key = text_query if isinstance(text_query, str) else ""
                    if dedup_key in futures:
                        future_targets[futures[dedup_key]].append((sample_idx, step_idx))
                        dedup_hits += 1
                        continue
                    future = executor.submit(
                        rewrite_query_with_llm,
                        client,
                        MODEL,
                        prompt,
                        sample.get("annotation_id", ""),
                        query_step.get("step_index", step_idx + 1),
                        query_step.get("text_query", ""),
                    )
                    futures[dedup_key] = future
                    future_targets[future].append((sample_idx, step_idx))
                    pending += 1

            print(f"Pending queries: {pending}")
            print(
                f"Skipped by existing rewrite cache: {cache_hits}; "
                f"deduplicated in current run: {dedup_hits}"
            )
            print(render_progress(0, pending), flush=True)

            # Persist cache-hit assignments before waiting on remote calls so they
            # are not lost if one request later fails.
            write_jsonl(args.output, items)

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
                        for sample_idx, step_idx in targets:
                            query_step = items[sample_idx]["query_steps"][step_idx]
                            query_step["rewrite_error"] = error_message
                        completed += 1
                        write_jsonl(args.output, items)
                        sys.stdout.write("\r" + render_progress(completed, pending))
                        sys.stdout.flush()
                        continue

                    for sample_idx, step_idx in targets:
                        query_step = items[sample_idx]["query_steps"][step_idx]
                        query_step["rewrite_query"] = rewritten_query
                        query_step.pop("rewrite_error", None)
                        text_query = query_step.get("text_query", "")
                        if isinstance(text_query, str) and text_query:
                            rewrite_cache[text_query] = rewritten_query
                    completed += 1
                    write_jsonl(args.output, items)
                    sys.stdout.write("\r" + render_progress(completed, pending))
                    sys.stdout.flush()

        if pending == 0:
            write_jsonl(args.output, items)
        else:
            print()
        print(f"Done. Output: {args.output}; failed rewrites: {failed}")
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


if __name__ == "__main__":
    main()
