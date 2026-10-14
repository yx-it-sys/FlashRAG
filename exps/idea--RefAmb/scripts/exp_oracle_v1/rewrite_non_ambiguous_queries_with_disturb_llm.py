#!/usr/bin/env python3
import argparse
import io
import json
import os
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
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

DEFAULT_PROMPT_PATH = Path(
    "/home/you/FlashRAG/exps/idea10/prompts/disturb_rewrite_prompt.toml"
)
DEFAULT_SUBSET_PATH = Path(
    "/home/you/FlashRAG/exps/idea10/data/datasets/RefAmb/new/task_balanced.jsonl"
)
ROOT_RESULT_DIRS = [
    Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_GPT_5.1/RefAmb_2026_05_29_11_04_refamb_mcsearch_gpt_5_1_ca_stage"),
    Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_qwen3_vl_32b/RefAmb_2026_05_19_21_03_refamb_mcsearch_qwen3_vl_32b_stage"),
    # Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_Qwen3-vl-8b/RefAmb_2026_05_23_12_53_refamb_mcsearch_qwen3_vl_8b_stage"),
    # Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_InternVL3.5-8B/RefAmb_2026_05_09_16_39_refamb_mcsearch_intervl3_5_8b_stage"),
]
MULTI_INPUT_REL_PATH = Path("label/deepseek/omnisearch_trajectories.entity_ambiguity_labeled.jsonl")
MULTI_OUTPUT_REL_PATH = Path(
    "label/deepseek/omnisearch_trajectories.entity_ambiguity_labeled.disturb_rewrite.jsonl"
)
PROMPT_STYLE = "system_disturb_rewrite_prompt"
SOURCE_FIELD = "text_query"
TARGET_FIELD = "disturb_rewrite_query"
META_FIELD = "disturb_rewrite_meta"
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
    return output_path.with_suffix(output_path.suffix + ".disturb_rewrite.log")


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
            "Use an LLM to rewrite entity-unambiguous text retrieval queries into "
            "more ambiguous disturb queries."
        )
    )
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


def load_allowed_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    ids = set()
    for item in read_jsonl(path):
        for key in ("id", "annotation_id", "data_id"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                ids.add(value)
                break
    return ids


def load_prompt_template(path: Path, prompt_style: str) -> str:
    with path.open("rb") as f:
        data = tomllib.load(f)
    prompt_node = data.get(prompt_style)
    if isinstance(prompt_node, str) and prompt_node.strip():
        return prompt_node
    if isinstance(prompt_node, dict):
        prompt = prompt_node.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            return prompt
    raise ValueError(f"Prompt `{prompt_style}` missing or empty in {path}")


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


def needs_disturb_rewrite(query_step: dict, source_query: str | None, target_field: str) -> bool:
    llm_label = query_step.get("llm_label")
    if not isinstance(llm_label, dict):
        return False
    if llm_label.get("entity_ambiguous") != "No":
        return False
    if not source_query:
        return False
    target_query = query_step.get(target_field)
    return not isinstance(target_query, str) or not target_query.strip()


def iter_rewrite_targets(sample: dict, source_field: str, target_field: str):
    query_steps = sample.get("query_steps")
    if isinstance(query_steps, list):
        for step_idx, query_step in enumerate(query_steps):
            if not isinstance(query_step, dict):
                continue
            source_query = normalize_query_key(query_step.get(source_field, ""))
            if not needs_disturb_rewrite(query_step, source_query, target_field):
                continue
            yield {
                "container": "query_steps",
                "index": step_idx,
                "target": query_step,
                "step_id": query_step.get("step_index", step_idx + 1),
                "source_query": source_query,
            }
            return
        return

    trajectory = sample.get("trajectory")
    if not isinstance(trajectory, list):
        return

    for step_idx, step in enumerate(trajectory):
        if not isinstance(step, dict) or step.get("action") != "search":
            continue
        source_query = normalize_query_key(
            step.get(source_field, "") or extract_text_retrieval_query_from_search(step.get("content", ""))
        )
        if not needs_disturb_rewrite(step, source_query, target_field):
            continue
        yield {
            "container": "trajectory",
            "index": step_idx,
            "target": step,
            "step_id": step_idx + 1,
            "source_query": source_query,
        }
        return


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


def sample_id_of(sample: dict) -> str:
    for key in ("id", "annotation_id", "data_id"):
        sample_id = sample.get(key)
        if isinstance(sample_id, str) and sample_id.strip():
            return sample_id
    return ""


def extract_json_block(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in model output: {text!r}")
    return json.loads(text[start : end + 1])


def render_prompt(template: str, query: str) -> str:
    return template.replace("[QUERY]", query)


def normalize_result(
    data: dict,
    fallback_query: str,
    include_meta: bool,
) -> tuple[str, dict | None]:
    rewritten_query = data.get("rewritten_query", fallback_query)
    if rewritten_query is None:
        rewritten_query = fallback_query
    if not isinstance(rewritten_query, str):
        rewritten_query = str(rewritten_query)
    rewritten_query = rewritten_query.strip() or fallback_query
    if not include_meta:
        return rewritten_query, None
    meta = dict(data)
    meta["rewritten_query"] = rewritten_query
    return rewritten_query, meta


def rewrite_query_with_llm(
    client: OpenAI,
    model: str,
    prompt: str,
    prompt_style: str,
    sample_id: str,
    query_index: int,
    fallback_query: str,
) -> tuple[str, dict | None]:
    last_error = None
    for attempt in range(RETRIES + 1):
        try:
            request_kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a query rewriting assistant."},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "timeout": TIMEOUT,
            }
            if prompt_style == "json":
                request_kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**request_kwargs)
            content = response.choices[0].message.content or ""
            with PRINT_LOCK:
                print(
                    f"\n[DISTURB REWRITE BLOCK] sample_id={sample_id} step_index={query_index}",
                    flush=True,
                )
                print("[BEFORE]", flush=True)
                print(fallback_query, flush=True)
                print("[LLM RAW]", flush=True)
                print(content, flush=True)
            if prompt_style == "json":
                return normalize_result(extract_json_block(content), fallback_query, True)
            rewritten_query = content.strip() or fallback_query
            return rewritten_query, None
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < RETRIES:
                time.sleep(min(2**attempt, 4))
    raise RuntimeError(
        f"Failed sample_id={sample_id} query_index={query_index}: {last_error}"
    ) from last_error


def build_existing_cache(
    items: list[dict],
    source_field: str,
    target_field: str,
    meta_field: str,
) -> dict[str, tuple[str, dict | None]]:
    cache: dict[str, tuple[str, dict | None]] = {}
    for sample in items:
        for target_info in iter_rewrite_targets(sample, source_field, target_field):
            query_step = target_info["target"]
            source_query = target_info["source_query"]
            target_query = query_step.get(target_field)
            if not isinstance(target_query, str) or not target_query.strip():
                continue
            meta = query_step.get(meta_field)
            cache[source_query] = (target_query.strip(), meta if isinstance(meta, dict) else None)
    return cache


def filter_items_by_allowed_ids(items: list[dict], allowed_ids: set[str] | None) -> list[dict]:
    if allowed_ids is None:
        return items
    filtered = []
    for sample in items:
        sample_id = sample_id_of(sample)
        if sample_id in allowed_ids:
            filtered.append(sample)
    return filtered


def build_multi_source_paths() -> list[tuple[Path, Path, Path]]:
    runs: list[tuple[Path, Path, Path]] = []
    for source_dir in ROOT_RESULT_DIRS:
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
        print(f"\n===== [{run_idx}/{len(runs)}] Disturb Rewrite =====", flush=True)
        print(f"[Run] input={input_path}", flush=True)
        print(f"[Run] output={output_path}", flush=True)
        print(f"[Run] log={log_path}", flush=True)

        if not input_path.exists() and not output_path.exists():
            raise FileNotFoundError(f"Neither input nor output exists for run: {input_path}")

        log_file, original_stdout, original_stderr = setup_logging(log_path)
        try:
            raw_items = read_jsonl(output_path) if output_path.exists() else read_jsonl(input_path)
            allowed_ids = load_allowed_ids(DEFAULT_SUBSET_PATH)
            items = filter_items_by_allowed_ids(raw_items, allowed_ids)
            if output_path.exists():
                print(f"Resuming from existing output: {output_path}")
            else:
                print(f"Starting new disturb rewrite run from: {input_path}")
            if allowed_ids is not None:
                print(f"Restrict rewriting to {len(allowed_ids)} ids from: {DEFAULT_SUBSET_PATH}")
                print(f"Filtered output rows to {len(items)} subset items.", flush=True)

            prompt_template = load_prompt_template(args.prompt, PROMPT_STYLE)
            client = OpenAI(api_key=api_key, base_url=BASE_URL)

            futures: dict[str, object] = {}
            future_targets: dict = defaultdict(list)
            pending = 0
            completed = 0
            failed = 0
            cache_hits = 0
            dedup_hits = 0
            rewrite_cache = build_existing_cache(
                items, SOURCE_FIELD, TARGET_FIELD, META_FIELD
            )

            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                for sample_idx, sample in enumerate(items):
                    sample_id = sample_id_of(sample)
                    if allowed_ids is not None and sample_id not in allowed_ids:
                        continue
                    for target_info in iter_rewrite_targets(
                        sample, SOURCE_FIELD, TARGET_FIELD
                    ):
                        source_query = target_info["source_query"]
                        query_step = target_info["target"]
                        if source_query in rewrite_cache:
                            rewritten_query, meta = rewrite_cache[source_query]
                            query_step[TARGET_FIELD] = rewritten_query
                            if meta is not None:
                                query_step[META_FIELD] = meta
                            cache_hits += 1
                            continue

                        prompt = render_prompt(prompt_template, source_query)
                        if source_query in futures:
                            future_targets[futures[source_query]].append(
                                (sample_idx, target_info["container"], target_info["index"])
                            )
                            dedup_hits += 1
                            continue

                        future = executor.submit(
                            rewrite_query_with_llm,
                            client,
                            MODEL,
                            prompt,
                            PROMPT_STYLE,
                            sample_id,
                            target_info["step_id"],
                            source_query,
                        )
                        futures[source_query] = future
                        future_targets[future].append(
                            (sample_idx, target_info["container"], target_info["index"])
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
                    try:
                        rewritten_query, meta = future.result()
                    except Exception as exc:  # noqa: BLE001
                        failed += 1
                        error_message = str(exc)
                        print(
                            f"\n[Disturb Rewrite Failed] {error_message}",
                            file=sys.stderr,
                            flush=True,
                        )
                        for sample_idx, container, step_idx in targets:
                            query_step = get_step_container(items[sample_idx], container)[step_idx]
                            query_step["disturb_rewrite_error"] = error_message
                        completed += 1
                        write_jsonl(output_path, items)
                        sys.stdout.write("\r" + render_progress(completed, pending))
                        sys.stdout.flush()
                        continue

                    for sample_idx, container, step_idx in targets:
                        query_step = get_step_container(items[sample_idx], container)[step_idx]
                        query_step[TARGET_FIELD] = rewritten_query
                        query_step.pop("disturb_rewrite_error", None)
                        if meta is not None:
                            query_step[META_FIELD] = meta
                        source_query = normalize_query_key(query_step.get(SOURCE_FIELD, ""))
                        if source_query is None and container == "trajectory":
                            source_query = normalize_query_key(
                                extract_text_retrieval_query_from_search(query_step.get("content", ""))
                            )
                        if source_query:
                            rewrite_cache[source_query] = (rewritten_query, meta)
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
