#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from string import Formatter
import tomllib
from openai import OpenAI

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from flashrag.config import Config
from flashrag.utils import get_generator

for k in [
    "http_proxy", "https_proxy",
    "HTTP_PROXY", "HTTPS_PROXY",
    "all_proxy", "ALL_PROXY"
]:
    os.environ.pop(k, None)

ROOT_RESULT_DIRS = [
    Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_MMSearch-R1-7B/RefAmb_2026_06_06_14_01_refamb_crag_mmsearch_r1_7b_stage"),
    Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_MMSearch-R1-7B/RefAmb_2026_06_06_16_59_refamb_oven_mmsearch_r1_7b_stage"),
    Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_MMSearch-R1-7B/RefAmb_2026_06_06_17_14_refamb_mcsearch_mmsearch_r1_7b_stage"),
    Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_MMSearch-R1-7B/RefAmb_2026_06_06_17_52_refamb_infoseek_mmsearch_r1_7b_stage"),
    # Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_Qwen3-vl-8b/RefAmb_2026_05_23_12_53_refamb_mcsearch_qwen3_vl_8b_stage"),
    # Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_qwen3_vl_32b/RefAmb_2026_05_19_21_03_refamb_mcsearch_qwen3_vl_32b_stage"),
]
INPUT_REL_PATH = Path("omnisearch_trajectories.jsonl")
OUTPUT_REL_PATH = Path("label/deepseek/omnisearch_trajectories.entity_ambiguity_labeled.jsonl")
PROMPT_PATH = Path("/home/you/FlashRAG/exps/idea10/prompts/label_entitiy_ambiguouty.toml")
MODEL = "Pro/deepseek-ai/DeepSeek-V3"
BASE_URL = "https://api.siliconflow.cn/v1"
INFERENCE_BACKEND = "api"  # "api" or "vllm_qwen25_7b"
LOCAL_MODEL_NAME = "Qwen2.5-7B-Instruct"
LOCAL_MODEL_PATH = Path("/mnt/data/you/modelscope/Qwen2.5-7B-Instruct")
RUN_CONFIG_PATH = Path("/home/you/FlashRAG/exps/idea10/configs/config.yaml")
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


def is_labeled(target: dict) -> bool:
    llm_label = target.get("llm_label")
    if not isinstance(llm_label, dict):
        return False
    entity = llm_label.get("entity_ambiguous")
    if entity == "No":
        return True
    if entity == "Yes":
        return True
    return False


def extract_json_block(text: str) -> dict:
    fenced_blocks = re.findall(r"```json\s*(\{.*?\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    for block in fenced_blocks:
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        start = match.start()
        try:
            parsed, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise ValueError(f"No JSON object found in model output: {text!r}")


def normalize_result(data: dict) -> dict:
    entity = data.get("entity_ambiguous")
    level = LABEL_ALIASES.get(data.get("ambiguity_level"), data.get("ambiguity_level"))
    remark = data.get("remark", "")

    if entity not in {"Yes", "No"}:
        raise ValueError(f"Invalid entity_ambiguous: {entity!r}")
    if entity == "No":
        level = None
    elif level is None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Label entity ambiguity for text retrieval queries in trajectory annotations "
            "or omnisearch trajectories across multiple mcsearch result directories."
        )
    )
    parser.add_argument("--prompt", type=Path, default=PROMPT_PATH)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--backend", choices=("api", "vllm_qwen25_7b"), default=INFERENCE_BACKEND)
    parser.add_argument("--local-model-name", default=LOCAL_MODEL_NAME)
    parser.add_argument("--local-model-path", type=Path, default=LOCAL_MODEL_PATH)
    return parser.parse_args()


def extract_text_retrieval_query_from_search(content: str) -> str:
    if not isinstance(content, str):
        return ""
    match = re.search(r'Text Retrieval[:\s"]*(.*?)(?=(?:<|$))', content, re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip().strip('"')


def iter_label_targets(sample: dict):
    query_steps = sample.get("query_steps")
    if isinstance(query_steps, list):
        for step_idx, query_step in enumerate(query_steps):
            if not isinstance(query_step, dict):
                continue
            yield {
                "container": "query_steps",
                "index": step_idx,
                "target": query_step,
                "step_id": query_step.get("step_index", step_idx + 1),
                "query": query_step.get("text_query", ""),
                "prompt_values": query_step,
            }
        return

    trajectory = sample.get("trajectory")
    if not isinstance(trajectory, list):
        return

    for step_idx, step in enumerate(trajectory):
        if not isinstance(step, dict):
            continue
        query = ""
        if step.get("mode") == "text_retrieval":
            query = step.get("query", "")
        elif step.get("action") == "search":
            query = extract_text_retrieval_query_from_search(step.get("content", ""))
        query = normalize_query_key(query) or ""
        if not query:
            continue
        yield {
            "container": "trajectory",
            "index": step_idx,
            "target": step,
            "step_id": step_idx + 1,
            "query": query,
            "prompt_values": {
                "sub_question": "",
                "text_query": query,
            },
        }


def build_existing_label_cache(items: list[dict]) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    for sample in items:
        for target_info in iter_label_targets(sample):
            target = target_info["target"]
            if not is_labeled(target):
                continue
            query_key = normalize_query_key(target_info["query"])
            if query_key is None:
                continue
            cache[query_key] = dict(target["llm_label"])
    return cache


def merge_partial_output(source_items: list[dict], existing_items: list[dict]) -> list[dict]:
    existing_by_id = {}
    for item in existing_items:
        item_id = item.get("id")
        if isinstance(item_id, str) and item_id.strip():
            existing_by_id[item_id] = item

    merged_items = []
    reused = 0
    for source_item in source_items:
        item_id = source_item.get("id")
        existing_item = existing_by_id.get(item_id)
        if existing_item is not None:
            merged_items.append(existing_item)
            reused += 1
        else:
            merged_items.append(source_item)

    print(
        f"Loaded partial existing output: reused {reused}/{len(source_items)} items from {len(existing_items)} rows",
        flush=True,
    )
    return merged_items


def resolve_result_dirs() -> list[Path]:
    root_dirs = ROOT_RESULT_DIRS
    resolved = []
    seen = set()
    for root_dir in root_dirs:
        path = Path(root_dir)
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        resolved.append(path)
    return resolved


def resolve_source_dir(result_dir: Path) -> Path:
    if not result_dir.exists():
        raise FileNotFoundError(f"Missing result dir: {result_dir}")
    if (result_dir / INPUT_REL_PATH).exists():
        return result_dir
    raise FileNotFoundError(f"Missing input file: {result_dir / INPUT_REL_PATH}")


def load_source_jobs(result_dirs: list[Path]) -> list[dict]:
    jobs = []
    for result_dir in result_dirs:
        source_dir = resolve_source_dir(result_dir)
        input_path = source_dir / INPUT_REL_PATH
        output_path = source_dir / OUTPUT_REL_PATH
        if not input_path.exists():
            raise FileNotFoundError(f"Missing input file: {input_path}")
        jobs.append(
            {
                "result_dir": result_dir,
                "source_dir": source_dir,
                "input_path": input_path,
                "output_path": output_path,
            }
        )
    return jobs


def load_allowed_sample_ids(path: Path) -> set[str]:
    allowed_ids: set[str] = set()
    for item in read_jsonl(path):
        sample_id = item.get("sample_id")
        if isinstance(sample_id, str) and sample_id.strip():
            allowed_ids.add(sample_id)
    return allowed_ids


def load_combined_items(source_jobs: list[dict]) -> list[dict]:
    combined_items = []
    for source_job in source_jobs:
        input_path = source_job["input_path"]
        output_path = source_job["output_path"]
        source_items = read_jsonl(input_path)
        print(f"Loaded {len(source_items)} items from {input_path}", flush=True)
        if output_path.exists():
            existing_items = read_jsonl(output_path)
            if len(existing_items) == len(source_items):
                items = existing_items
            else:
                items = merge_partial_output(source_items, existing_items)
            print(f"Resuming from existing output: {output_path}")
        else:
            items = source_items
            print(f"Starting new labeling run from: {input_path}")

        source_job["items"] = items
        source_job["item_count"] = len(items)
        combined_items.extend(items)
    return combined_items


def write_all_outputs(source_jobs: list[dict]) -> None:
    for source_job in source_jobs:
        write_jsonl(source_job["output_path"], source_job["items"])


def build_local_generator(model_name: str, model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Local model path does not exist: {model_path}")
    if not RUN_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Run config path does not exist: {RUN_CONFIG_PATH}")
    config = Config(
        str(RUN_CONFIG_PATH),
        config_dict={
            "disable_save": True,
            "generator_model": model_name,
            "generator_model_path": str(model_path),
            "generator_max_input_len": 8192,
            "generator_batch_size": 1,
            "framework": "vllm",
            "gpu_id": "0",
            "generation_params": {
                "max_new_tokens": 1024,
                "do_sample": False,
                "temperature": 0.0,
                "top_p": 1.0,
            },
        },
    )
    return get_generator(config)


def annotate_query(
    client: OpenAI | None,
    local_generator,
    backend: str,
    model: str,
    system_prompt: str,
    sample_id: str,
    query_index: int,
) -> dict:
    last_error = None
    for attempt in range(RETRIES + 1):
        try:
            if backend == "vllm_qwen25_7b":
                if local_generator is None:
                    raise RuntimeError("Local generator is not initialized.")
                response = local_generator.generate(
                    [system_prompt],
                    max_new_tokens=1024,
                )
                content = response[0] if isinstance(response, list) else response
            else:
                if client is None:
                    raise RuntimeError("OpenAI client is not initialized.")
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
    args = parse_args()
    api_key = None
    if args.backend == "api":
        api_key = os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Set SILICONFLOW_API_KEY or OPENAI_API_KEY before running.")

    prompt_path = args.prompt

    result_dirs = resolve_result_dirs()
    print(f"Processing {len(result_dirs)} result dirs", flush=True)
    for result_dir in result_dirs:
        print(f"  - {result_dir}", flush=True)

    source_jobs = load_source_jobs(result_dirs)
    items = load_combined_items(source_jobs)
    print(f"Loaded {len(source_jobs)} sources, total items: {len(items)}", flush=True)

    system_template = load_system_prompt(prompt_path)
    client = OpenAI(api_key=api_key, base_url=args.base_url) if args.backend == "api" else None
    local_generator = None
    if args.backend == "vllm_qwen25_7b":
        print(f"Loading local vLLM generator: {args.local_model_path}", flush=True)
        local_generator = build_local_generator(args.local_model_name, args.local_model_path)
        print("Finished loading local vLLM generator.", flush=True)

    lock = threading.Lock()
    futures = {}
    pending = 0
    completed = 0
    cache_hits = 0
    dedup_hits = 0
    label_cache = build_existing_label_cache(items)

    pending_tasks = []
    for sample_idx, sample in enumerate(items):
        for target_info in iter_label_targets(sample):
            target = target_info["target"]
            if is_labeled(target):
                continue
            query_key = normalize_query_key(target_info["query"])
            if query_key is not None and query_key in label_cache:
                target["llm_label"] = dict(label_cache[query_key])
                cache_hits += 1
                continue
            prompt = render_prompt(system_template, sample, target_info["prompt_values"])
            if query_key is not None and query_key in futures:
                pending_tasks[futures[query_key]]["targets"].append(
                    (sample_idx, target_info["container"], target_info["index"])
                )
                dedup_hits += 1
                continue
            task = {
                "client": client,
                "local_generator": local_generator,
                "backend": args.backend,
                "model": args.model,
                "prompt": prompt,
                "sample_id": sample.get("annotation_id") or sample.get("id", ""),
                "step_id": target_info["step_id"],
                "targets": [(sample_idx, target_info["container"], target_info["index"])],
                "query_key": query_key,
            }
            if query_key is not None:
                futures[query_key] = len(pending_tasks)
            pending_tasks.append(task)
            pending += 1

    print(f"Pending queries: {pending}")
    print(
        f"Skipped by existing label cache: {cache_hits}; "
        f"deduplicated in current run: {dedup_hits}"
    )
    print(render_progress(0, pending), flush=True)

    if cache_hits > 0:
        write_all_outputs(source_jobs)

    if args.backend == "vllm_qwen25_7b":
        print("[Execution] backend=vllm_qwen25_7b, running sequentially without multithreading.", flush=True)
        for task_idx, task in enumerate(pending_tasks):
            result = annotate_query(
                task["client"],
                task["local_generator"],
                task["backend"],
                task["model"],
                task["prompt"],
                task["sample_id"],
                task["step_id"],
            )
            with lock:
                for sample_idx, container, step_idx in task["targets"]:
                    target = items[sample_idx][container][step_idx]
                    target["llm_label"] = dict(result)
                    query_key = task["query_key"]
                    if query_key is not None:
                        label_cache[query_key] = dict(result)
                completed += 1
                write_all_outputs(source_jobs)
                sys.stdout.write("\r" + render_progress(completed, pending))
                sys.stdout.flush()
    else:
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            submitted = {}
            for task_idx, task in enumerate(pending_tasks):
                future = executor.submit(
                    annotate_query,
                    task["client"],
                    task["local_generator"],
                    task["backend"],
                    task["model"],
                    task["prompt"],
                    task["sample_id"],
                    task["step_id"],
                )
                submitted[future] = task_idx
            for future in as_completed(submitted):
                task = pending_tasks[submitted[future]]
                result = future.result()
                with lock:
                    for sample_idx, container, step_idx in task["targets"]:
                        target = items[sample_idx][container][step_idx]
                        target["llm_label"] = dict(result)
                        query_key = task["query_key"]
                        if query_key is not None:
                            label_cache[query_key] = dict(result)
                    completed += 1
                    write_all_outputs(source_jobs)
                    sys.stdout.write("\r" + render_progress(completed, pending))
                    sys.stdout.flush()

    if pending == 0:
        write_all_outputs(source_jobs)
    elif pending > 0:
        print()
    print("Done. Outputs:", flush=True)
    for source_job in source_jobs:
        print(f"  - {source_job['output_path']}", flush=True)


if __name__ == "__main__":
    main()
