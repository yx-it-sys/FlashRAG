#!/usr/bin/env python3
import contextlib
import io
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_SCRIPTS_DIR = SCRIPT_DIR.parent
if str(PARENT_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_SCRIPTS_DIR))

from openai import OpenAI

from flashrag.config import Config
from flashrag.utils import get_generator
from evaluate_trajectory_quality_for_test import (
    DEFAULT_FACT_PROMPT,
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT,
    DEFAULT_UTILITY_PROMPT,
    DenseSimilarity,
    build_iterations,
    classify_facts_by_novelty,
    extract_json_block,
    load_prompt,
    normalize_fact,
    normalize_utility_label,
    normalize_whitespace,
    read_jsonl,
    render_prompt,
    resolve_retriever_config_path,
    resolve_run_config_path,
    split_evidence_into_docs,
    write_json,
)

UTILITY_TO_SCORE = {
    "no_contribution": 0.0,
    "partial_contribution": 0.5,
    "full_contribution": 1.0,
}


def build_vllm_generator(model_path: Path, model_name: str, run_config_path: Path, batch_size: int):
    config = Config(
        str(run_config_path),
        config_dict={
            "disable_save": True,
            "generator_model": model_name,
            "generator_model_path": str(model_path),
            "generator_max_input_len": 8192,
            "generator_batch_size": batch_size,
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


def infer_local_model_name(model_path: Path) -> str:
    return model_path.name.rstrip("/")


def make_openai_client(base_url: str, api_key_env: str) -> OpenAI:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f"Set {api_key_env} before running.")
    return OpenAI(api_key=api_key, base_url=base_url)


def render_chat_prompt(generator, system_prompt: str, user_prompt: str) -> str:
    tokenizer = getattr(generator, "tokenizer", None)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"


def local_vllm_generate_json(generator, prompt_text: str, max_new_tokens: int = 1024) -> dict:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        response = generator.generate([prompt_text], max_new_tokens=max_new_tokens)
    content = response[0] if isinstance(response, list) else response
    return extract_json_block(content or "")


def local_vllm_generate_json_batch(generator, prompt_texts: list[str], max_new_tokens: int = 1024) -> list[dict | Exception]:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        responses = generator.generate(prompt_texts, max_new_tokens=max_new_tokens)
    if not isinstance(responses, list):
        responses = [responses]
    outputs: list[dict | Exception] = []
    for item in responses:
        try:
            outputs.append(extract_json_block(item or ""))
        except Exception as exc:  # noqa: BLE001
            outputs.append(exc)
    return outputs


def normalize_extracted_facts(data: dict) -> list[str]:
    raw_facts = data.get("facts", [])
    if not isinstance(raw_facts, list):
        raise ValueError(f"`facts` must be a list: {data!r}")
    facts = []
    seen = set()
    for fact in raw_facts:
        if fact is None:
            continue
        if not isinstance(fact, str):
            fact = str(fact)
        fact = normalize_whitespace(fact)
        if not fact:
            continue
        key = normalize_fact(fact)
        if not key or key in seen:
            continue
        seen.add(key)
        if not fact.endswith("."):
            fact += "."
        facts.append(fact)
    return facts


def write_jsonl_incremental(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
