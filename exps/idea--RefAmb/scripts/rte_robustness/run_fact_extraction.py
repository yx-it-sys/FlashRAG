#!/usr/bin/env python3
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

from common import (
    DEFAULT_FACT_PROMPT,
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT,
    build_iterations,
    build_vllm_generator,
    infer_local_model_name,
    load_prompt,
    local_vllm_generate_json_batch,
    make_openai_client,
    normalize_extracted_facts,
    normalize_whitespace,
    read_jsonl,
    render_chat_prompt,
    render_prompt,
    resolve_run_config_path,
    split_evidence_into_docs,
    write_json,
)

INPUT_PATH = Path(
    "/home/you/FlashRAG/exps/idea10/data/result/cross-model-rte/label/human/TQS/human_label_150.trajectories.jsonl"
)
OUTPUT_JSONL = Path(
    "/home/you/FlashRAG/exps/idea10/scripts/rte_robustness/qwen3_32b/fact_outputs.jsonl"
)
OUTPUT_SUMMARY = Path(
    "/home/you/FlashRAG/exps/idea10/scripts/rte_robustness/qwen3_32b/fact_outputs.summary.json"
)

MODEL_MODE = "api"  # "vllm" or "api"
# MODEL_REF = "/mnt/data/you/modelscope/Qwen2.5-7B-Instruct"  # local abs path if vllm; API model name if api
MODEL_REF = "Qwen/Qwen3-VL-32B-Instruct"
API_BASE_URL = "https://api.siliconflow.cn/v1"
API_KEY_ENV = "SILICONFLOW_API_KEY"
MAX_DOCS_PER_ITERATION = 10
MAX_CHARS_PER_DOC = 1800
MAX_TOTAL_EVIDENCE_CHARS = 5000
MAX_NEW_TOKENS = 1024
RETRIES = DEFAULT_RETRIES
TIMEOUT = 180.0
API_DOC_WORKERS = 8


def truncate_doc_text(doc_text: str, max_chars: int) -> str:
    text = normalize_whitespace(doc_text)
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars]
    last_boundary = max(clipped.rfind(". "), clipped.rfind("; "), clipped.rfind(" "))
    if last_boundary > max_chars * 0.6:
        clipped = clipped[: last_boundary + 1]
    return clipped.rstrip() + " ..."


def budget_evidence_chunks(evidence_chunks: list[str]) -> list[str]:
    budgeted = []
    used_chars = 0
    for chunk in evidence_chunks[:MAX_DOCS_PER_ITERATION]:
        clipped_chunk = truncate_doc_text(chunk, MAX_CHARS_PER_DOC)
        if used_chars >= MAX_TOTAL_EVIDENCE_CHARS:
            break
        remaining = MAX_TOTAL_EVIDENCE_CHARS - used_chars
        if len(clipped_chunk) > remaining:
            clipped_chunk = truncate_doc_text(clipped_chunk, remaining)
        if not clipped_chunk.strip():
            continue
        budgeted.append(clipped_chunk)
        used_chars += len(clipped_chunk)
    return budgeted


def extract_facts_for_single_prompt(system_prompt: str, user_prompt: str, client) -> list[str]:
    response = client.chat.completions.create(
        model=MODEL_REF,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
        timeout=TIMEOUT,
        response_format={"type": "json_object"},
    )
    return normalize_extracted_facts(json.loads(response.choices[0].message.content))


def extract_facts_for_iteration(query: str, evidence_text: str, generator, client) -> list[str]:
    fact_prompt = load_prompt(DEFAULT_FACT_PROMPT)
    evidence_chunks = split_evidence_into_docs(evidence_text)
    if not evidence_chunks:
        evidence_chunks = [evidence_text]
    evidence_chunks = budget_evidence_chunks(evidence_chunks)
    prompt_pairs = [
        (
            "You extract atomic fact sets for trajectory-level evaluation.",
            render_prompt(fact_prompt, query=query, evidence=chunk),
        )
        for chunk in evidence_chunks
    ]

    facts: list[str] = []
    seen = set()
    if MODEL_MODE == "vllm":
        prompt_texts = [render_chat_prompt(generator, system_prompt, user_prompt) for system_prompt, user_prompt in prompt_pairs]
        parsed_outputs = local_vllm_generate_json_batch(generator, prompt_texts, max_new_tokens=MAX_NEW_TOKENS)
        for parsed_output in parsed_outputs:
            if isinstance(parsed_output, Exception):
                continue
            for fact in normalize_extracted_facts(parsed_output):
                key = fact.lower()
                if key in seen:
                    continue
                seen.add(key)
                facts.append(fact)
    else:
        with ThreadPoolExecutor(max_workers=API_DOC_WORKERS) as executor:
            future_to_prompt = {
                executor.submit(extract_facts_for_single_prompt, system_prompt, user_prompt, client): (
                    system_prompt,
                    user_prompt,
                )
                for system_prompt, user_prompt in prompt_pairs
            }
            for future in as_completed(future_to_prompt):
                try:
                    doc_facts = future.result()
                except Exception as exc:  # noqa: BLE001
                    _, failed_user_prompt = future_to_prompt[future]
                    print(
                        "[Facts] doc extraction failed "
                        f"query={query!r} prompt_snippet={failed_user_prompt[:160]!r} error={exc}",
                        flush=True,
                    )
                    continue
                for fact in doc_facts:
                    key = fact.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    facts.append(fact)
    return facts


def process_sample(sample: dict, generator, client) -> dict:
    steps = []
    for iteration in build_iterations(sample.get("trajectory", [])):
        if iteration["retrieval_action"] == "no_retrieval_result":
            continue
        query = iteration["query"] or sample.get("question", "")
        reaction_text = normalize_whitespace(iteration.get("reaction_text", ""))
        evidence_text = normalize_whitespace(iteration.get("retrieval_content", ""))
        evidence_source = iteration.get("evidence_source", "explicit_retrieval_result")
        if not evidence_text and reaction_text:
            evidence_text = reaction_text
            evidence_source = "reaction_text_fallback"
        sub_question = normalize_whitespace(iteration.get("sub_question", "")) or normalize_whitespace(query)
        facts = extract_facts_for_iteration(query, evidence_text, generator, client) if reaction_text else []
        retrieval_type = "Image Retrieval" if iteration["retrieval_action"] == "image_retrieval_result" else "Text Retrieval"
        steps.append(
            {
                "iteration_index": iteration["iteration_index"],
                "retrieval_action": iteration["retrieval_action"],
                "retrieval_type": retrieval_type,
                "mode": iteration["mode"],
                "query": iteration["query"],
                "sub_question": sub_question,
                "retrieval_index": iteration["retrieval_index"],
                "evidence_source": evidence_source,
                "evidence_text": evidence_text,
                "reaction_index": iteration["reaction_index"],
                "reaction_action": iteration["reaction_action"],
                "reaction_text": reaction_text,
                "facts_t": facts,
            }
        )
    return {
        "id": sample.get("id"),
        "question": sample.get("question"),
        "status": sample.get("status"),
        "num_iterations": len(steps),
        "steps": steps,
    }


def main() -> None:
    generator = None
    client = None
    model_name = MODEL_REF
    model_path = ""
    if MODEL_MODE == "vllm":
        model_path_obj = Path(MODEL_REF)
        model_name = infer_local_model_name(model_path_obj)
        model_path = str(model_path_obj)
        generator = build_vllm_generator(model_path_obj, model_name, resolve_run_config_path(type("Args", (), {})()), batch_size=8)
    else:
        client = make_openai_client(API_BASE_URL, API_KEY_ENV)

    samples = read_jsonl(INPUT_PATH)
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_JSONL.exists():
        OUTPUT_JSONL.unlink()
    results = [None] * len(samples)
    for sample_idx, sample in enumerate(samples, start=1):
        result = process_sample(sample, generator, client)
        results[sample_idx - 1] = result
        with OUTPUT_JSONL.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(
            f"[Facts] completed {sample_idx}/{len(samples)} id={result['id']} "
            f"iters={result['num_iterations']}",
            flush=True,
        )

    final_results = [item for item in results if item is not None]
    write_json(
        OUTPUT_SUMMARY,
        {
            "meta": {
                "input_path": str(INPUT_PATH),
                "output_jsonl": str(OUTPUT_JSONL),
                "model_mode": MODEL_MODE,
                "model_ref": MODEL_REF,
                "resolved_local_model_name": model_name,
                "resolved_local_model_path": model_path,
                "num_samples": len(final_results),
                "mean_iterations": mean(item["num_iterations"] for item in final_results) if final_results else 0.0,
            }
        },
    )


if __name__ == "__main__":
    main()
