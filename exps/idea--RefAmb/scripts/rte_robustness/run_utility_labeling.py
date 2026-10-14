#!/usr/bin/env python3
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

from common import (
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT,
    DEFAULT_UTILITY_PROMPT,
    UTILITY_TO_SCORE,
    build_iterations,
    build_vllm_generator,
    infer_local_model_name,
    load_prompt,
    local_vllm_generate_json,
    make_openai_client,
    normalize_utility_label,
    normalize_whitespace,
    read_jsonl,
    render_chat_prompt,
    render_prompt,
    resolve_run_config_path,
    write_json,
)

INPUT_PATH = Path(
    "/home/you/FlashRAG/exps/idea10/data/result/cross-model-rte/label/human/TQS/human_label_150.trajectories.jsonl"
)
OUTPUT_JSONL = Path(
    "/home/you/FlashRAG/exps/idea10/scripts/rte_robustness/qwen3_32b/utility_outputs.jsonl"
)
OUTPUT_SUMMARY = Path(
    "/home/you/FlashRAG/exps/idea10/scripts/rte_robustness/qwen3_32b/utility_outputs.summary.json"
)

MODEL_MODE = "api"  # "vllm" or "api"
# MODEL_REF = "/mnt/data/you/modelscope/Qwen2.5-7B-Instruct"  # local abs path if vllm; API model name if api
MODEL_REF = "Qwen/Qwen3-32B"
API_BASE_URL = "https://api.siliconflow.cn/v1"
API_KEY_ENV = "SILICONFLOW_API_KEY"
MAX_NEW_TOKENS = 1024
RETRIES = DEFAULT_RETRIES
TIMEOUT = DEFAULT_TIMEOUT
API_WORKERS = 8


def classify_utility(sub_question: str, reasoning: str, generator, client) -> tuple[str, str, float]:
    system_prompt = "You label iteration utility for trajectory-level evaluation."
    user_prompt = render_prompt(
        load_prompt(DEFAULT_UTILITY_PROMPT),
        sub_question=sub_question,
        reasoning=reasoning,
    )
    if MODEL_MODE == "vllm":
        prompt_text = render_chat_prompt(generator, system_prompt, user_prompt)
        data = local_vllm_generate_json(generator, prompt_text, max_new_tokens=MAX_NEW_TOKENS)
    else:
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
        data = json.loads(response.choices[0].message.content)
    label = normalize_utility_label(data.get("label"))
    reason = normalize_whitespace(str(data.get("reason", "")))
    if not label:
        raise ValueError(f"Unsupported utility label: {data!r}")
    return label, reason, UTILITY_TO_SCORE[label]


def process_sample(sample: dict, generator, client) -> dict:
    steps = []
    for iteration in build_iterations(sample.get("trajectory", [])):
        if iteration["retrieval_action"] == "no_retrieval_result":
            continue
        query = iteration["query"] or sample.get("question", "")
        sub_question = normalize_whitespace(iteration.get("sub_question", "")) or normalize_whitespace(query)
        reaction_text = normalize_whitespace(iteration.get("reaction_text", ""))
        if not reaction_text:
            continue
        try:
            utility_label, utility_reason, utility_score = classify_utility(
                sub_question=sub_question,
                reasoning=reaction_text,
                generator=generator,
                client=client,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[Utility] skip id={sample.get('id')} iter={iteration['iteration_index']} error={exc}",
                flush=True,
            )
            continue
        steps.append(
            {
                "iteration_index": iteration["iteration_index"],
                "query": iteration["query"],
                "sub_question": sub_question,
                "reaction_text": reaction_text,
                "utility_label": utility_label,
                "utility_reason": utility_reason,
                "utility": utility_score,
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
        generator = build_vllm_generator(model_path_obj, model_name, resolve_run_config_path(type("Args", (), {})()), batch_size=1)
    else:
        client = make_openai_client(API_BASE_URL, API_KEY_ENV)

    samples = read_jsonl(INPUT_PATH)
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_JSONL.exists():
        OUTPUT_JSONL.unlink()
    results = [None] * len(samples)
    if MODEL_MODE == "api":
        with ThreadPoolExecutor(max_workers=API_WORKERS) as executor:
            future_to_idx = {
                executor.submit(process_sample, sample, generator, client): idx
                for idx, sample in enumerate(samples)
            }
            completed = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                result = future.result()
                results[idx] = result
                completed += 1
                with OUTPUT_JSONL.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                print(
                    f"[Utility] completed {completed}/{len(samples)} id={result['id']} "
                    f"iters={result['num_iterations']}",
                    flush=True,
                )
    else:
        for sample_idx, sample in enumerate(samples, start=1):
            result = process_sample(sample, generator, client)
            results[sample_idx - 1] = result
            with OUTPUT_JSONL.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(
                f"[Utility] completed {sample_idx}/{len(samples)} id={result['id']} "
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
