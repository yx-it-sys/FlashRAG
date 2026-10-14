#!/usr/bin/env python3
import os
from pathlib import Path
from statistics import mean

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from evaluate_trajectory_quality_for_test import (
    append_jsonl,
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT,
    DEFAULT_UTILITY_PROMPT,
    LLMJudge,
    OpenAI,
    UtilityOutputParseError,
    build_local_utility_judge_generator,
    load_prompt,
    read_jsonl,
    resolve_run_config_path,
    write_json,
    write_jsonl,
)

DEFAULT_INPUT = Path(
    os.environ.get(
        "TRAJECTORY_UTILITY_INPUT",
        "/home/you/FlashRAG/exps/idea10/data/result/cross-model-rte/label/human/TQS/llama_fact_stage/trajectory_facts.jsonl",
    )
)
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get(
        "TRAJECTORY_UTILITY_OUTPUT_DIR",
        "/home/you/FlashRAG/exps/idea10/data/result/cross-model-rte/label/human/TQS/llama_qwen_eval",
    )
)
DEFAULT_LOCAL_UTILITY_MODEL = os.environ.get(
    "TRAJECTORY_UTILITY_MODEL_NAME",
    "Qwen/Qwen3-VL-32B-Instruct",
)
DEFAULT_LOCAL_UTILITY_MODEL_PATH = Path(
    os.environ.get(
        "TRAJECTORY_UTILITY_MODEL_PATH",
        "/mnt/data/you/modelscope/Qwen2.5-7B-Instruct",
    )
)
DEFAULT_UTILITY_BACKEND = os.environ.get("TRAJECTORY_UTILITY_BACKEND", "api")
DEFAULT_UTILITY_API_BASE_URL = os.environ.get(
    "TRAJECTORY_UTILITY_BASE_URL",
    "https://api.siliconflow.cn/v1",
)
DEFAULT_UTILITY_API_KEY_ENV = os.environ.get(
    "TRAJECTORY_UTILITY_API_KEY_ENV",
    "SILICONFLOW_API_KEY",
)
DEFAULT_CACHE_FLUSH_EVERY = 1


class Args:
    input = DEFAULT_INPUT
    output_dir = DEFAULT_OUTPUT_DIR
    utility_prompt = DEFAULT_UTILITY_PROMPT
    utility_judge_backend = DEFAULT_UTILITY_BACKEND
    local_utility_model = DEFAULT_LOCAL_UTILITY_MODEL
    local_utility_model_path = DEFAULT_LOCAL_UTILITY_MODEL_PATH
    local_fact_model = "Meta-Llama-3-8B-Instruct"
    local_fact_model_path = Path("/mnt/data/you/modelscope/Meta-Llama-3-8B-Instruct")
    model = DEFAULT_LOCAL_UTILITY_MODEL
    fact_api_model = "Qwen/Qwen2.5-7B-Instruct"
    timeout = DEFAULT_TIMEOUT
    retries = DEFAULT_RETRIES
    max_docs_per_iteration = 10
    fact_extractor = "local_qwen"
    cache_flush_every = DEFAULT_CACHE_FLUSH_EVERY
    base_url = DEFAULT_UTILITY_API_BASE_URL
    api_key_env = DEFAULT_UTILITY_API_KEY_ENV


def score_sample(sample: dict, judge: LLMJudge) -> dict:
    step_rows = []
    for step in sample.get("steps", []):
        facts = list(step.get("facts_t", []))
        sub_question = step.get("sub_question", "")
        reaction_text = step.get("reaction_text", "")
        if step.get("delta_reason") == "similarity_gate":
            utility_label = "no_contribution"
            label_reason = "Similarity gate triggered because query similarity exceeded the threshold."
            filtered_facts = []
            utility = 0.0
            utility_defined = False
            utility_reason_code = "similarity_gate"
        else:
            try:
                utility_label, label_reason, filtered_facts = judge.classify_utility(
                    sub_question=sub_question,
                    reasoning=reaction_text,
                    facts=facts,
                )
            except UtilityOutputParseError as exc:
                print(
                    f"[Utility Stage] skip id={sample.get('id')} "
                    f"iter={step.get('iteration_index')} error={exc}",
                    flush=True,
                )
                continue
            utility = {
                "no_contribution": 0.0,
                "partial_contribution": 0.5,
                "full_contribution": 1.0,
            }[utility_label]
            utility_defined = True
            utility_reason_code = "llm_three_level_utility"

        updated_step = dict(step)
        updated_step["filtered_facts_t"] = filtered_facts
        updated_step["utility_label"] = utility_label
        updated_step["utility_defined"] = utility_defined
        updated_step["utility_reason_code"] = utility_reason_code
        updated_step["utility"] = utility
        updated_step["utility_reason"] = label_reason
        updated_step["step_score"] = utility * step.get("delta_f", 0.0)
        step_rows.append(updated_step)

    return {
        "id": sample.get("id"),
        "question": sample.get("question"),
        "status": sample.get("status"),
        "num_iterations": len(step_rows),
        "total_delta_f": sum(step["delta_f"] for step in step_rows),
        "total_step_score": sum(step["step_score"] for step in step_rows),
        "trajectory_quality_score": mean(step["step_score"] for step in step_rows) if step_rows else 0.0,
        "steps": step_rows,
    }


def summarize(results: list[dict]) -> dict:
    steps = [step for sample in results for step in sample["steps"]]
    return {
        "meta": {
            "input": str(DEFAULT_INPUT),
            "output_dir": str(DEFAULT_OUTPUT_DIR),
            "local_utility_model": DEFAULT_LOCAL_UTILITY_MODEL,
            "local_utility_model_path": str(DEFAULT_LOCAL_UTILITY_MODEL_PATH),
            "num_samples": len(results),
            "num_steps": len(steps),
        },
        "aggregate": {
            "mean_tqs": mean(item["trajectory_quality_score"] for item in results) if results else 0.0,
            "mean_iterations": mean(item["num_iterations"] for item in results) if results else 0.0,
            "mean_delta_f": mean(step["delta_f"] for step in steps) if steps else 0.0,
            "mean_utility": mean(step["utility"] for step in steps) if steps else 0.0,
            "mean_step_score": mean(step["step_score"] for step in steps) if steps else 0.0,
        },
    }


def main() -> None:
    args = Args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[Utility Stage] input={args.input} output_dir={args.output_dir} "
        f"backend={args.utility_judge_backend} "
        f"utility_model={args.local_utility_model} utility_model_path={args.local_utility_model_path}",
        flush=True,
    )
    utility_prompt = load_prompt(args.utility_prompt)
    local_utility_generator = None
    client = None
    if args.utility_judge_backend == "vllm_qwen25_7b":
        local_utility_generator = build_local_utility_judge_generator(
            args,
            resolve_run_config_path(args),
        )
    elif args.utility_judge_backend == "api":
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise ValueError(f"Set {args.api_key_env} before running.")
        client = OpenAI(api_key=api_key, base_url=args.base_url)
    else:
        raise ValueError(f"Unsupported utility_judge_backend: {args.utility_judge_backend}")
    judge = LLMJudge(
        client=client,
        model=args.model,
        fact_api_model=args.fact_api_model,
        timeout=args.timeout,
        retries=args.retries,
        max_docs_per_iteration=args.max_docs_per_iteration,
        fact_extractor=args.fact_extractor,
        local_fact_generator=None,
        utility_judge_backend=args.utility_judge_backend,
        local_utility_generator=local_utility_generator,
        fact_prompt="",
        utility_prompt=utility_prompt,
        fact_cache_path=args.output_dir / "fact_cache_unused.json",
        utility_cache_path=args.output_dir / "utility_cache.json",
        cache_flush_every=args.cache_flush_every,
    )

    samples = read_jsonl(args.input)
    output_path = args.output_dir / "trajectory_quality_samples.jsonl"
    summary_path = args.output_dir / "trajectory_quality_summary.json"
    if output_path.exists():
        output_path.unlink()
    if summary_path.exists():
        summary_path.unlink()

    results = []
    total = len(samples)
    for idx, sample in enumerate(samples, start=1):
        result = score_sample(sample, judge)
        results.append(result)
        append_jsonl(output_path, result)
        print(
            f"[Utility Stage] completed {idx}/{total} id={result['id']} "
            f"tqs={result['trajectory_quality_score']:.4f}",
            flush=True,
        )
    judge.flush_caches()
    write_json(summary_path, summarize(results))


if __name__ == "__main__":
    main()
