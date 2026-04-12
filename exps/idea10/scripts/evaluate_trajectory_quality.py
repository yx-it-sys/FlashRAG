#!/usr/bin/env python3
import argparse
import io
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from string import Formatter
import tomllib

import numpy as np
import yaml
from openai import OpenAI
from flashrag.retriever.encoder import Encoder

for k in [
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
]:
    os.environ.pop(k, None)


DEFAULT_INPUT = Path(
    "/home/you/FlashRAG/exps/idea10/data/result/crag_mm_2026_03_31_14_06_experiment/omnisearch_trajectories.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/you/FlashRAG/exps/idea10/data/result/crag_mm_2026_03_31_14_06_experiment/trajectory_quality_eval"
)
DEFAULT_FACT_PROMPT = Path(
    "/home/you/FlashRAG/exps/idea10/prompts/extract_atomic_fact_set.toml"
)
DEFAULT_UTILITY_PROMPT = Path(
    "/home/you/FlashRAG/exps/idea10/prompts/label_iteration_utility.toml"
)
DEFAULT_MODEL = os.environ.get("TRAJECTORY_EVAL_MODEL", "Pro/deepseek-ai/DeepSeek-R1")
DEFAULT_BASE_URL = os.environ.get("TRAJECTORY_EVAL_BASE_URL", "https://api.siliconflow.cn/v1")
DEFAULT_API_KEY_ENV = "SILICONFLOW_API_KEY"
DEFAULT_WORKERS = 2
DEFAULT_TIMEOUT = 120.0
DEFAULT_RETRIES = 2
DEFAULT_SIM_THRESHOLD = 0.85
DEFAULT_EMBED_BATCH_SIZE = 64

UTILITY_TO_SCORE = {
    "no_contribution": 0.0,
    "partial_contribution": 0.5,
    "full_contribution": 1.0,
}


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trajectory quality with Information Increment, Iteration Utility, and TQS."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fact-prompt", type=Path, default=DEFAULT_FACT_PROMPT)
    parser.add_argument("--utility-prompt", type=Path, default=DEFAULT_UTILITY_PROMPT)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", type=str, default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--sim-threshold", type=float, default=DEFAULT_SIM_THRESHOLD)
    parser.add_argument("--embed-batch-size", type=int, default=DEFAULT_EMBED_BATCH_SIZE)
    parser.add_argument("--retriever-config", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log", type=Path, default=None)
    return parser.parse_args()


def setup_logging(log_path: Path) -> tuple[io.TextIOBase, io.TextIOBase, io.TextIOBase]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_file)
    sys.stderr = TeeStream(original_stderr, log_file)
    print(f"[Logging] Appending stdout/stderr to {log_path}", flush=True)
    return log_file, original_stdout, original_stderr


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


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_prompt(path: Path) -> str:
    with path.open("rb") as f:
        data = tomllib.load(f)
    prompt = data.get("system_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Missing `system_prompt` in {path}")
    return prompt


def render_prompt(template: str, **values: str) -> str:
    required = {
        field_name
        for _, field_name, _, _ in Formatter().parse(template)
        if field_name
    }
    missing = [key for key in required if key not in values]
    if missing:
        raise KeyError(f"Missing prompt variables: {missing}")

    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", value)
    return rendered


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def strip_trailing_tag(text: str) -> str:
    return re.sub(r"</[^>]+>\s*$", "", text or "", flags=re.IGNORECASE).strip()


def parse_search_query(content: str) -> str:
    text = strip_trailing_tag(content)
    if ":" in text:
        text = text.split(":", 1)[1]
    return normalize_whitespace(text.strip().strip('"'))


def extract_json_block(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in model output: {text!r}")
    return json.loads(text[start : end + 1])


def normalize_fact(text: str) -> str:
    text = normalize_whitespace(text).rstrip(".")
    return text.lower()


def resolve_retriever_config_path(args: argparse.Namespace) -> Path:
    if args.retriever_config is not None:
        return args.retriever_config
    candidate = args.input.parent / "config.yaml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        "Could not infer retriever config from input path. Pass --retriever-config explicitly."
    )


class DenseSimilarity:
    def __init__(self, config_path: Path, batch_size: int) -> None:
        self.config_path = config_path
        self.batch_size = batch_size
        self.cache_lock = threading.Lock()
        self.encode_lock = threading.Lock()
        self.embedding_cache: dict[str, np.ndarray] = {}
        self.encoder = self._load_encoder(config_path)

    def _load_encoder(self, config_path: Path) -> Encoder:
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        retriever_cfg = config.get("text_retriever_config") or {}
        retrieval_method = retriever_cfg.get("retrieval_method") or config.get("retrieval_method")
        retrieval_model_path = retriever_cfg.get("retrieval_model_path") or config.get("retrieval_model_path")
        pooling_method = retriever_cfg.get("retrieval_pooling_method") or config.get(
            "retrieval_pooling_method", "mean"
        )
        max_length = retriever_cfg.get("retrieval_query_max_length") or config.get(
            "retrieval_query_max_length", 64
        )
        use_fp16 = retriever_cfg.get("retrieval_use_fp16")
        if use_fp16 is None:
            use_fp16 = config.get("retrieval_use_fp16", True)
        instruction = retriever_cfg.get("instruction")
        if instruction is None:
            instruction = config.get("instruction")

        if not retrieval_method or not retrieval_model_path:
            raise ValueError(f"Invalid text retriever config in {config_path}")

        return Encoder(
            model_name=retrieval_method,
            model_path=retrieval_model_path,
            pooling_method=pooling_method,
            max_length=max_length,
            use_fp16=use_fp16,
            instruction=instruction,
            silent=True,
        )

    def encode(self, query: str) -> np.ndarray | None:
        query = normalize_whitespace(query)
        if not query:
            return None
        with self.cache_lock:
            cached = self.embedding_cache.get(query)
        if cached is not None:
            return cached

        with self.encode_lock:
            with self.cache_lock:
                cached = self.embedding_cache.get(query)
            if cached is not None:
                return cached
            emb = self.encoder.encode([query], batch_size=self.batch_size, is_query=True)
            vec = np.asarray(emb[0], dtype=np.float32)
            with self.cache_lock:
                self.embedding_cache[query] = vec
            return vec

    def similarity(self, a: str, b: str) -> float | None:
        vec_a = self.encode(a)
        vec_b = self.encode(b)
        if vec_a is None or vec_b is None:
            return None
        denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
        if denom == 0:
            return None
        return float(np.dot(vec_a, vec_b) / denom)


def build_iterations(trajectory: list[dict]) -> list[dict]:
    iterations = []
    pending_search = None
    for idx, step in enumerate(trajectory):
        action = step.get("action")
        if action == "search":
            pending_search = {
                "search_index": idx,
                "query": parse_search_query(step.get("content", "")),
            }
            continue

        if action not in {"text_retrieval_result", "image_retrieval_result", "no_retrieval_result"}:
            continue

        query = normalize_whitespace(step.get("query") or "")
        if not query and pending_search:
            query = pending_search["query"]

        reaction_index = None
        reaction_action = None
        reaction_text = ""
        for later_idx in range(idx + 1, len(trajectory)):
            later = trajectory[later_idx]
            later_action = later.get("action")
            if later_action in {"thought", "final_answer"}:
                reaction_index = later_idx
                reaction_action = later_action
                reaction_text = later.get("content", "")
                break
            if later_action == "search":
                break

        iterations.append(
            {
                "iteration_index": len(iterations) + 1,
                "search_index": pending_search["search_index"] if pending_search else None,
                "retrieval_index": idx,
                "retrieval_action": action,
                "mode": step.get("mode"),
                "query": query,
                "retrieval_content": step.get("content", "") or "",
                "reaction_index": reaction_index,
                "reaction_action": reaction_action,
                "reaction_text": reaction_text,
            }
        )
        pending_search = None
    return iterations


def render_progress(completed: int, total: int, width: int = 32) -> str:
    if total <= 0:
        return "[--------------------------------] 0/0"
    filled = int(width * completed / total)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {completed}/{total}"


def build_sample_id(sample: dict) -> str:
    sample_id = sample.get("id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError(f"Sample missing valid `id`: {sample!r}")
    return sample_id


class LLMJudge:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        timeout: float,
        retries: int,
        fact_prompt: str,
        utility_prompt: str,
        fact_cache_path: Path,
        utility_cache_path: Path,
    ) -> None:
        self.client = client
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.fact_prompt = fact_prompt
        self.utility_prompt = utility_prompt
        self.fact_cache_path = fact_cache_path
        self.utility_cache_path = utility_cache_path
        self.fact_cache_lock = threading.Lock()
        self.utility_cache_lock = threading.Lock()
        self.fact_cache = self._load_cache(fact_cache_path)
        self.utility_cache = self._load_cache(utility_cache_path)

    def _load_cache(self, path: Path) -> dict[str, dict]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _save_cache(self, path: Path, lock: threading.Lock, cache: dict[str, dict]) -> None:
        with lock:
            write_json(path, dict(cache))

    def _chat_json(self, system_prompt: str, user_prompt: str) -> dict:
        last_error = None
        for attempt in range(self.retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    stream=False,
                    timeout=self.timeout,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or ""
                return extract_json_block(content)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(2**attempt, 4))
        raise RuntimeError(f"LLM request failed: {last_error}") from last_error

    def extract_facts(self, query: str, evidence: str) -> list[str]:
        cache_key = json.dumps(
            {"query": query, "evidence": evidence},
            ensure_ascii=False,
            sort_keys=True,
        )
        with self.fact_cache_lock:
            cached = self.fact_cache.get(cache_key)
        if cached is not None:
            return list(cached["facts"])

        user_prompt = render_prompt(self.fact_prompt, query=query, evidence=evidence)
        data = self._chat_json(
            "You extract atomic fact sets for trajectory-level evaluation.",
            user_prompt,
        )
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

        with self.fact_cache_lock:
            self.fact_cache[cache_key] = {"facts": facts}
            write_json(self.fact_cache_path, dict(self.fact_cache))
        return facts

    def classify_utility(self, question: str, evidence: str, reasoning: str) -> tuple[str, str]:
        cache_key = json.dumps(
            {"question": question, "evidence": evidence, "reasoning": reasoning},
            ensure_ascii=False,
            sort_keys=True,
        )
        with self.utility_cache_lock:
            cached = self.utility_cache.get(cache_key)
        if cached is not None:
            return cached["label"], cached.get("reason", "")

        user_prompt = render_prompt(
            self.utility_prompt,
            question=question,
            evidence=evidence,
            reasoning=reasoning,
        )
        data = self._chat_json(
            "You label iteration utility for trajectory-level evaluation.",
            user_prompt,
        )
        label = data.get("label")
        if label not in UTILITY_TO_SCORE:
            raise ValueError(f"Unsupported utility label: {label!r}")
        reason = data.get("reason", "")
        if not isinstance(reason, str):
            reason = str(reason)
        reason = normalize_whitespace(reason)

        with self.utility_cache_lock:
            self.utility_cache[cache_key] = {"label": label, "reason": reason}
            write_json(self.utility_cache_path, dict(self.utility_cache))
        return label, reason


def evaluate_sample(
    sample: dict,
    judge: LLMJudge,
    sim_threshold: float,
    dense_similarity: DenseSimilarity,
) -> dict:
    iterations = build_iterations(sample.get("trajectory", []))
    prev_query = None
    cumulative_fact_keys = set()
    step_rows = []

    for iteration in iterations:
        query = iteration["query"] or sample.get("question", "")
        query_sim = dense_similarity.similarity(prev_query, query) if prev_query and query else None
        facts = judge.extract_facts(query=query, evidence=iteration["retrieval_content"])
        fact_map = {normalize_fact(fact): fact for fact in facts}

        if query_sim is not None and query_sim > sim_threshold:
            delta_f = 0
            novel_facts = []
            delta_reason = "similarity_gate"
        else:
            novel_facts = [fact for key, fact in fact_map.items() if key not in cumulative_fact_keys]
            delta_f = len(novel_facts)
            delta_reason = "set_difference"

        for key in fact_map:
            cumulative_fact_keys.add(key)

        label, label_reason = judge.classify_utility(
            question=sample.get("question", ""),
            evidence=iteration["retrieval_content"],
            reasoning=iteration["reaction_text"],
        )
        utility = UTILITY_TO_SCORE[label]
        step_score = utility * delta_f

        step_rows.append(
            {
                "iteration_index": iteration["iteration_index"],
                "retrieval_action": iteration["retrieval_action"],
                "mode": iteration["mode"],
                "query": iteration["query"],
                "query_similarity_with_previous": query_sim,
                "similarity_threshold": sim_threshold,
                "retrieval_index": iteration["retrieval_index"],
                "reaction_index": iteration["reaction_index"],
                "reaction_action": iteration["reaction_action"],
                "reaction_text": iteration["reaction_text"],
                "facts_t": facts,
                "delta_f": delta_f,
                "novel_facts": novel_facts,
                "delta_reason": delta_reason,
                "utility_label": label,
                "utility": utility,
                "utility_reason": label_reason,
                "step_score": step_score,
            }
        )
        prev_query = query

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


def summarize(results: list[dict], args: argparse.Namespace) -> dict:
    steps = [step for sample in results for step in sample["steps"]]
    label_counts = {key: 0 for key in UTILITY_TO_SCORE}
    for step in steps:
        label_counts[step["utility_label"]] += 1

    return {
        "meta": {
            "input": str(args.input),
            "fact_prompt": str(args.fact_prompt),
            "utility_prompt": str(args.utility_prompt),
            "model": args.model,
            "similarity_threshold": args.sim_threshold,
            "num_samples": len(results),
            "num_steps": len(steps),
            "definitions": {
                "delta_f_t": "If cosine_similarity(q_t, q_t-1) > tau, set delta_f_t=0; otherwise delta_f_t is the number of novel atomic facts in F_<=t \\ F_<t.",
                "query_similarity": "Cosine similarity between dense query embeddings produced by flashrag.retriever.encoder.Encoder using text_retriever_config.",
                "utility_mapping": UTILITY_TO_SCORE,
                "step_score": "S_t = u_t * delta_f_t",
                "trajectory_quality_score": "TQS = average_t S_t within each trajectory",
            },
        },
        "aggregate": {
            "mean_tqs": mean(item["trajectory_quality_score"] for item in results) if results else 0.0,
            "mean_iterations": mean(item["num_iterations"] for item in results) if results else 0.0,
            "mean_delta_f": mean(step["delta_f"] for step in steps) if steps else 0.0,
            "mean_step_score": mean(step["step_score"] for step in steps) if steps else 0.0,
            "informative_step_rate": (sum(1 for step in steps if step["delta_f"] > 0) / len(steps)) if steps else 0.0,
            "utility_label_counts": label_counts,
        },
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log or (args.output_dir / "evaluate_trajectory_quality.log")
    log_file, original_stdout, original_stderr = setup_logging(log_path)
    try:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise ValueError(f"Set {args.api_key_env} before running.")
        if not args.input.exists():
            raise FileNotFoundError(args.input)

        sample_output = args.output_dir / "trajectory_quality_samples.jsonl"
        summary_output = args.output_dir / "trajectory_quality_summary.json"
        fact_cache_path = args.output_dir / "fact_cache.json"
        utility_cache_path = args.output_dir / "utility_cache.json"

        fact_prompt = load_prompt(args.fact_prompt)
        utility_prompt = load_prompt(args.utility_prompt)
        samples = read_jsonl(args.input)
        if args.max_samples is not None:
            samples = samples[: args.max_samples]
        retriever_config_path = resolve_retriever_config_path(args)
        dense_similarity = DenseSimilarity(
            config_path=retriever_config_path,
            batch_size=args.embed_batch_size,
        )

        existing_results_by_id = {}
        if sample_output.exists() and not args.overwrite:
            existing_results = read_jsonl(sample_output)
            existing_results_by_id = {
                build_sample_id(item): item for item in existing_results
            }
            print(
                f"Resuming from existing output: {sample_output} "
                f"({len(existing_results_by_id)} completed)",
                flush=True,
            )
        elif args.overwrite and sample_output.exists():
            print(f"Overwriting existing output: {sample_output}", flush=True)

        client = OpenAI(api_key=api_key, base_url=args.base_url)
        judge = LLMJudge(
            client=client,
            model=args.model,
            timeout=args.timeout,
            retries=args.retries,
            fact_prompt=fact_prompt,
            utility_prompt=utility_prompt,
            fact_cache_path=fact_cache_path,
            utility_cache_path=utility_cache_path,
        )

        results = [None] * len(samples)
        pending = []
        completed = 0
        for idx, sample in enumerate(samples):
            sample_id = build_sample_id(sample)
            if sample_id in existing_results_by_id:
                results[idx] = existing_results_by_id[sample_id]
                completed += 1
            else:
                pending.append((idx, sample))

        print(f"Evaluating {len(samples)} samples from {args.input}", flush=True)
        print(f"\r{render_progress(completed, len(samples))}", end="", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    evaluate_sample,
                    sample,
                    judge,
                    args.sim_threshold,
                    dense_similarity,
                ): idx
                for idx, sample in pending
            }
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
                completed += 1
                print(f"\r{render_progress(completed, len(samples))}", end="", flush=True)
        print("", flush=True)

        final_results = [item for item in results if item is not None]
        write_jsonl(sample_output, final_results)
        write_json(summary_output, summarize(final_results, args))
        print(f"Saved {sample_output}")
        print(f"Saved {summary_output}")
        print(f"Saved {fact_cache_path}")
        print(f"Saved {utility_cache_path}")
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


if __name__ == "__main__":
    main()
