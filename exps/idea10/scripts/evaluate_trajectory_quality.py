#!/usr/bin/env python3
import argparse
import ast
import contextlib
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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from flashrag.config import Config
from flashrag.utils import get_generator
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
    "/home/you/FlashRAG/exps/idea10/data/result/crag_mm_2026_04_16_10_21_api_experiment/omnisearch_trajectories.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/you/FlashRAG/exps/idea10/data/result/crag_mm_2026_04_16_10_21_api_experiment/trajectory_quality_eval"
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
DEFAULT_FACT_EXTRACTOR = "local_qwen" #"api" or "local_qwen"
DEFAULT_LOCAL_FACT_MODEL = "Qwen2.5-7B-Instruct"
DEFAULT_LOCAL_FACT_MODEL_PATH = Path("/mnt/data/you/modelscope/Qwen2.5-7B-Instruct")
DEFAULT_WORKERS = 2
DEFAULT_TIMEOUT = 120.0
DEFAULT_RETRIES = 2
DEFAULT_SIM_THRESHOLD = 0.85
DEFAULT_FACT_SIM_THRESHOLD = 0.9
DEFAULT_EMBED_BATCH_SIZE = 64
DEFAULT_EPSILON = 1e-8
DEFAULT_MAX_DOCS_PER_ITERATION = 1
DEFAULT_CACHE_FLUSH_EVERY = 200

UTILITY_TO_SCORE = {
    "no_contribution": 0.0,
    "partial_contribution": 0.5,
    "full_contribution": 1.0,
}

UTILITY_LABEL_ALIASES = {
    "no contribution": "no_contribution",
    "partial contribution": "partial_contribution",
    "full contribution": "full_contribution",
    "none": "no_contribution",
    "no": "no_contribution",
    "partial": "partial_contribution",
    "full": "full_contribution",
}


class InvalidUtilityLabelError(ValueError):
    pass


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
    parser.add_argument(
        "--fact-extractor",
        type=str,
        choices=("api", "local_qwen"),
        default=DEFAULT_FACT_EXTRACTOR,
    )
    parser.add_argument("--local-fact-model", type=str, default=DEFAULT_LOCAL_FACT_MODEL)
    parser.add_argument("--local-fact-model-path", type=Path, default=DEFAULT_LOCAL_FACT_MODEL_PATH)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--sim-threshold", type=float, default=DEFAULT_SIM_THRESHOLD)
    parser.add_argument("--fact-sim-threshold", type=float, default=DEFAULT_FACT_SIM_THRESHOLD)
    parser.add_argument("--embed-batch-size", type=int, default=DEFAULT_EMBED_BATCH_SIZE)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--max-docs-per-iteration", type=int, default=DEFAULT_MAX_DOCS_PER_ITERATION)
    parser.add_argument("--cache-flush-every", type=int, default=DEFAULT_CACHE_FLUSH_EVERY)
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


def append_jsonl(path: Path, item: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


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
    return template.format_map(values)


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
    payload = text[start : end + 1]
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        print(
            "[JSON Parse Error] Failed to parse model output as strict JSON. "
            f"Raw payload:\n{payload}",
            flush=True,
        )
        try:
            parsed = ast.literal_eval(payload)
        except Exception:
            raise ValueError(
                f"Invalid JSON object in model output: {payload!r}"
            ) from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"Parsed object is not a dict: {parsed!r}") from exc
        return parsed


def normalize_fact(text: str) -> str:
    text = normalize_whitespace(text).rstrip(".")
    return text.lower()


def normalize_utility_label(label: object) -> str:
    if label is None:
        return ""
    if not isinstance(label, str):
        label = str(label)
    normalized = normalize_whitespace(label).lower().replace("-", "_")
    normalized = re.sub(r"\s+", "_", normalized)
    if normalized in UTILITY_TO_SCORE:
        return normalized
    return UTILITY_LABEL_ALIASES.get(normalized, "")


def split_evidence_into_docs(evidence: str) -> list[str]:
    text = evidence or ""
    if not text.strip():
        return []
    matches = list(re.finditer(r"(?m)^Doc\d+:\n", text))
    if not matches:
        return [text]
    chunks = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def resolve_retriever_config_path(args: argparse.Namespace) -> Path:
    if args.retriever_config is not None:
        return args.retriever_config
    candidate = args.input.parent / "config.yaml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        "Could not infer retriever config from input path. Pass --retriever-config explicitly."
    )


def resolve_run_config_path(args: argparse.Namespace) -> Path:
    candidate = args.input.parent / "config.yaml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        "Could not infer run config from input path. Pass an input whose parent contains config.yaml."
    )


def build_local_fact_generator(args: argparse.Namespace, run_config_path: Path):
    if not args.local_fact_model_path.exists():
        raise FileNotFoundError(
            f"Local fact model path does not exist: {args.local_fact_model_path}"
        )
    config = Config(
        str(run_config_path),
        config_dict={
            "disable_save": True,
            "generator_model": args.local_fact_model,
            "generator_model_path": str(args.local_fact_model_path),
            "generator_max_input_len": 8192,
            "generator_batch_size": 1,
            "framework": "hf",
            "gpu_id": "0",
            "generation_params": {
                "max_new_tokens": 4096,
                "do_sample": False,
                "temperature": 0.0,
                "top_p": 1.0,
            },
        },
    )
    return get_generator(config)


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


def format_duration(seconds: float) -> str:
    total_seconds = max(int(seconds), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def find_semantic_match(
    fact: str,
    candidates: list[str],
    dense_similarity: DenseSimilarity,
    sim_threshold: float,
) -> tuple[str | None, float | None]:
    fact_key = normalize_fact(fact)
    best_match = None
    best_score = None
    for candidate in candidates:
        candidate_key = normalize_fact(candidate)
        if fact_key == candidate_key:
            return candidate, 1.0
        sim = dense_similarity.similarity(fact, candidate)
        if sim is None:
            continue
        if best_score is None or sim > best_score:
            best_score = sim
            best_match = candidate
        if sim >= sim_threshold:
            return candidate, sim
    return None, best_score


def classify_facts_by_novelty(
    facts: list[str],
    cumulative_facts: list[str],
    dense_similarity: DenseSimilarity,
    fact_sim_threshold: float,
) -> tuple[list[str], list[str], list[dict]]:
    accepted_facts: list[str] = []
    accepted_exact_keys: set[str] = set()
    novel_facts: list[str] = []
    fact_match_rows: list[dict] = []

    for fact in facts:
        fact_key = normalize_fact(fact)
        if not fact_key or fact_key in accepted_exact_keys:
            continue

        within_step_match, within_step_score = find_semantic_match(
            fact,
            accepted_facts,
            dense_similarity,
            fact_sim_threshold,
        )
        if within_step_match is not None:
            accepted_exact_keys.add(fact_key)
            fact_match_rows.append(
                {
                    "fact": fact,
                    "is_novel": False,
                    "duplicate_scope": "current_iteration",
                    "matched_fact": within_step_match,
                    "matched_similarity": within_step_score,
                }
            )
            continue

        cumulative_match, cumulative_score = find_semantic_match(
            fact,
            cumulative_facts,
            dense_similarity,
            fact_sim_threshold,
        )
        is_novel = cumulative_match is None
        accepted_facts.append(fact)
        accepted_exact_keys.add(fact_key)
        if is_novel:
            novel_facts.append(fact)
        fact_match_rows.append(
            {
                "fact": fact,
                "is_novel": is_novel,
                "duplicate_scope": None if is_novel else "cumulative_history",
                "matched_fact": cumulative_match,
                "matched_similarity": cumulative_score,
            }
        )

    return accepted_facts, novel_facts, fact_match_rows


class ProgressTracker:
    def __init__(self, total: int) -> None:
        self.total = total
        self.completed = 0
        self.start_time = time.time()
        self.lock = threading.Lock()
        self.print_lock = threading.Lock()
        self.last_status_width = 0
        self.sample_states: dict[str, dict[str, int | str | None]] = {}
        self.latest_metrics: dict[str, dict[str, float | str | None]] = {}

    def resume_completed(self, completed: int) -> None:
        with self.lock:
            self.completed = completed

    def set_stage(
        self,
        sample_id: str,
        stage: str,
        iteration_index: int | None = None,
        total_iterations: int | None = None,
    ) -> None:
        with self.lock:
            self.sample_states[sample_id] = {
                "stage": stage,
                "iteration_index": iteration_index,
                "total_iterations": total_iterations,
            }

    def mark_completed(self, sample_id: str) -> None:
        with self.lock:
            self.completed += 1
            self.sample_states.pop(sample_id, None)

    def set_metric(
        self,
        sample_id: str,
        metric: str,
        value: float | str | None,
        iteration_index: int | None = None,
    ) -> None:
        with self.lock:
            metric_state = self.latest_metrics.setdefault(sample_id, {})
            metric_state[metric] = value
            metric_state["iteration_index"] = iteration_index

    def _build_status_line(self) -> str:
        with self.lock:
            completed = self.completed
            active_states = list(self.sample_states.items())
            latest_metrics = dict(self.latest_metrics)
            start_time = self.start_time

        details = []
        for sample_id, state in active_states[:2]:
            stage = state.get("stage") or "pending"
            iteration_index = state.get("iteration_index")
            total_iterations = state.get("total_iterations")
            iter_text = ""
            if iteration_index is not None and total_iterations is not None:
                iter_text = f" {iteration_index}/{total_iterations}"
            metric_state = latest_metrics.get(sample_id, {})
            cos_value = metric_state.get("query_similarity")
            delta_value = metric_state.get("delta_f")
            utility_value = metric_state.get("utility")

            metric_parts = []
            if isinstance(cos_value, float):
                metric_parts.append(f"cos={cos_value:.4f}")
            elif cos_value == "n/a":
                metric_parts.append("cos=n/a")
            if isinstance(delta_value, float):
                metric_parts.append(f"deltaF={delta_value:.4f}")
            if isinstance(utility_value, float):
                metric_parts.append(f"utility={utility_value:.2f}")

            metric_text = f" [{' '.join(metric_parts)}]" if metric_parts else ""
            details.append(f"{sample_id[:8]}:{stage}{iter_text}{metric_text}")

        elapsed_seconds = max(time.time() - start_time, 0.0)
        if completed > 0 and elapsed_seconds > 0:
            samples_per_min = completed / elapsed_seconds * 60.0
            remaining = max(self.total - completed, 0)
            eta_seconds = (remaining / completed) * elapsed_seconds
            eta_text = (
                f" | elapsed {format_duration(elapsed_seconds)}"
                f" | eta {format_duration(eta_seconds)}"
                f" | {samples_per_min:.2f} sample/min"
            )
        else:
            eta_text = f" | elapsed {format_duration(elapsed_seconds)} | eta --:--:--"

        detail_text = f" | running {', '.join(details)}" if details else ""
        return f"{render_progress(completed, self.total)}{eta_text}{detail_text}"

    def refresh(self) -> None:
        status = self._build_status_line()
        with self.print_lock:
            width = max(self.last_status_width, len(status))
            padded = status.ljust(width)
            print(f"\r{padded}", end="", flush=True)
            self.last_status_width = len(status)

    def println(self, message: str) -> None:
        with self.print_lock:
            clear_width = self.last_status_width
            if clear_width:
                print(f"\r{' ' * clear_width}\r", end="", flush=True)
            print(message, flush=True)
            self.last_status_width = 0


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
        max_docs_per_iteration: int,
        fact_extractor: str,
        local_fact_generator,
        fact_prompt: str,
        utility_prompt: str,
        fact_cache_path: Path,
        utility_cache_path: Path,
        cache_flush_every: int,
    ) -> None:
        self.client = client
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.max_docs_per_iteration = max_docs_per_iteration
        self.fact_extractor = fact_extractor
        self.local_fact_generator = local_fact_generator
        self.local_fact_generator_lock = threading.Lock()
        self.fact_prompt = fact_prompt
        self.utility_prompt = utility_prompt
        self.fact_cache_path = fact_cache_path
        self.utility_cache_path = utility_cache_path
        self.cache_flush_every = max(int(cache_flush_every), 1)
        self.fact_cache_lock = threading.Lock()
        self.utility_cache_lock = threading.Lock()
        self.fact_cache = self._load_cache(fact_cache_path)
        self.utility_cache = self._load_cache(utility_cache_path)
        self.fact_cache_pending_writes = 0
        self.utility_cache_pending_writes = 0

    def _load_cache(self, path: Path) -> dict[str, dict]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _save_cache(self, path: Path, lock: threading.Lock, cache: dict[str, dict]) -> None:
        with lock:
            write_json(path, dict(cache))

    def _maybe_flush_fact_cache(self, force: bool = False) -> None:
        with self.fact_cache_lock:
            if not force and self.fact_cache_pending_writes < self.cache_flush_every:
                return
            write_json(self.fact_cache_path, dict(self.fact_cache))
            self.fact_cache_pending_writes = 0

    def _maybe_flush_utility_cache(self, force: bool = False) -> None:
        with self.utility_cache_lock:
            if not force and self.utility_cache_pending_writes < self.cache_flush_every:
                return
            write_json(self.utility_cache_path, dict(self.utility_cache))
            self.utility_cache_pending_writes = 0

    def flush_caches(self) -> None:
        self._maybe_flush_fact_cache(force=True)
        self._maybe_flush_utility_cache(force=True)

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

    def _local_generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        if self.local_fact_generator is None:
            raise RuntimeError("Local fact generator is not initialized.")
        last_error = None
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        for attempt in range(self.retries + 1):
            try:
                with self.local_fact_generator_lock:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        response = self.local_fact_generator.generate([messages], max_new_tokens=1024)
                if isinstance(response, list):
                    content = response[0]
                else:
                    content = response
                return extract_json_block(content or "")
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(2**attempt, 4))
        raise RuntimeError(f"Local fact extraction request failed: {last_error}") from last_error

    def _extract_facts_from_single_evidence(self, query: str, evidence: str) -> list[str]:
        user_prompt = render_prompt(self.fact_prompt, query=query, evidence=evidence)
        if self.fact_extractor == "local_qwen":
            data = self._local_generate_json(
                "You extract atomic fact sets for trajectory-level evaluation.",
                user_prompt,
            )
        else:
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
        return facts

    def extract_facts(self, query: str, evidence: str, retrieval_type: str) -> list[str]:
        cache_key = json.dumps(
            {"query": query, "evidence": evidence, "retrieval_type": retrieval_type},
            ensure_ascii=False,
            sort_keys=True,
        )
        with self.fact_cache_lock:
            cached = self.fact_cache.get(cache_key)
        if cached is not None:
            return list(cached["facts"])

        evidence_chunks = split_evidence_into_docs(evidence)
        if not evidence_chunks:
            evidence_chunks = [evidence]
        if self.max_docs_per_iteration > 0:
            evidence_chunks = evidence_chunks[: self.max_docs_per_iteration]

        facts = []
        seen = set()
        for evidence_chunk in evidence_chunks:
            chunk_facts = self._extract_facts_from_single_evidence(query=query, evidence=evidence_chunk)
            for fact in chunk_facts:
                key = normalize_fact(fact)
                if not key or key in seen:
                    continue
                seen.add(key)
                facts.append(fact)

        with self.fact_cache_lock:
            self.fact_cache[cache_key] = {
                "retrieval_type": retrieval_type,
                "num_evidence_chunks": len(evidence_chunks),
                "facts": facts,
            }
            self.fact_cache_pending_writes += 1
        self._maybe_flush_fact_cache()
        return facts

    def classify_utility(
        self,
        query: str,
        reasoning: str,
        facts: list[str],
    ) -> tuple[str, str, list[str]]:
        cache_key = json.dumps(
            {"query": query, "reasoning": reasoning, "facts": facts},
            ensure_ascii=False,
            sort_keys=True,
        )
        with self.utility_cache_lock:
            cached = self.utility_cache.get(cache_key)
        if cached is not None:
            cached_label = normalize_utility_label(cached.get("label"))
            cached_reason = cached.get("reason", "")
            cached_filtered_facts = cached.get("filtered_facts", [])
            if not isinstance(cached_filtered_facts, list):
                cached_filtered_facts = []
            cached_filtered_facts = [
                normalize_whitespace(str(fact))
                for fact in cached_filtered_facts
                if normalize_whitespace(str(fact))
            ]
            if cached_label:
                return cached_label, cached_reason, cached_filtered_facts
            # Drop malformed cached labels and recompute.
            with self.utility_cache_lock:
                self.utility_cache.pop(cache_key, None)
                self.utility_cache_pending_writes += 1
            self._maybe_flush_utility_cache()

        user_prompt = render_prompt(
            self.utility_prompt,
            question=query,
            reasoning=reasoning,
            facts=json.dumps(facts, ensure_ascii=False),
        )
        data = self._chat_json(
            "You label iteration utility for trajectory-level evaluation.",
            user_prompt,
        )
        raw_label = data.get("label")
        label = normalize_utility_label(raw_label)
        reason = data.get("reason", "")
        if not isinstance(reason, str):
            reason = str(reason)
        reason = normalize_whitespace(reason)
        if not label:
            raise InvalidUtilityLabelError(
                f"Unsupported utility label: {raw_label!r}; reason={reason!r}; "
                f"query={query!r}; reasoning={reasoning!r}"
            )
        raw_filtered_facts = data.get("filtered_facts", [])
        if not isinstance(raw_filtered_facts, list):
            raw_filtered_facts = []
        fact_lookup = {normalize_fact(fact): fact for fact in facts}
        filtered_facts = []
        seen_filtered_fact_keys = set()
        for fact in raw_filtered_facts:
            fact_text = normalize_whitespace(str(fact))
            fact_key = normalize_fact(fact_text)
            if not fact_key or fact_key in seen_filtered_fact_keys:
                continue
            original_fact = fact_lookup.get(fact_key)
            if original_fact is None:
                continue
            filtered_facts.append(original_fact)
            seen_filtered_fact_keys.add(fact_key)
        if label == "no_contribution":
            filtered_facts = []

        with self.utility_cache_lock:
            self.utility_cache[cache_key] = {
                "label": label,
                "reason": reason,
                "filtered_facts": filtered_facts,
            }
            self.utility_cache_pending_writes += 1
        self._maybe_flush_utility_cache()
        return label, reason, filtered_facts


def evaluate_sample(
    sample: dict,
    judge: LLMJudge,
    sim_threshold: float,
    fact_sim_threshold: float,
    epsilon: float,
    dense_similarity: DenseSimilarity,
    progress_tracker: ProgressTracker | None = None,
) -> dict:
    sample_id = build_sample_id(sample)
    iterations = build_iterations(sample.get("trajectory", []))
    prev_query = None
    cumulative_facts: list[str] = []
    step_rows = []
    total_iterations = len(iterations)

    for iteration in iterations:
        if iteration["retrieval_action"] == "no_retrieval_result":
            continue
        query = iteration["query"] or sample.get("question", "")
        use_query_similarity = iteration["retrieval_action"] != "image_retrieval_result"
        retrieval_type = (
            "Image Retrieval"
            if iteration["retrieval_action"] == "image_retrieval_result"
            else "Text Retrieval"
        )
        if progress_tracker is not None:
            progress_tracker.set_stage(
                sample_id,
                "cosine_sim",
                iteration["iteration_index"],
                total_iterations,
            )
        query_sim = (
            dense_similarity.similarity(prev_query, query)
            if use_query_similarity and prev_query and query
            else None
        )
        if progress_tracker is not None:
            progress_tracker.set_metric(
                sample_id,
                "query_similarity",
                query_sim if query_sim is not None else "n/a",
                iteration["iteration_index"],
            )
        if use_query_similarity and query_sim is not None and query_sim > sim_threshold:
            facts = []
            filtered_facts = []
            accepted_facts = []
            fact_match_rows = []
            delta_f = 0.0
            novel_facts = []
            delta_reason = "similarity_gate"
            label = "no_contribution"
            label_reason = "Similarity gate triggered because query similarity exceeded the threshold."
            utility = 0.0
        else:
            if progress_tracker is not None:
                progress_tracker.set_stage(
                    sample_id,
                    "fact_extraction",
                    iteration["iteration_index"],
                    total_iterations,
                )
            facts = judge.extract_facts(
                query=query,
                evidence=iteration["retrieval_content"],
                retrieval_type=retrieval_type,
            )

            if progress_tracker is not None:
                progress_tracker.set_stage(
                    sample_id,
                    "utility",
                    iteration["iteration_index"],
                    total_iterations,
                )
            label, label_reason, filtered_facts = judge.classify_utility(
                query=query,
                reasoning=iteration["reaction_text"],
                facts=facts,
            )
            utility = UTILITY_TO_SCORE[label]

            if label == "no_contribution":
                accepted_facts = []
                fact_match_rows = []
                delta_f = 0.0
                novel_facts = []
                delta_reason = "no_contribution"
            else:
                if progress_tracker is not None:
                    progress_tracker.set_stage(
                        sample_id,
                        "delta_f",
                        iteration["iteration_index"],
                        total_iterations,
                    )
                accepted_facts, novel_facts, fact_match_rows = classify_facts_by_novelty(
                    filtered_facts,
                    cumulative_facts,
                    dense_similarity,
                    fact_sim_threshold,
                )
                cumulative_fact_count = len(cumulative_facts) + len(accepted_facts)
                delta_f = len(novel_facts) / (cumulative_fact_count + epsilon)
                delta_reason = (
                    "set_difference" if use_query_similarity else "set_difference_no_query_similarity"
                )
                cumulative_facts.extend(accepted_facts)
        if progress_tracker is not None:
            progress_tracker.set_metric(
                sample_id,
                "delta_f",
                delta_f,
                iteration["iteration_index"],
            )
        if progress_tracker is not None:
            progress_tracker.set_metric(
                sample_id,
                "utility",
                utility,
                iteration["iteration_index"],
            )
        step_score = utility * delta_f

        step_rows.append(
            {
                "iteration_index": iteration["iteration_index"],
                "retrieval_action": iteration["retrieval_action"],
                "retrieval_type": retrieval_type,
                "mode": iteration["mode"],
                "previous_query_for_similarity": prev_query,
                "query": iteration["query"],
                "query_similarity_enabled": use_query_similarity,
                "query_similarity_with_previous": query_sim,
                "similarity_threshold": sim_threshold,
                "fact_similarity_threshold": fact_sim_threshold,
                "retrieval_index": iteration["retrieval_index"],
                "reaction_index": iteration["reaction_index"],
                "reaction_action": iteration["reaction_action"],
                "reaction_text": iteration["reaction_text"],
                "facts_t": facts,
                "filtered_facts_t": filtered_facts,
                "accepted_facts_t": accepted_facts,
                "delta_f": delta_f,
                "novel_facts": novel_facts,
                "fact_novelty_matches": fact_match_rows,
                "cumulative_fact_count_after_dedup": len(cumulative_facts),
                "delta_reason": delta_reason,
                "utility_label": label,
                "utility": utility,
                "utility_reason": label_reason,
                "step_score": step_score,
            }
        )
        if use_query_similarity:
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
            "fact_extractor": args.fact_extractor,
            "local_fact_model": args.local_fact_model,
            "local_fact_model_path": str(args.local_fact_model_path),
            "model": args.model,
            "similarity_threshold": args.sim_threshold,
            "fact_similarity_threshold": args.fact_sim_threshold,
            "epsilon": args.epsilon,
            "max_docs_per_iteration": args.max_docs_per_iteration,
            "num_samples": len(results),
            "num_steps": len(steps),
            "definitions": {
                "delta_f_t": "If cosine_similarity(q_t, q_t-1) > tau, set delta_f_t=0. Otherwise extract facts, use the utility prompt to predict utility and filtered_facts, and compute delta_f_t from the semantically novel filtered facts in F_<=t \\ F_<t divided by (|F_<=t| + epsilon).",
                "query_similarity": "Cosine similarity between dense query embeddings produced by flashrag.retriever.encoder.Encoder using text_retriever_config.",
                "fact_similarity": "Atomic facts are treated as duplicates if their normalized strings match exactly or their dense embedding cosine similarity is >= fact_similarity_threshold.",
                "utility_mapping": UTILITY_TO_SCORE,
                "utility_filtering": "The utility prompt first predicts utility_label and then filters facts based on the model's Next Reasoning Text. If utility_label is no_contribution, filtered_facts is empty and u_t=0.",
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
        run_config_path = resolve_run_config_path(args)
        dense_similarity = DenseSimilarity(
            config_path=retriever_config_path,
            batch_size=args.embed_batch_size,
        )
        local_fact_generator = None
        if args.fact_extractor == "local_qwen":
            print(
                f"Loading local fact extractor via FlashRAG generator: {args.local_fact_model_path}",
                flush=True,
            )
            local_fact_generator = build_local_fact_generator(args, run_config_path)
            print("Finished loading local fact extractor.", flush=True)

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
            sample_output.unlink()

        client = OpenAI(api_key=api_key, base_url=args.base_url)
        judge = LLMJudge(
            client=client,
            model=args.model,
            timeout=args.timeout,
            retries=args.retries,
            max_docs_per_iteration=args.max_docs_per_iteration,
            fact_extractor=args.fact_extractor,
            local_fact_generator=local_fact_generator,
            fact_prompt=fact_prompt,
            utility_prompt=utility_prompt,
            fact_cache_path=fact_cache_path,
            utility_cache_path=utility_cache_path,
            cache_flush_every=args.cache_flush_every,
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
        progress_tracker = ProgressTracker(total=len(samples))
        progress_tracker.resume_completed(completed)

        stop_event = threading.Event()

        def progress_monitor() -> None:
            while not stop_event.is_set():
                progress_tracker.refresh()
                time.sleep(0.5)
            progress_tracker.refresh()

        monitor_thread = threading.Thread(target=progress_monitor, daemon=True)
        monitor_thread.start()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    evaluate_sample,
                    sample,
                    judge,
                    args.sim_threshold,
                    args.fact_sim_threshold,
                    args.epsilon,
                    dense_similarity,
                    progress_tracker,
                ): idx
                for idx, sample in pending
            }
            sample_retry_counts: dict[int, int] = {idx: 0 for idx, _ in pending}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                except InvalidUtilityLabelError as exc:
                    sample_retry_counts[idx] += 1
                    retry_count = sample_retry_counts[idx]
                    sample_id = build_sample_id(samples[idx])
                    progress_tracker.println(
                        f"[Retry] id={sample_id} invalid utility label, retrying sample "
                        f"({retry_count}/{args.retries}): {exc}"
                    )
                    if retry_count <= args.retries:
                        new_future = executor.submit(
                            evaluate_sample,
                            samples[idx],
                            judge,
                            args.sim_threshold,
                            args.fact_sim_threshold,
                            args.epsilon,
                            dense_similarity,
                            progress_tracker,
                        )
                        futures[new_future] = idx
                        continue
                    raise
                results[idx] = result
                append_jsonl(sample_output, result)
                completed += 1
                progress_tracker.mark_completed(build_sample_id(samples[idx]))

                partial_results = [item for item in results if item is not None]
                partial_mean_tqs = (
                    mean(item["trajectory_quality_score"] for item in partial_results)
                    if partial_results
                    else 0.0
                )
                progress_tracker.println(
                    "[Stage Result] "
                    f"id={result['id']} "
                    f"iters={result['num_iterations']} "
                    f"total_delta_f={result['total_delta_f']:.4f} "
                    f"tqs={result['trajectory_quality_score']:.4f} "
                    f"partial_mean_tqs={partial_mean_tqs:.4f}"
                )
        stop_event.set()
        monitor_thread.join()
        progress_tracker.println("[Progress] Evaluation completed.")

        final_results = [item for item in results if item is not None]
        judge.flush_caches()
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
