#!/usr/bin/env python3
import argparse
import csv
import gc
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path

from PIL import Image

for k in [
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
]:
    os.environ.pop(k, None)

ROOT = Path("/home/you/FlashRAG")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PROJECT_ROOT = ROOT / "exps" / "idea10"

from flashrag.config import Config
from flashrag.dataset.dataset import Dataset
from flashrag.pipeline import OmniSearchPipeline
from flashrag.utils import get_generator


ORACLE_REWRITE_LABEL_REL_PATH = Path(
    "label/deepseek/omnisearch_trajectories.entity_ambiguity_labeled.rewrite.jsonl"
)
MULTI_SOURCE_OUTPUT_DIR_NAME = "first_round_oracle_rewrite"
SLEEP_AFTER_CLEANUP_SECONDS = 3
PKILL_PATTERNS = [
    "vllm",
    "api_server",
    "openai.api_server",
]
DEFAULT_MAX_TURNS = 5
TASK_BALANCED_JSONL_PATH = Path(
    "/home/you/FlashRAG/exps/idea10/data/datasets/RefAmb/task_balanced.jsonl"
)


class FrameworkConfig(dict):
    def __init__(self, payload: dict):
        super().__init__(payload)
        self.final_config = self

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay OmniSearch on the subset whose first entity-ambiguous text-retrieval query "
            "is rewritten with the oracle query, while preserving subsequent turns."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        action="append",
        required=True,
        help="Result directory containing config.yaml and oracle rewrite labels. Repeat for multiple sources.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sample-ids-path", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    return parser.parse_args()


def cleanup_gpu_processes() -> None:
    gc.collect()
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as exc:  # noqa: BLE001
        print(f"[Cleanup] torch cleanup skipped: {exc}", flush=True)

    for pattern in PKILL_PATTERNS:
        subprocess.run(
            ["pkill", "-f", pattern],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    time.sleep(SLEEP_AFTER_CLEANUP_SECONDS)


def read_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    return (
        Path("/home/you/FlashRAG/exps/idea10/data/result")
        / f"{timestamp}_first_round_oracle_rewrite_experiment"
    )


def load_id_filter(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(line)
    return ids


def build_framework_config(config: Config) -> FrameworkConfig:
    return FrameworkConfig(deepcopy(config.final_config))


def resolve_data_root(config: FrameworkConfig) -> Path:
    data_dir = Path(config["data_dir"])
    if not data_dir.is_absolute():
        data_dir = (PROJECT_ROOT / data_dir).resolve()
    return data_dir


def sample_id_of(item: dict) -> str:
    for key in ("id", "annotation_id", "data_id"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def iter_labeled_query_steps(item: dict):
    query_steps = item.get("query_steps")
    if isinstance(query_steps, list):
        for ordinal, step in enumerate(query_steps, start=1):
            if not isinstance(step, dict):
                continue
            yield {
                "ordinal": ordinal,
                "step_id": step.get("step_index", ordinal),
                "text_query": step.get("text_query"),
                "rewrite_query": step.get("rewrite_query"),
                "sub_question": step.get("sub_question"),
                "llm_label": step.get("llm_label"),
            }
        return

    trajectory = item.get("trajectory")
    if not isinstance(trajectory, list):
        return

    ordinal = 0
    for step_idx, step in enumerate(trajectory):
        if not isinstance(step, dict):
            continue
        if step.get("action") != "search":
            continue
        text_query = parse_text_retrieval_query(step.get("content", ""))
        if not text_query:
            continue
        ordinal += 1
        yield {
            "ordinal": ordinal,
            "step_id": step_idx + 1,
            "text_query": text_query,
            "rewrite_query": step.get("rewrite_query"),
            "sub_question": "",
            "llm_label": step.get("llm_label"),
        }


def build_rewrite_map(
    rewrite_label_path: Path,
    selected_ids: set[str] | None = None,
) -> tuple[dict[str, dict], dict]:
    rewrite_map: dict[str, dict] = {}
    total = 0
    any_yes = 0
    first_yes = 0
    first_yes_with_rewrite = 0

    for item in read_jsonl(rewrite_label_path):
        total += 1
        annotation_id = sample_id_of(item)
        if not annotation_id:
            continue
        query_steps = list(iter_labeled_query_steps(item))
        any_flag = any(
            (step.get("llm_label") or {}).get("entity_ambiguous") == "Yes"
            for step in query_steps
        )
        if any_flag:
            any_yes += 1

        first_step = next(
            (
                step for step in query_steps
                if (step.get("llm_label") or {}).get("entity_ambiguous") == "Yes"
            ),
            None,
        )
        if not first_step:
            continue
        first_yes += 1

        rewrite_query = str(first_step.get("rewrite_query") or "").strip()
        if not rewrite_query:
            continue
        first_yes_with_rewrite += 1

        if selected_ids is not None and annotation_id not in selected_ids:
            continue

        rewrite_map[annotation_id] = {
            "annotation_id": annotation_id,
            "question": item.get("question"),
            "status": item.get("status"),
            "query_count": item.get("query_count") or len(query_steps),
            "oracle_first_query": rewrite_query,
            "original_labeled_first_query": first_step.get("text_query"),
            "sub_question": first_step.get("sub_question"),
            "rewrite_query_step_index": first_step.get("step_id"),
            "rewrite_query_ordinal": first_step.get("ordinal"),
            "llm_label": deepcopy(first_step.get("llm_label")),
            "answers": deepcopy(item.get("Answers") or item.get("answer")),
        }

    summary = {
        "total_items": total,
        "items_with_any_entity_ambiguity": any_yes,
        "items_with_first_ambiguous_text_query": first_yes,
        "items_with_first_ambiguous_text_query_and_rewrite": first_yes_with_rewrite,
        "selected_after_optional_id_filter": len(rewrite_map),
    }
    return rewrite_map, summary


def load_baseline_maps(result_dir: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    trajectory_map = {item["id"]: item for item in read_jsonl(result_dir / "omnisearch_trajectories.jsonl")}
    intermediate_items = json.loads((result_dir / "intermediate_data.json").read_text(encoding="utf-8"))
    intermediate_map = {}
    for item in intermediate_items:
        sample_id = item.get("id") or item.get("data_id")
        if sample_id:
            intermediate_map[sample_id] = item
    return trajectory_map, intermediate_map


def load_source_subset_rows(config: FrameworkConfig) -> list[dict]:
    subset_path = TASK_BALANCED_JSONL_PATH
    source = str(config.get("source") or "").strip()
    if not source:
        raise ValueError("Config is missing `source`, which is required for source-based subset filtering.")
    if not subset_path.exists():
        raise FileNotFoundError(f"Subset file not found: {subset_path}")

    selected_rows = []
    for row in read_jsonl(subset_path):
        if row.get("source") == source:
            selected_rows.append(row)
    if not selected_rows:
        raise ValueError(f"No items with source `{source}` found in {subset_path}")
    print(
        f"[Dataset] Loaded {len(selected_rows)} rows for source `{source}` from {subset_path}",
        flush=True,
    )
    return selected_rows


def build_subset_dataset(
    full_dataset: Dataset,
    rewrite_map: dict[str, dict],
    max_samples: int | None,
) -> Dataset:
    selected_items = [
        item for item in full_dataset.data
        if item.id in rewrite_map
    ]
    selected_items.sort(key=lambda item: item.id)
    if max_samples is not None:
        selected_items = selected_items[:max_samples]
    return Dataset(config=full_dataset.config, data=selected_items)


def parse_text_retrieval_query(response: str) -> str:
    pattern = r'Text Retrieval[:\s"]*(.*?)(?=<|$)'
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip().strip('"')


def replace_first_text_retrieval_query(response: str, new_query: str) -> str:
    pattern = r'(Text Retrieval[:\s"]*)(.*?)(?=(?:<|$))'
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        return response
    prefix = match.group(1)
    return response[: match.start()] + f"{prefix}{new_query}" + response[match.end() :]


def render_action_node(action: str, content: str) -> str:
    if action == "thought":
        return f"<Thought>\n{content}\n"
    if action == "sub-question":
        return f"<Sub-Question>\n{content}\n"
    if action == "search":
        return f"<Search>\n{content}\n</Search>"
    if action == "final_answer":
        return f"<Final Answer>\n{content}\n</Final Answer>"
    raise ValueError(f"Unsupported trajectory action for assistant reconstruction: {action}")


def extract_final_answer_text(response: str) -> str | None:
    pattern = r"(?:<Final Answer>|Final Answer:)\s*(.*?)(?=<|$)"
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        return None
    return match.group(1).strip().replace("\n", "")


def build_initial_messages(pipeline: OmniSearchPipeline, question: str, image_path: Path) -> tuple[Image.Image, list[dict]]:
    img = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": pipeline.prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Input Question: {question}"},
                {"type": "image", "image": img},
            ],
        },
    ]
    return img, messages


def reconstruct_messages_before_target(
    pipeline: OmniSearchPipeline,
    question: str,
    image_path: Path,
    baseline_trajectory: list[dict],
    target_query_ordinal: int,
) -> tuple[Image.Image, list[dict], str, list[dict]]:
    img, messages = build_initial_messages(pipeline, question, image_path)
    assistant_buffer: list[dict] = []
    prefix_trajectory: list[dict] = []
    text_query_ordinal = 0

    for step in baseline_trajectory:
        if not isinstance(step, dict):
            continue
        action = step.get("action")
        if action in {"thought", "sub-question", "search", "final_answer"}:
            assistant_buffer.append(step)
            if action != "search":
                continue

            search_content = str(step.get("content") or "")
            if not search_content.startswith("Text Retrieval"):
                continue
            text_query_ordinal += 1
            if text_query_ordinal == target_query_ordinal:
                assistant_response = "".join(
                    render_action_node(node["action"], str(node.get("content") or ""))
                    for node in assistant_buffer
                )
                return img, messages, assistant_response, prefix_trajectory
            continue

        if action in {"text_retrieval_result", "image_retrieval_result", "no_retrieval_result"}:
            if assistant_buffer:
                assistant_response = "".join(
                    render_action_node(node["action"], str(node.get("content") or ""))
                    for node in assistant_buffer
                )
                messages.append({"role": "assistant", "content": assistant_response})
                prefix_trajectory.extend(deepcopy(assistant_buffer))
                assistant_buffer = []

            retrieval_mode = str(step.get("mode") or "")
            retrieval_content = step.get("content")
            prefix_trajectory.append(deepcopy(step))
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": pipeline._build_followup_message(
                                retrieval_mode, retrieval_content
                            ),
                        }
                    ],
                }
            )

    raise ValueError(
        f"Failed to reconstruct prefix: target text retrieval ordinal {target_query_ordinal} "
        "was not found in baseline trajectory."
    )


def log_sample_header(item_id: str, question: str) -> None:
    print(f"Question ID: {item_id}")
    print(f"Question: {question}")


def log_generation_response(label: str, response: str) -> None:
    print(f"{label}: {response}")


def log_oracle_rewrite(original_query: str | None, oracle_query: str, applied: bool) -> None:
    print(
        "Oracle Rewrite: "
        f"applied={applied}, original_first_query={original_query!r}, oracle_first_query={oracle_query!r}"
    )


def log_retrieval_step(retrieval_mode: str, query_txt: str) -> None:
    if retrieval_mode == "text_retrieval":
        print("Start Text Retrieval...")
        print(f"Query Text: {query_txt}")
    elif retrieval_mode == "image_retrieval":
        print("Start Image Retrieval...")
        print(f"Image Query: {query_txt}")
    elif retrieval_mode == "no_retrieval":
        print("No Retrieval selected.")


def replay_one_sample(
    pipeline: OmniSearchPipeline,
    item,
    rewrite_meta: dict,
    max_turns: int,
    data_root: Path,
    baseline_trajectory_record: dict,
) -> dict:
    image_path = data_root / pipeline.dataset_name / "images" / f"{item.image_id}.jpg"
    start_time = time.time()
    if not image_path.exists():
        return {
            "id": item.id,
            "question": item.question,
            "pred": "",
            "status": "missing_image",
            "duration_seconds": 0.0,
            "trajectory_record": {
                "question": item.question,
                "id": item.id,
                "final_answer": "",
                "status": "missing_image",
                "duration_seconds": 0.0,
                "trajectory": [],
            },
            "run_meta": {
                "oracle_rewrite_applied": False,
                "oracle_rewrite_attempted": False,
                "error": f"Image not found: {image_path}",
            },
        }

    baseline_trajectory = baseline_trajectory_record.get("trajectory")
    if not isinstance(baseline_trajectory, list):
        return {
            "id": item.id,
            "question": item.question,
            "pred": "",
            "status": "missing_baseline_trajectory",
            "duration_seconds": 0.0,
            "trajectory_record": {
                "question": item.question,
                "id": item.id,
                "final_answer": "",
                "status": "missing_baseline_trajectory",
                "duration_seconds": 0.0,
                "trajectory": [],
            },
            "run_meta": {
                "oracle_rewrite_applied": False,
                "oracle_rewrite_attempted": False,
                "error": f"Missing baseline trajectory for {item.id}",
            },
        }

    trajectory: list[dict] = []
    oracle_rewrite_applied = False
    original_first_query = None
    oracle_first_query = rewrite_meta["oracle_first_query"]
    first_turn_expected_step_index = rewrite_meta["rewrite_query_step_index"]
    target_query_ordinal = int(rewrite_meta.get("rewrite_query_ordinal") or 1)
    log_sample_header(item.id, item.question)

    try:
        img, messages, response, prefix_trajectory = reconstruct_messages_before_target(
            pipeline=pipeline,
            question=item.question,
            image_path=image_path,
            baseline_trajectory=baseline_trajectory,
            target_query_ordinal=target_query_ordinal,
        )
        trajectory = deepcopy(prefix_trajectory)
        response = pipeline._truncate_after_search(response)
        original_first_query = parse_text_retrieval_query(response)
        if original_first_query:
            response = replace_first_text_retrieval_query(response, oracle_first_query)
            oracle_rewrite_applied = True
            log_oracle_rewrite(original_first_query, oracle_first_query, True)
            print(
                f"Rewrite baseline assistant query and retrieval query at targeted ambiguous step: "
                f"generated={original_first_query!r}, retrieval={oracle_first_query!r}"
            )
        log_generation_response("Replayed Target Response", response)
        pipeline._record_response_actions(trajectory, response)
        messages.append({"role": "assistant", "content": response})

        conversation_num = 0
        while conversation_num < max_turns:
            final_answer = extract_final_answer_text(response)
            if final_answer is not None:
                print(f"Final Answer: {final_answer}")
                duration = time.time() - start_time
                return {
                    "id": item.id,
                    "question": item.question,
                    "pred": final_answer,
                    "status": "ok",
                    "duration_seconds": duration,
                    "trajectory_record": {
                        "question": item.question,
                        "id": item.id,
                        "final_answer": final_answer,
                        "status": "ok",
                        "duration_seconds": duration,
                        "trajectory": trajectory,
                    },
                    "run_meta": {
                        "oracle_rewrite_applied": oracle_rewrite_applied,
                        "oracle_rewrite_attempted": True,
                        "original_first_query": original_first_query,
                        "oracle_first_query": oracle_first_query,
                        "rewrite_query_step_index": first_turn_expected_step_index,
                        "rewrite_query_ordinal": target_query_ordinal,
                    },
                }

            need_txt_ret = "Text Retrieval" in response
            need_img_ret = "Image Retrieval" in response
            need_no_ret = "No Retrieval" in response
            if not (need_txt_ret or need_img_ret or need_no_ret):
                break

            retrieval_content = ""
            retrieval_mode = None
            query_txt = ""
            if need_txt_ret:
                retrieval_mode = "text_retrieval"
                query_txt = parse_text_retrieval_query(response)
                log_retrieval_step(retrieval_mode, query_txt)
                retrieved_docs = pipeline._search_text_docs(query_txt)
                retrieval_content = pipeline._format_retrieval_content(retrieved_docs)
                retrieval_content = pipeline._clip_text(
                    retrieval_content, pipeline.retrieval_char_limit
                )
                pipeline._log_retrieval_preview(retrieval_content)
            elif need_img_ret:
                retrieval_mode = "image_retrieval"
                log_retrieval_step(retrieval_mode, query_txt)
                retrieved_docs = pipeline._search_image_docs(img)
                retrieval_content = pipeline._format_retrieval_content(
                    retrieved_docs,
                    preferred_field="title",
                )
                retrieval_content = pipeline._clip_text(
                    retrieval_content, pipeline.retrieval_char_limit
                )
                pipeline._log_retrieval_preview(retrieval_content)
            else:
                retrieval_mode = "no_retrieval"
                retrieval_content = None
                log_retrieval_step(retrieval_mode, query_txt)

            pipeline._record_retrieval_result(
                trajectory=trajectory,
                retrieval_mode=retrieval_mode,
                query_txt=query_txt,
                retrieval_content=retrieval_content,
            )

            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": pipeline._build_followup_message(
                                retrieval_mode, retrieval_content
                            ),
                        }
                    ],
                }
            )

            response = pipeline._generate_text(messages)
            response = pipeline._truncate_after_search(response)
            log_generation_response("Response", response)
            pipeline._record_response_actions(trajectory, response)
            messages.append({"role": "assistant", "content": response})
            conversation_num += 1

        duration = time.time() - start_time
        print(
            f"Warning: reached end of agent loop for item {item.id} "
            "without a 'Final Answer'. returning last response"
        )
        return {
            "id": item.id,
            "question": item.question,
            "pred": response,
            "status": "missing_final_answer",
            "duration_seconds": duration,
            "trajectory_record": {
                "question": item.question,
                "id": item.id,
                "final_answer": response,
                "status": "missing_final_answer",
                "duration_seconds": duration,
                "trajectory": trajectory,
            },
            "run_meta": {
                "oracle_rewrite_applied": oracle_rewrite_applied,
                "oracle_rewrite_attempted": True,
                "original_first_query": original_first_query,
                "oracle_first_query": oracle_first_query,
                "rewrite_query_step_index": first_turn_expected_step_index,
                "rewrite_query_ordinal": target_query_ordinal,
            },
        }
    except Exception as exc:  # noqa: BLE001
        duration = time.time() - start_time
        error_text = f"{exc.__class__.__name__}: {exc}"
        print(f"Inference error, hidden states ignored: {error_text}")
        trajectory.append({"action": "error", "content": error_text})
        return {
            "id": item.id,
            "question": item.question,
            "pred": response if "response" in locals() else "",
            "status": "generation_error",
            "error": error_text,
            "duration_seconds": duration,
            "trajectory_record": {
                "question": item.question,
                "id": item.id,
                "final_answer": response if "response" in locals() else "",
                "status": "generation_error",
                "error": error_text,
                "duration_seconds": duration,
                "trajectory": trajectory,
            },
            "run_meta": {
                "oracle_rewrite_applied": oracle_rewrite_applied,
                "oracle_rewrite_attempted": True,
                "original_first_query": original_first_query,
                "oracle_first_query": oracle_first_query,
                "rewrite_query_step_index": first_turn_expected_step_index,
                "rewrite_query_ordinal": target_query_ordinal,
                "error": f"{type(exc).__name__}: {exc}",
            },
        }


def summarize_metrics(items: list[dict]) -> dict[str, float]:
    if not items:
        return {}
    metric_sums: dict[str, float] = {}
    metric_counts: dict[str, int] = {}
    for item in items:
        for metric_name, score in (item.get("output", {}).get("metric_score") or {}).items():
            metric_sums[metric_name] = metric_sums.get(metric_name, 0.0) + float(score)
            metric_counts[metric_name] = metric_counts.get(metric_name, 0) + 1
    return {
        metric_name: metric_sums[metric_name] / metric_counts[metric_name]
        for metric_name in sorted(metric_sums)
    }


def build_paired_comparison(
    subset_dataset: Dataset,
    baseline_intermediate_map: dict[str, dict],
    baseline_trajectory_map: dict[str, dict],
    run_results: list[dict],
    output_dir: Path,
) -> dict:
    new_intermediate_items = [item.to_dict() for item in subset_dataset]
    new_intermediate_map = {
        (item.get("id") or item.get("data_id")): item for item in new_intermediate_items
    }
    run_meta_map = {item["id"]: item for item in run_results}

    paired_rows = []
    status_shift_counter = Counter()
    rewrite_applied_count = 0

    for sample_id in sorted(new_intermediate_map):
        baseline_item = baseline_intermediate_map[sample_id]
        new_item = new_intermediate_map[sample_id]
        baseline_traj = baseline_trajectory_map.get(sample_id, {})
        new_meta = run_meta_map[sample_id]
        baseline_status = baseline_traj.get("status")
        new_status = new_meta.get("status")
        status_shift_counter[f"{baseline_status} -> {new_status}"] += 1
        if new_meta.get("run_meta", {}).get("oracle_rewrite_applied"):
            rewrite_applied_count += 1

        baseline_metrics = baseline_item.get("output", {}).get("metric_score") or {}
        new_metrics = new_item.get("output", {}).get("metric_score") or {}
        row = {
            "id": sample_id,
            "question": new_item.get("question", ""),
            "baseline_status": baseline_status,
            "new_status": new_status,
            "baseline_pred": baseline_item.get("output", {}).get("pred", ""),
            "new_pred": new_item.get("output", {}).get("pred", ""),
            "original_first_query": new_meta.get("run_meta", {}).get("original_first_query", ""),
            "oracle_first_query": new_meta.get("run_meta", {}).get("oracle_first_query", ""),
            "oracle_rewrite_applied": new_meta.get("run_meta", {}).get("oracle_rewrite_applied", False),
        }
        metric_names = sorted(set(baseline_metrics) | set(new_metrics))
        for metric_name in metric_names:
            baseline_score = baseline_metrics.get(metric_name)
            new_score = new_metrics.get(metric_name)
            row[f"baseline_{metric_name}"] = baseline_score
            row[f"new_{metric_name}"] = new_score
            if baseline_score is None or new_score is None:
                row[f"delta_{metric_name}"] = None
            else:
                row[f"delta_{metric_name}"] = float(new_score) - float(baseline_score)
        paired_rows.append(row)

    baseline_subset_items = [baseline_intermediate_map[item.id] for item in subset_dataset]
    comparison = {
        "subset_size": len(subset_dataset),
        "oracle_rewrite_applied_count": rewrite_applied_count,
        "status_shift_counts": dict(status_shift_counter),
        "baseline_metric_means": summarize_metrics(baseline_subset_items),
        "new_metric_means": summarize_metrics(new_intermediate_items),
    }

    metric_names = sorted(
        {
            key[len("delta_") :]
            for row in paired_rows
            for key in row
            if key.startswith("delta_")
        }
    )
    delta_means = {}
    for metric_name in metric_names:
        values = [
            row[f"delta_{metric_name}"]
            for row in paired_rows
            if row.get(f"delta_{metric_name}") is not None
        ]
        if values:
            delta_means[metric_name] = sum(values) / len(values)
    comparison["mean_metric_delta"] = delta_means

    write_json(output_dir / "paired_comparison.json", comparison)
    fieldnames = list(paired_rows[0].keys()) if paired_rows else ["id"]
    write_csv(output_dir / "paired_comparison.csv", paired_rows, fieldnames)
    return comparison


def run_single_source(args: argparse.Namespace) -> None:
    output_dir = make_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_ids = load_id_filter(args.sample_ids_path)

    config_override = {
        "save_dir": str(output_dir),
        "output_dir": str(output_dir),
        "save_new_dir": False,
        "seed": args.seed,
    }
    raw_config = Config(str(args.config), config_dict=config_override)
    config = build_framework_config(raw_config)
    data_root = resolve_data_root(config)
    baseline_result_dir = Path(args.config).parent
    rewrite_label_path = baseline_result_dir / ORACLE_REWRITE_LABEL_REL_PATH
    if not rewrite_label_path.exists():
        raise FileNotFoundError(f"Missing oracle rewrite label file: {rewrite_label_path}")
    rewrite_map, rewrite_summary = build_rewrite_map(rewrite_label_path, selected_ids)

    source_subset_rows = load_source_subset_rows(config)
    full_dataset = Dataset(config=config, data=source_subset_rows)
    subset_dataset = build_subset_dataset(
        full_dataset,
        rewrite_map,
        args.max_samples,
    )

    subset_ids = {item.id for item in subset_dataset}
    rewrite_map = {sample_id: meta for sample_id, meta in rewrite_map.items() if sample_id in subset_ids}

    baseline_trajectory_map, baseline_intermediate_map = load_baseline_maps(baseline_result_dir)

    generator = get_generator(config)
    pipeline = OmniSearchPipeline(config=config, retriever=None, generator=generator)
    pipeline.trajectory_path = str(output_dir / "omnisearch_trajectories.jsonl")

    selected_subset_records = []
    for item in subset_dataset:
        baseline_traj = baseline_trajectory_map.get(item.id, {})
        baseline_intermediate = baseline_intermediate_map.get(item.id, {})
        rewrite_meta = rewrite_map[item.id]
        selected_subset_records.append(
            {
                "id": item.id,
                "question": item.question,
                "image_id": item.image_id,
                "baseline_status": baseline_traj.get("status"),
                "baseline_pred": baseline_intermediate.get("output", {}).get("pred"),
                **rewrite_meta,
            }
        )
    write_jsonl(output_dir / "selected_subset.jsonl", selected_subset_records)

    predictions = []
    run_results = []
    status_counter = Counter()

    for idx, item in enumerate(subset_dataset, start=1):
        print(f"[{idx}/{len(subset_dataset)}] id={item.id}")
        result = replay_one_sample(
            pipeline=pipeline,
            item=item,
            rewrite_meta=rewrite_map[item.id],
            max_turns=args.max_turns,
            data_root=data_root,
            baseline_trajectory_record=baseline_trajectory_map.get(item.id, {}),
        )
        predictions.append(result["pred"])
        run_results.append(result)
        status_counter[result["status"]] += 1
        trajectory_record = deepcopy(result["trajectory_record"])
        trajectory_record["oracle_rewrite_meta"] = deepcopy(result["run_meta"])
        pipeline.safe_write(pipeline.trajectory_path, pipeline._serialize_for_log(trajectory_record))

    subset_dataset.update_output("pred", predictions)

    if not args.skip_eval:
        pipeline.evaluate(subset_dataset, do_eval=True)
    else:
        subset_dataset.save(str(output_dir / "intermediate_data.json"))

    comparison = build_paired_comparison(
        subset_dataset=subset_dataset,
        baseline_intermediate_map=baseline_intermediate_map,
        baseline_trajectory_map=baseline_trajectory_map,
        run_results=run_results,
        output_dir=output_dir,
    )

    run_summary = {
        "experiment_tag": "first_round_oracle_rewrite",
        "config_path": str(args.config),
        "baseline_result_dir": str(baseline_result_dir),
        "rewrite_label_path": str(rewrite_label_path),
        "output_dir": str(output_dir),
        "max_turns": args.max_turns,
        "skip_eval": args.skip_eval,
        "rewrite_selection_summary": rewrite_summary,
        "actual_run_subset_size": len(subset_dataset),
        "status_counts": dict(status_counter),
        "paired_comparison": comparison,
    }
    write_json(output_dir / "run_summary.json", run_summary)

    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


def _run_multi_source(args: argparse.Namespace) -> None:
    source_dirs = list(args.source_dir)
    if not source_dirs:
        raise ValueError("At least one --source-dir must be provided.")
    total = len(source_dirs)
    for idx, source_dir in enumerate(source_dirs, start=1):
        source_args = deepcopy(args)
        source_args.config = source_dir / "config.yaml"
        source_args.output_dir = source_dir / MULTI_SOURCE_OUTPUT_DIR_NAME

        if not source_args.config.exists():
            raise FileNotFoundError(f"Missing config: {source_args.config}")

        print(f"\n===== [{idx}/{total}] Start Oracle Replay Experiment =====", flush=True)
        print(f"[Run] source_dir={source_dir}", flush=True)
        print(f"[Run] config={source_args.config}", flush=True)
        print(f"[Run] output_dir={source_args.output_dir}", flush=True)
        try:
            run_single_source(source_args)
        finally:
            print("[Cleanup] Releasing vLLM / CUDA-related resources.", flush=True)
            cleanup_gpu_processes()
        print(f"===== [{idx}/{total}] Done =====", flush=True)

    print(f"\nAll {total} oracle replay experiments completed.", flush=True)


def main() -> None:
    args = parse_args()
    _run_multi_source(args)


if __name__ == "__main__":
    main()
