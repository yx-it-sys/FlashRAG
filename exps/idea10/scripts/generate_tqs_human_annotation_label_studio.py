#!/usr/bin/env python3
import html
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


SOURCE_TRAJ = Path(
    "/home/you/FlashRAG/exps/idea10/data/result/crag_mm_2026_03_31_14_06_experiment/omnisearch_trajectories.jsonl"
)
SOURCE_TQS = Path(
    "/home/you/FlashRAG/exps/idea10/data/result/crag_mm_2026_03_31_14_06_experiment/trajectory_quality_eval/trajectory_quality_samples.jsonl"
)
OUTPUT_DIR = Path(
    "/home/you/FlashRAG/exps/idea10/data/result/crag_mm_2026_03_31_14_06_experiment/label/human"
)
OUTPUT_JSON = OUTPUT_DIR / "trajectory_quality_human_annotation_200.json"
OUTPUT_XML = OUTPUT_DIR / "trajectory_quality_human_annotation_config.xml"
OUTPUT_MANIFEST = OUTPUT_DIR / "trajectory_quality_human_annotation_200_manifest.json"

SAMPLE_SIZE = 200
MAX_STEPS = 5
SEED = 20260413
RETRIEVAL_CONTENT_CHAR_LIMIT = 2500
REASONING_CHAR_LIMIT = 1400
FINAL_ANSWER_CHAR_LIMIT = 1200


def read_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def assign_rank_bins(items: list[dict]) -> dict[str, str]:
    ordered = sorted(
        [(idx, float(item["trajectory_quality_score"]), item["id"]) for idx, item in enumerate(items)],
        key=lambda x: (x[1], x[2]),
    )
    n = len(ordered)
    bins = {}
    for rank, (_, _, sample_id) in enumerate(ordered):
        if rank < n / 3:
            bins[sample_id] = "low"
        elif rank < 2 * n / 3:
            bins[sample_id] = "medium"
        else:
            bins[sample_id] = "high"
    return bins


def length_bin(num_iterations: int) -> str:
    if num_iterations <= 1:
        return "short"
    if num_iterations <= 3:
        return "medium"
    return "long"


def mode_profile(steps: list[dict]) -> str:
    has_text = any(step.get("mode") == "text_retrieval" for step in steps)
    has_image = any(step.get("mode") == "image_retrieval" for step in steps)
    has_none = any(step.get("mode") == "no_retrieval" for step in steps)
    if has_text and has_image and has_none:
        return "text+image+no_retrieval"
    if has_text and has_image:
        return "text+image"
    if has_text and has_none:
        return "text+no_retrieval"
    if has_image and has_none:
        return "image+no_retrieval"
    if has_text:
        return "text_only"
    if has_image:
        return "image_only"
    if has_none:
        return "no_retrieval_only"
    return "other"


def allocate_counts(grouped: dict[tuple, list[dict]], target_total: int) -> dict[tuple, int]:
    keys = [key for key, items in grouped.items() if items]
    total = sum(len(grouped[key]) for key in keys)
    raw = {key: len(grouped[key]) * target_total / total for key in keys}
    alloc = {key: min(len(grouped[key]), int(raw[key])) for key in keys}

    for key in keys:
        if alloc[key] == 0:
            alloc[key] = 1

    current = sum(alloc.values())
    if current > target_total:
        reducible = sorted(
            keys,
            key=lambda key: (alloc[key] - raw[key], alloc[key] > 1),
            reverse=True,
        )
        idx = 0
        while current > target_total and reducible:
            key = reducible[idx % len(reducible)]
            if alloc[key] > 1:
                alloc[key] -= 1
                current -= 1
            idx += 1
    elif current < target_total:
        remainders = sorted(
            keys,
            key=lambda key: (raw[key] - int(raw[key]), len(grouped[key]) - alloc[key]),
            reverse=True,
        )
        idx = 0
        while current < target_total and remainders:
            key = remainders[idx % len(remainders)]
            if alloc[key] < len(grouped[key]):
                alloc[key] += 1
                current += 1
            idx += 1
    return alloc


def trim_text(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 15].rstrip() + "\n...[truncated]"


def plain_text(text: str) -> str:
    return (text or "").replace("\r\n", "\n").strip()


def compress_retrieval_content_to_titles(text: str) -> str:
    text = plain_text(text)
    if not text:
        return ""
    lines = text.split("\n")
    out = []
    seen_titles = set()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if re.match(r"^Doc\d+:\s*$", line):
            title = "[empty]"
            j = i + 1
            while j < len(lines):
                candidate = lines[j].strip()
                if candidate:
                    title = candidate
                    break
                j += 1
            norm_title = title.strip().lower()
            if norm_title not in seen_titles:
                seen_titles.add(norm_title)
                out.append(title)
            i = j + 1
            continue
        i += 1
    if out:
        return "\n".join(out)
    return trim_text(text, RETRIEVAL_CONTENT_CHAR_LIMIT)


def rich_block(title: str, body: str) -> str:
    safe_title = html.escape(title)
    safe_body = html.escape(body or "").replace("\n", "<br/>")
    return f"<h3>{safe_title}</h3><div style='white-space: normal; line-height: 1.35;'>{safe_body}</div>"


def get_neighbor_content(trajectory: list[dict], index: int, expected_action: str) -> str:
    if 0 <= index < len(trajectory):
        step = trajectory[index]
        if step.get("action") == expected_action:
            return plain_text(step.get("content", ""))
    return ""


def build_step_payload(eval_step: dict, source_sample: dict) -> dict:
    trajectory = source_sample.get("trajectory", [])
    retrieval_index = eval_step.get("retrieval_index")
    reaction_index = eval_step.get("reaction_index")

    retrieval_content = ""
    if retrieval_index is not None and 0 <= retrieval_index < len(trajectory):
        retrieval_content = compress_retrieval_content_to_titles(
            trajectory[retrieval_index].get("content", "")
        )
    next_reasoning = ""
    if reaction_index is not None and 0 <= reaction_index < len(trajectory):
        next_reasoning = plain_text(trajectory[reaction_index].get("content", ""))

    query = eval_step.get("query") or ""
    retrieval_type = eval_step.get("retrieval_type") or ""
    iteration_index = eval_step.get("iteration_index")

    text_block = "\n\n".join(
        part
        for part in [
            f"Retrieval Type: {retrieval_type}" if retrieval_type else "Retrieval Type: [none]",
            f"Query: {query}" if query else "Query: [none]",
            (
                "Retrieved Content:\n"
                + trim_text(retrieval_content, RETRIEVAL_CONTENT_CHAR_LIMIT)
                if retrieval_content
                else "Retrieved Content:\n[none]"
            ),
            (
                "Next Reasoning:\n"
                + trim_text(next_reasoning, REASONING_CHAR_LIMIT)
                if next_reasoning
                else "Next Reasoning:\n[none]"
            ),
        ]
        if part
    )

    html_block = "".join(
        [
            rich_block(
                "Iteration Metadata",
                "\n".join(
                    [
                        f"Retrieval Type: {retrieval_type or '[none]'}",
                        f"Query: {query or '[none]'}",
                    ]
                ),
            ),
            rich_block("Retrieved Content", trim_text(retrieval_content, RETRIEVAL_CONTENT_CHAR_LIMIT) or "[none]"),
            rich_block("Next Reasoning", trim_text(next_reasoning, REASONING_CHAR_LIMIT) or "[none]"),
        ]
    )
    return {"text": text_block, "html": html_block}


def build_xml(max_steps: int) -> str:
    lines = [
        "<View>",
        '  <Header value="Original Question" />',
        '  <Text name="question_text" value="$question" />',
        "",
    ]

    for idx in range(1, max_steps + 1):
        block = [
            f'  <Header value="Step-Level Annotation: Iteration {idx}" />',
            f'  <HyperText name="step_html_{idx}" value="$step_html_{idx}" />',
        ]
        if idx == 1:
            block.extend(
                [
                    '  <Text name="step_first_iteration_note" value="Iteration 1 is treated as Clear new information and Not redundant by definition. Only annotate usefulness and optional comments." />',
                ]
            )
        else:
            block.extend(
                [
                    f'  <Text name="step_new_information_prompt_{idx}" value="Question: Does this iteration add new information compared with earlier iterations?" />',
                    f'  <Choices name="step_new_information_{idx}" toName="step_html_{idx}" choice="single" showInline="true">',
                    '    <Choice value="No new information" />',
                    '    <Choice value="Partial new information" />',
                    '    <Choice value="Clear new information" />',
                    "  </Choices>",
                    f'  <Text name="step_redundant_prompt_{idx}" value="Question: Is this iteration redundant with earlier iterations?" />',
                    f'  <Choices name="step_redundant_{idx}" toName="step_html_{idx}" choice="single" showInline="true">',
                    '    <Choice value="Not redundant" />',
                    '    <Choice value="Redundant" />',
                    "  </Choices>",
                ]
            )
        block.extend(
            [
                f'  <Text name="step_usefulness_prompt_{idx}" value="Question: Based on the model feedback after retrieval, how useful is this iteration?" />',
                f'  <Choices name="step_usefulness_{idx}" toName="step_html_{idx}" choice="single" showInline="true">',
                '    <Choice value="Not useful" />',
                '    <Choice value="Partly useful" />',
                '    <Choice value="Clearly useful" />',
                "  </Choices>",
                f'  <TextArea name="step_comment_{idx}" toName="step_html_{idx}" placeholder="Optional notes for iteration {idx}." editable="true" maxSubmissions="1" />',
                "",
            ]
        )
        lines.extend(block)

    lines.extend(
        [
            '  <Header value="Trajectory-Level Annotation" />',
            '  <Header value="Stratum" />',
            '  <Text name="trajectory_level_stratum_text" value="$sampling_stratum" />',
            '  <Header value="Final Answer" />',
            '  <Text name="final_answer_text" value="$final_answer" />',
            '  <Header value="Annotation Question" />',
            '  <Text name="trajectory_quality_prompt" value="Overall, how would you rate the quality of this trajectory?" />',
            '  <Choices name="trajectory_quality" toName="question_text" choice="single" showInline="true">',
            '    <Choice value="Low: mostly unhelpful trajectory" />',
            '    <Choice value="Medium: mixed progress" />',
            '    <Choice value="High: consistently useful trajectory" />',
            "  </Choices>",
            '  <Text name="meaningful_progress_prompt" value="Does this trajectory make meaningful progress toward answering the question?" />',
            '  <Choices name="meaningful_progress" toName="question_text" choice="single" showInline="true">',
            '    <Choice value="No meaningful progress" />',
            '    <Choice value="Partial progress" />',
            '    <Choice value="Clear progress" />',
            "  </Choices>",
            '  <Text name="trajectory_efficiency_prompt" value="How efficient is this trajectory?" />',
            '  <Choices name="trajectory_efficiency" toName="question_text" choice="single" showInline="true">',
            '    <Choice value="Inefficient: many wasted steps" />',
            '    <Choice value="Mixed efficiency" />',
            '    <Choice value="Efficient: little wasted effort" />',
            "  </Choices>",
            '  <Text name="wasted_steps_prompt" value="How many obviously wasted steps does this trajectory contain?" />',
            '  <Choices name="wasted_steps" toName="question_text" choice="single" showInline="true">',
            '    <Choice value="No obvious wasted steps" />',
            '    <Choice value="Some wasted steps" />',
            '    <Choice value="Many wasted steps" />',
            "  </Choices>",
            '  <TextArea name="trajectory_comment" toName="question_text" placeholder="Overall notes on trajectory quality, efficiency, and failure mode." editable="true" maxSubmissions="1" />',
            "",
        ]
    )

    lines.append("</View>")
    return "\n".join(lines) + "\n"


def main() -> None:
    random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source_samples = {item["id"]: item for item in read_jsonl(SOURCE_TRAJ)}
    tqs_samples = read_jsonl(SOURCE_TQS)
    rank_bins = assign_rank_bins(tqs_samples)

    enriched = []
    for item in tqs_samples:
        source = source_samples[item["id"]]
        steps = item.get("steps", [])
        enriched.append(
            {
                "id": item["id"],
                "question": source.get("question", ""),
                "final_answer": trim_text(plain_text(source.get("final_answer", "")), FINAL_ANSWER_CHAR_LIMIT),
                "status": item.get("status", ""),
                "num_iterations": int(item.get("num_iterations", 0)),
                "tqs": float(item.get("trajectory_quality_score", 0.0)),
                "tqs_bin": rank_bins[item["id"]],
                "length_bin": length_bin(int(item.get("num_iterations", 0))),
                "mode_profile": mode_profile(steps),
                "sampling_stratum": None,
                "steps": steps,
                "source": source,
            }
        )

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for item in enriched:
        key = (item["status"], item["tqs_bin"], item["length_bin"])
        item["sampling_stratum"] = "|".join(key)
        grouped[key].append(item)

    allocations = allocate_counts(grouped, SAMPLE_SIZE)
    selected = []
    for key, items in grouped.items():
        items = list(items)
        random.shuffle(items)
        selected.extend(items[: allocations[key]])

    if len(selected) != SAMPLE_SIZE:
        raise RuntimeError(f"Expected {SAMPLE_SIZE} samples, got {len(selected)}")

    selected.sort(key=lambda item: (item["sampling_stratum"], item["id"]))

    tasks = []
    for task_id, item in enumerate(selected, start=1):
        task = {
            "id": task_id,
            "data": {
                "sample_id": item["id"],
                "question": item["question"],
                "final_answer": item["final_answer"] or "[empty]",
                "sampling_stratum": item["sampling_stratum"],
                "mode_profile": item["mode_profile"],
                "num_iterations": item["num_iterations"],
            },
        }
        for idx in range(1, MAX_STEPS + 1):
            if idx <= len(item["steps"]):
                payload = build_step_payload(item["steps"][idx - 1], item["source"])
                task["data"][f"step_text_{idx}"] = payload["text"]
                task["data"][f"step_html_{idx}"] = payload["html"]
            else:
                task["data"][f"step_text_{idx}"] = f"Iteration {idx}: [not applicable]"
                task["data"][f"step_html_{idx}"] = rich_block(
                    f"Iteration {idx}", "[not applicable]"
                )
        tasks.append(task)

    manifest = {
        "sample_size": SAMPLE_SIZE,
        "seed": SEED,
        "source_trajectory_file": str(SOURCE_TRAJ),
        "source_tqs_file": str(SOURCE_TQS),
        "tqs_binning": "rank-based terciles over sorted trajectory_quality_score values",
        "selected_status_counts": dict(Counter(item["status"] for item in selected)),
        "selected_tqs_bin_counts": dict(Counter(item["tqs_bin"] for item in selected)),
        "selected_length_bin_counts": dict(Counter(item["length_bin"] for item in selected)),
        "selected_mode_profile_counts": dict(Counter(item["mode_profile"] for item in selected)),
        "selected_stratum_counts": dict(Counter(item["sampling_stratum"] for item in selected)),
        "selected_sample_ids": [item["id"] for item in selected],
    }

    OUTPUT_JSON.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUTPUT_XML.write_text(build_xml(MAX_STEPS), encoding="utf-8")
    OUTPUT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(tasks)} tasks to {OUTPUT_JSON}")
    print(f"Wrote config to {OUTPUT_XML}")
    print(f"Wrote manifest to {OUTPUT_MANIFEST}")


if __name__ == "__main__":
    main()
