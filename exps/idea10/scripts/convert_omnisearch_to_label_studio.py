#!/usr/bin/env python3
import json
from pathlib import Path


SOURCE = Path("/home/you/FlashRAG/exps/idea10/data/result/crag_mm_2026_03_26_21_07_experiment/omnisearch_trajectories.jsonl")
OUTPUT_DIR = Path("/home/you/FlashRAG/exps/idea10/data/result/crag_mm_2026_03_26_21_07_experiment/label-studio")
OUTPUT_JSON = OUTPUT_DIR / "search_action_label_studio_grouped.json"
OUTPUT_CONFIG = OUTPUT_DIR / "search_action_label_config.xml"

TARGET_MODES = {"text_retrieval", "image_retrieval"}
MAX_ACTIONS = 5


def xml_block(index: int) -> str:
    return f"""  <Header value="Trajectory {index}" />
  <Text name="action_text_{index}" value="Action: $action_{index}" />
  <Text name="mode_text_{index}" value="Mode: $mode_{index}" />
  <Text name="query_text_{index}" value="$query_{index}" />
  <Choices name="entity_ambiguous_{index}" toName="query_text_{index}" choice="single" showInline="true">
    <Choice value="No" />
    <Choice value="Yes" />
  </Choices>
  <Choices name="ambiguity_level_{index}" toName="query_text_{index}" choice="single" showInline="true">
    <Choice value="A" />
    <Choice value="B" />
    <Choice value="C" />
    <Choice value="D" />
  </Choices>
  <TextArea name="remark_{index}" toName="query_text_{index}" placeholder="Remark for trajectory {index}" editable="true" maxSubmissions="1" />
"""


def build_config(max_actions: int) -> str:
    parts = [
        "<View>",
        '  <Header value="Question" />',
        '  <Text name="sample_id_text" value="Sample ID: $sample_id" />',
        '  <Text name="question_text" value="$question" />',
        '  <Text name="trajectory_total_text" value="Annotated retrieval actions: $trajectory_total" />',
        "",
    ]
    for index in range(1, max_actions + 1):
        parts.append(xml_block(index))
    parts.append("</View>")
    return "\n".join(parts) + "\n"


def build_task(task_id: int, sample: dict) -> dict:
    retrieval_steps = [
        step for step in sample.get("trajectory", [])
        if step.get("mode") in TARGET_MODES and step.get("action")
    ]

    task = {
        "id": task_id,
        "data": {
            "sample_id": sample.get("id", ""),
            "question": sample.get("question", ""),
            "trajectory_total": len(retrieval_steps),
        },
    }

    for index in range(1, MAX_ACTIONS + 1):
        step = retrieval_steps[index - 1] if index <= len(retrieval_steps) else {}
        task["data"][f"action_{index}"] = step.get("action", "")
        task["data"][f"mode_{index}"] = step.get("mode", "")
        task["data"][f"query_{index}"] = step.get("query") or ""

    return task


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = []
    with SOURCE.open("r", encoding="utf-8") as f:
        for task_id, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            tasks.append(build_task(task_id, json.loads(line)))

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)
        f.write("\n")

    OUTPUT_CONFIG.write_text(build_config(MAX_ACTIONS), encoding="utf-8")

    print(f"Wrote {len(tasks)} tasks to {OUTPUT_JSON}")
    print(f"Wrote config to {OUTPUT_CONFIG}")


if __name__ == "__main__":
    main()
