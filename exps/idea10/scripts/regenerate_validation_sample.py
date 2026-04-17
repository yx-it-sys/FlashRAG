from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


BASE_DIR = Path("/home/you/FlashRAG/exps/idea10")
DATASET_DIR = BASE_DIR / "data/datasets/crag_mm"
VALIDATION_PATH = DATASET_DIR / "validation.jsonl"
TRAJECTORY_PATH = (
    BASE_DIR
    / "data/result/crag_mm_2026_03_31_14_06_experiment/omnisearch_trajectories.jsonl"
)
LABEL_PATH = (
    BASE_DIR
    / "data/result/crag_mm_2026_03_31_14_06_experiment/label/qwen/"
    / "trajectory_annotation.llm_labeled.adjudicated.jsonl"
)
OUTPUT_SAMPLE_PATH = DATASET_DIR / "validation_sample.jsonl"
OUTPUT_MANIFEST_PATH = DATASET_DIR / "sample_manifest.csv"

SET_A_NAME = "A_population"
SET_B_NAME = "B_ambiguity_focused"
DEFAULT_SEED = 2024


def load_jsonl(path: Path, key: str) -> dict[str, dict]:
    records = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            records[obj[key]] = obj
    return records


def derive_item_level(levels: list[str]) -> tuple[bool, str]:
    if not levels:
        return False, "No"
    unique_levels = set(levels)
    if len(unique_levels) == 1:
        return True, levels[0]
    return True, "Mixed"


def derive_length_bucket(query_count: int) -> str:
    if query_count <= 1:
        return "short"
    if query_count <= 3:
        return "medium"
    return "long"


def build_records() -> list[dict]:
    validation = load_jsonl(VALIDATION_PATH, "id")
    trajectories = load_jsonl(TRAJECTORY_PATH, "id")
    labels = load_jsonl(LABEL_PATH, "annotation_id")

    records = []
    for item_id, item in validation.items():
        trajectory = trajectories.get(item_id)
        label = labels.get(item_id)
        if trajectory is None or label is None:
            continue

        ambiguous_levels = []
        for step in label.get("query_steps", []):
            llm_label = step.get("llm_label") or {}
            if str(llm_label.get("entity_ambiguous", "")).strip().lower() == "yes":
                ambiguous_levels.append(llm_label.get("ambiguity_level") or "Unknown")

        item_ambiguous, item_level = derive_item_level(ambiguous_levels)
        query_count = int(label.get("query_count", 0))
        num_ambiguous_steps = len(ambiguous_levels)

        metadata = item.get("metadata", {})
        records.append(
            {
                "id": item_id,
                "item": item,
                "status": trajectory.get("status", "unknown"),
                "item_ambiguous": item_ambiguous,
                "item_level": item_level,
                "query_count": query_count,
                "num_ambiguous_steps": num_ambiguous_steps,
                "length_bucket": derive_length_bucket(query_count),
                "domain": metadata.get("domain", ""),
                "query_category": metadata.get("query_category", ""),
                "dynamism": metadata.get("dynamism", ""),
            }
        )
    return records


def allocate_proportional(
    total: int,
    groups: list[tuple[str, list[dict]]],
    preferred: list[str] | None = None,
) -> dict[str, int]:
    if total <= 0:
        return {name: 0 for name, _ in groups}

    allocation = {name: 0 for name, _ in groups}
    capacities = {name: len(rows) for name, rows in groups}
    remaining = total

    order = preferred or [name for name, _ in groups]
    for name in order:
        if remaining <= 0:
            break
        capacity = capacities.get(name, 0)
        if capacity <= allocation.get(name, 0):
            continue
        allocation[name] += 1
        remaining -= 1

    if remaining <= 0:
        return allocation

    weighted_groups = [(name, rows) for name, rows in groups if rows]
    total_available = sum(len(rows) for _, rows in weighted_groups)
    if total_available == 0:
        return allocation

    fractional = []
    for name, rows in weighted_groups:
        capacity_left = capacities[name] - allocation[name]
        if capacity_left <= 0:
            continue
        share = remaining * len(rows) / total_available
        base = min(capacity_left, math.floor(share))
        allocation[name] += base
        fractional.append((share - base, name))

    used = sum(allocation.values())
    leftover = total - used
    if leftover > 0:
        fractional.sort(key=lambda x: (-x[0], x[1]))
        for _, name in fractional:
            if leftover <= 0:
                break
            if allocation[name] < capacities[name]:
                allocation[name] += 1
                leftover -= 1

    if leftover > 0:
        for name, _ in weighted_groups:
            while leftover > 0 and allocation[name] < capacities[name]:
                allocation[name] += 1
                leftover -= 1

    return allocation


def sample_without_replacement(
    rows: list[dict], count: int, rng: random.Random
) -> list[dict]:
    if count <= 0:
        return []
    if count >= len(rows):
        return list(rows)
    indexed = list(rows)
    rng.shuffle(indexed)
    return indexed[:count]


def sample_set_a(records: list[dict], target: int, rng: random.Random) -> list[dict]:
    non_ambiguous = [r for r in records if not r["item_ambiguous"]]
    ambiguous = [r for r in records if r["item_ambiguous"]]
    ambiguity_targets = {"False": round(target * 78 / 120), "True": target - round(target * 78 / 120)}
    selected = []

    for ambiguous_flag, group_rows in [("False", non_ambiguous), ("True", ambiguous)]:
        group_target = min(ambiguity_targets[ambiguous_flag], len(group_rows))
        status_groups = defaultdict(list)
        for row in group_rows:
            status_groups[row["status"]].append(row)

        status_alloc = allocate_proportional(
            group_target,
            [(status, rows) for status, rows in status_groups.items()],
            preferred=["ok", "missing_final_answer", "generation_error"],
        )

        for status, status_target in status_alloc.items():
            candidates = status_groups[status]
            length_groups = defaultdict(list)
            for row in candidates:
                length_groups[row["length_bucket"]].append(row)
            length_alloc = allocate_proportional(
                status_target,
                [(length, rows) for length, rows in length_groups.items()],
                preferred=["short", "medium", "long"],
            )
            for length, length_target in length_alloc.items():
                selected.extend(
                    sample_without_replacement(length_groups[length], length_target, rng)
                )

    # Fill any rounding gap from the remaining pool while preserving ambiguity split priority.
    if len(selected) < target:
        chosen_ids = {row["id"] for row in selected}
        remainder = [row for row in records if row["id"] not in chosen_ids]
        selected.extend(sample_without_replacement(remainder, target - len(selected), rng))

    return selected[:target]


def status_priority(status: str) -> int:
    if status == "missing_final_answer":
        return 0
    if status not in {"ok", "missing_final_answer"}:
        return 1
    return 2


def sample_set_b(
    records: list[dict], target: int, excluded_ids: set[str], rng: random.Random
) -> list[dict]:
    candidates = [
        r for r in records if r["item_ambiguous"] and r["id"] not in excluded_ids
    ]
    by_level = defaultdict(list)
    for row in candidates:
        by_level[row["item_level"]].append(row)

    level_targets = {
        "Object Identification": 70,
        "Indirect Entity Ambiguity": 45,
        "Description": 40,
        "Mixed": 40,
        "No Object Involved": 5,
    }
    capacities = {level: len(rows) for level, rows in by_level.items()}
    if capacities.get("No Object Involved", 0) < level_targets["No Object Involved"]:
        shortfall = level_targets["No Object Involved"] - capacities.get(
            "No Object Involved", 0
        )
        level_targets["No Object Involved"] = capacities.get("No Object Involved", 0)
        level_targets["Mixed"] += shortfall

    selected = []
    for level in [
        "Object Identification",
        "Indirect Entity Ambiguity",
        "Description",
        "Mixed",
        "No Object Involved",
    ]:
        pool = by_level.get(level, [])
        level_target = min(level_targets.get(level, 0), len(pool))
        if level_target <= 0:
            continue

        status_groups = defaultdict(list)
        for row in pool:
            status_groups[row["status"]].append(row)

        preferred_status_order = [
            "missing_final_answer",
            *sorted(
                [s for s in status_groups if s not in {"missing_final_answer", "ok"}],
                key=status_priority,
            ),
            "ok",
        ]
        status_alloc = allocate_proportional(
            level_target,
            [(status, rows) for status, rows in status_groups.items()],
            preferred=preferred_status_order,
        )

        for status in preferred_status_order:
            if status not in status_groups:
                continue
            status_target = status_alloc.get(status, 0)
            if status_target <= 0:
                continue
            length_groups = defaultdict(list)
            for row in status_groups[status]:
                length_groups[row["length_bucket"]].append(row)
            length_alloc = allocate_proportional(
                status_target,
                [(length, rows) for length, rows in length_groups.items()],
                preferred=["long", "short", "medium"],
            )
            for length, length_target in length_alloc.items():
                selected.extend(
                    sample_without_replacement(length_groups[length], length_target, rng)
                )

    if len(selected) < target:
        chosen_ids = {row["id"] for row in selected}
        remainder = [row for row in candidates if row["id"] not in chosen_ids]
        remainder.sort(key=lambda row: (status_priority(row["status"]), row["id"]))
        selected.extend(sample_without_replacement(remainder, target - len(selected), rng))

    return selected[:target]


def build_manifest_row(row: dict, sample_set: str, seed: int) -> dict[str, object]:
    return {
        "id": row["id"],
        "data_id": row["item"].get("data_id", row["id"]),
        "image_id": row["item"].get("image_id", row["id"]),
        "question": row["item"].get("question", ""),
        "sample_set": sample_set,
        "baseline_item_ambiguous": row["item_ambiguous"],
        "baseline_item_level": row["item_level"],
        "baseline_status": row["status"],
        "baseline_num_query_steps": row["query_count"],
        "baseline_num_ambiguous_steps": row["num_ambiguous_steps"],
        "baseline_length_bucket": row["length_bucket"],
        "seed": seed,
        "domain": row["domain"],
        "query_category": row["query_category"],
        "dynamism": row["dynamism"],
    }


def inject_sampling_plan(item: dict, manifest_row: dict) -> dict:
    enriched = json.loads(json.dumps(item))
    metadata = dict(enriched.get("metadata", {}))
    metadata["sampling_plan"] = {
        "seed": manifest_row["seed"],
        "sample_set": manifest_row["sample_set"],
        "baseline_item_ambiguous": manifest_row["baseline_item_ambiguous"],
        "baseline_item_level": manifest_row["baseline_item_level"],
        "baseline_status": manifest_row["baseline_status"],
        "baseline_num_query_steps": manifest_row["baseline_num_query_steps"],
        "baseline_num_ambiguous_steps": manifest_row["baseline_num_ambiguous_steps"],
        "baseline_length_bucket": manifest_row["baseline_length_bucket"],
    }
    enriched["metadata"] = metadata
    return enriched


def write_outputs(selected_a: list[dict], selected_b: list[dict], seed: int) -> None:
    manifest_rows = []
    sample_records = []

    combined = [(SET_A_NAME, row) for row in selected_a] + [
        (SET_B_NAME, row) for row in selected_b
    ]
    combined.sort(key=lambda item: (item[1]["id"], item[0]))

    for sample_set, row in combined:
        manifest_row = build_manifest_row(row, sample_set, seed)
        manifest_rows.append(manifest_row)
        sample_records.append(inject_sampling_plan(row["item"], manifest_row))

    with OUTPUT_SAMPLE_PATH.open("w", encoding="utf-8") as f:
        for record in sample_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    fieldnames = list(manifest_rows[0].keys())
    with OUTPUT_MANIFEST_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=500)
    parser.add_argument("--set-a", type=int, default=300)
    parser.add_argument("--set-b", type=int, default=200)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if args.set_a + args.set_b != args.total:
        raise ValueError("--set-a + --set-b must equal --total")

    rng = random.Random(args.seed)
    records = build_records()
    selected_a = sample_set_a(records, args.set_a, rng)
    selected_b = sample_set_b(records, args.set_b, {row["id"] for row in selected_a}, rng)
    write_outputs(selected_a, selected_b, args.seed)

    manifest_counter = Counter([SET_A_NAME] * len(selected_a) + [SET_B_NAME] * len(selected_b))
    print(
        json.dumps(
            {
                "total_records": len(selected_a) + len(selected_b),
                "set_counts": dict(manifest_counter),
                "output_sample": str(OUTPUT_SAMPLE_PATH),
                "output_manifest": str(OUTPUT_MANIFEST_PATH),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
