#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from statistics import mean


DATASET_PATH = Path("/home/you/FlashRAG/exps/idea10/data/datasets/RefAmb/new/task_balanced.jsonl")

RUN_MAPPINGS = [
    {
        "name": "GPT_5.1",
        "new_dir": Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_2026_06_14_19_08_refamb_mcsearch_gpt_5_1_ca_stage"),
        "old_dir": Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_GPT_5.1/RefAmb_2026_05_29_11_04_refamb_mcsearch_gpt_5_1_ca_stage"),
        "target_dir": Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_GPT_5.1/RefAmb_2026_05_29_11_04_refamb_mcsearch_gpt_5_1_ca_stage"),
    },
    {
        "name": "Qwen2.5-vl-7B",
        "new_dir": Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_2026_06_14_19_08_refamb_mcsearch_qwen2_5_7b_stage"),
        "old_dir": Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_Qwen2.5-vl-7B/RefAmb_2026_05_01_11_26_refamb_mcsearch_stage"),
        "target_dir": Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_Qwen2.5-vl-7B/RefAmb_2026_05_01_11_26_refamb_mcsearch_stage"),
    },
    {
        "name": "InternVL3.5-8B",
        "new_dir": Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_2026_06_14_19_48_refamb_mcsearch_intervl3_5_8b_stage"),
        "old_dir": Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_InternVL3.5-8B/RefAmb_2026_05_09_16_39_refamb_mcsearch_intervl3_5_8b_stage"),
        "target_dir": Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_InternVL3.5-8B/RefAmb_2026_05_09_16_39_refamb_mcsearch_intervl3_5_8b_stage"),
    },
    {
        "name": "MMSearch-R1-7B",
        "new_dir": Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_2026_06_14_20_34_refamb_mcsearch_mmsearch_r1_7b_stage"),
        "old_dir": Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_MMSearch-R1-7B/RefAmb_2026_06_06_17_14_refamb_mcsearch_mmsearch_r1_7b_stage"),
        "target_dir": Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_MMSearch-R1-7B/RefAmb_2026_06_06_17_14_refamb_mcsearch_mmsearch_r1_7b_stage"),
    },
    {
        "name": "Qwen3-vl-8b",
        "new_dir": Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_2026_06_14_21_26_refamb_mcsearch_qwen3_vl_8b_stage"),
        "old_dir": Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_Qwen3-vl-8b/RefAmb_2026_05_23_12_53_refamb_mcsearch_qwen3_vl_8b_stage"),
        "target_dir": Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_Qwen3-vl-8b/RefAmb_2026_05_23_12_53_refamb_mcsearch_qwen3_vl_8b_stage"),
    },
    {
        "name": "qwen3_vl_32b",
        "new_dir": Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_2026_06_14_21_29_refamb_mcsearch_qwen3_vl_32b_stage"),
        "old_dir": Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_qwen3_vl_32b/RefAmb_2026_05_19_21_03_refamb_mcsearch_qwen3_vl_32b_stage"),
        "target_dir": Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_qwen3_vl_32b/RefAmb_2026_05_19_21_03_refamb_mcsearch_qwen3_vl_32b_stage"),
    },
]

TOP_LEVEL_FILES = [
    "metric_score.txt",
    "intermediate_data.json",
    "gpt_acc_score.json",
    "omnisearch_trajectories.jsonl",
]

RTE_FILENAMES = [
    "evaluate_trajectory_quality.log",
    "fact_cache.json",
    "trajectory_quality_samples.jsonl",
    "trajectory_quality_summary.json",
    "utility_cache.json",
]

METRIC_ORDER = ["em", "f1", "acc", "precision", "recall", "gpt_acc"]
UTILITY_TO_SCORE = {
    "no_contribution": 0.0,
    "partial_contribution": 0.5,
    "full_contribution": 1.0,
}


def load_current_mcsearch_ids() -> list[str]:
    ids: list[str] = []
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("source") == "mcsearch":
                ids.append(obj["id"])
    return ids


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_json_list_map(path: Path) -> OrderedDict[str, dict]:
    data = load_json(path)
    return OrderedDict((item["id"], item) for item in data)


def load_jsonl_map(path: Path) -> OrderedDict[str, dict]:
    items: OrderedDict[str, dict] = OrderedDict()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            items[obj["id"]] = obj
    return items


def write_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def choose_record(item_id: str, new_map: dict[str, dict], old_map: dict[str, dict]) -> tuple[dict, str]:
    if item_id in new_map:
        return new_map[item_id], "new"
    if item_id in old_map:
        return old_map[item_id], "old"
    raise KeyError(item_id)


def merge_records(current_ids: list[str], new_map: dict[str, dict], old_map: dict[str, dict]) -> tuple[list[dict], dict[str, int]]:
    merged: list[dict] = []
    stats = {"new": 0, "old": 0}
    for item_id in current_ids:
        record, source = choose_record(item_id, new_map, old_map)
        merged.append(record)
        stats[source] += 1
    return merged, stats


def compute_metric_summary(rows: list[dict]) -> OrderedDict[str, float]:
    metrics: OrderedDict[str, list[float]] = OrderedDict()
    for name in METRIC_ORDER:
        metrics[name] = []
    for row in rows:
        score_dict = row.get("output", {}).get("metric_score", {})
        for key, value in score_dict.items():
            if key not in metrics:
                metrics[key] = []
            metrics[key].append(float(value))
    return OrderedDict((k, mean(v)) for k, v in metrics.items() if v)


def write_metric_score(path: Path, rows: list[dict]) -> None:
    metric_summary = compute_metric_summary(rows)
    with path.open("w", encoding="utf-8") as f:
        for key, value in metric_summary.items():
            f.write(f"{key}: {value}\n")


def list_rte_dirs(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {p.name for p in path.iterdir() if p.is_dir() and p.name.startswith("trajectory_quality_eval_whole")}


def load_json_dict(path: Path) -> dict:
    if not path.exists():
        return {}
    data = load_json(path)
    if isinstance(data, dict):
        return data
    raise TypeError(f"Expected dict json: {path}")


def build_trajectory_retrieval_steps(trajectory_record: dict) -> list[dict]:
    steps = []
    for entry in trajectory_record.get("trajectory", []):
        action = entry.get("action")
        if action in {"text_retrieval_result", "image_retrieval_result"}:
            retrieval_type = "Text Retrieval" if action == "text_retrieval_result" else "Image Retrieval"
            steps.append(
                {
                    "query": entry.get("query"),
                    "evidence": entry.get("content", ""),
                    "retrieval_type": retrieval_type,
                }
            )
    return steps


def build_fact_cache_for_samples(
    samples: list[dict],
    trajectories: dict[str, dict],
    new_cache: dict,
    old_cache: dict,
) -> dict:
    merged_cache: dict = {}
    for sample in samples:
        retrieval_steps = build_trajectory_retrieval_steps(trajectories[sample["id"]])
        sample_steps = sample.get("steps", [])
        for idx, sample_step in enumerate(sample_steps):
            if idx >= len(retrieval_steps):
                continue
            retrieval = retrieval_steps[idx]
            key = json.dumps(
                {
                    "query": retrieval["query"],
                    "evidence": retrieval["evidence"],
                    "retrieval_type": sample_step.get("retrieval_type", retrieval["retrieval_type"]),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            if key in new_cache:
                merged_cache[key] = new_cache[key]
            elif key in old_cache:
                merged_cache[key] = old_cache[key]
    return merged_cache


def build_utility_cache_for_samples(samples: list[dict], new_cache: dict, old_cache: dict) -> dict:
    merged_cache: dict = {}
    for sample in samples:
        for step in sample.get("steps", []):
            key = json.dumps(
                {
                    "sub_question": step.get("sub_question"),
                    "reasoning": step.get("reaction_text"),
                    "facts": step.get("facts_t", []),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            if key in new_cache:
                merged_cache[key] = new_cache[key]
            elif key in old_cache:
                merged_cache[key] = old_cache[key]
    return merged_cache


def build_rte_summary(base_summary: dict, samples: list[dict]) -> dict:
    steps = [step for sample in samples for step in sample.get("steps", [])]
    meta = dict(base_summary.get("meta", {}))
    meta["num_samples"] = len(samples)
    meta["num_steps"] = len(steps)
    aggregate = {
        "mean_tqs": mean(sample["trajectory_quality_score"] for sample in samples) if samples else 0.0,
        "mean_iterations": mean(sample["num_iterations"] for sample in samples) if samples else 0.0,
        "mean_delta_f": mean(step["delta_f"] for step in steps) if steps else 0.0,
        "mean_utility": mean(UTILITY_TO_SCORE.get(step["utility"], step["utility"]) for step in steps) if steps else 0.0,
        "mean_step_score": mean(step["step_score"] for step in steps) if steps else 0.0,
        "informative_step_rate": (sum(1 for step in steps if step["delta_f"] > 0) / len(steps)) if steps else 0.0,
    }
    return {"meta": meta, "aggregate": aggregate}


def update_rte_dir(
    current_ids: list[str],
    dir_name: str,
    new_dir: Path,
    old_dir: Path,
    target_dir: Path,
    merged_trajectory_map: dict[str, dict],
) -> dict[str, int]:
    source_new = new_dir / dir_name
    source_old = old_dir / dir_name
    target = target_dir / dir_name
    target.mkdir(parents=True, exist_ok=True)

    new_sample_map = load_jsonl_map(source_new / "trajectory_quality_samples.jsonl") if (source_new / "trajectory_quality_samples.jsonl").exists() else OrderedDict()
    old_sample_map = load_jsonl_map(source_old / "trajectory_quality_samples.jsonl") if (source_old / "trajectory_quality_samples.jsonl").exists() else OrderedDict()
    merged_samples, stats = merge_records(
        [item_id for item_id in current_ids if item_id in new_sample_map or item_id in old_sample_map],
        new_sample_map,
        old_sample_map,
    )
    write_jsonl(target / "trajectory_quality_samples.jsonl", merged_samples)

    base_summary = {}
    if (source_new / "trajectory_quality_summary.json").exists():
        base_summary = load_json(source_new / "trajectory_quality_summary.json")
    elif (source_old / "trajectory_quality_summary.json").exists():
        base_summary = load_json(source_old / "trajectory_quality_summary.json")
    write_json(target / "trajectory_quality_summary.json", build_rte_summary(base_summary, merged_samples))

    new_fact_cache = load_json_dict(source_new / "fact_cache.json")
    old_fact_cache = load_json_dict(source_old / "fact_cache.json")
    write_json(
        target / "fact_cache.json",
        build_fact_cache_for_samples(merged_samples, merged_trajectory_map, new_fact_cache, old_fact_cache),
    )

    new_utility_cache = load_json_dict(source_new / "utility_cache.json")
    old_utility_cache = load_json_dict(source_old / "utility_cache.json")
    write_json(
        target / "utility_cache.json",
        build_utility_cache_for_samples(merged_samples, new_utility_cache, old_utility_cache),
    )

    with (target / "evaluate_trajectory_quality.log").open("w", encoding="utf-8") as f:
        f.write(f"[Merged] directory={dir_name}\n")
        f.write(f"target={target}\n")
        f.write(f"new_source={source_new if source_new.exists() else '<missing>'}\n")
        f.write(f"old_source={source_old if source_old.exists() else '<missing>'}\n")
        f.write(f"merged_samples={len(merged_samples)} new={stats['new']} old={stats['old']}\n")

    return {"samples": len(merged_samples), "new": stats["new"], "old": stats["old"]}


def update_mapping(current_ids: list[str], mapping: dict) -> dict:
    new_dir = mapping["new_dir"]
    old_dir = mapping["old_dir"]
    target_dir = mapping["target_dir"]

    new_intermediate = load_json_list_map(new_dir / "intermediate_data.json")
    old_intermediate = load_json_list_map(old_dir / "intermediate_data.json")
    merged_intermediate, stats = merge_records(current_ids, new_intermediate, old_intermediate)
    write_json(target_dir / "intermediate_data.json", merged_intermediate)
    write_metric_score(target_dir / "metric_score.txt", merged_intermediate)

    new_gpt_path = new_dir / "gpt_acc_score.json"
    old_gpt_path = old_dir / "gpt_acc_score.json"
    new_gpt_map = load_json_list_map(new_gpt_path) if new_gpt_path.exists() else new_intermediate
    old_gpt_map = load_json_list_map(old_gpt_path) if old_gpt_path.exists() else old_intermediate
    merged_gpt, _ = merge_records(current_ids, new_gpt_map, old_gpt_map)
    write_json(target_dir / "gpt_acc_score.json", merged_gpt)

    new_traj = load_jsonl_map(new_dir / "omnisearch_trajectories.jsonl")
    old_traj = load_jsonl_map(old_dir / "omnisearch_trajectories.jsonl")
    merged_traj, _ = merge_records(current_ids, new_traj, old_traj)
    write_jsonl(target_dir / "omnisearch_trajectories.jsonl", merged_traj)
    merged_traj_map = OrderedDict((item["id"], item) for item in merged_traj)

    rte_names = list_rte_dirs(target_dir) | list_rte_dirs(new_dir) | list_rte_dirs(old_dir)
    rte_stats = {}
    for dir_name in sorted(rte_names):
        rte_stats[dir_name] = update_rte_dir(current_ids, dir_name, new_dir, old_dir, target_dir, merged_traj_map)

    return {
        "model": mapping["name"],
        "intermediate_total": len(merged_intermediate),
        "intermediate_from_new": stats["new"],
        "intermediate_from_old": stats["old"],
        "rte": rte_stats,
    }


def main() -> None:
    current_ids = load_current_mcsearch_ids()
    report = []
    for mapping in RUN_MAPPINGS:
        report.append(update_mapping(current_ids, mapping))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
