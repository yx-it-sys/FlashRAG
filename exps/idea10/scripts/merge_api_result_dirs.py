from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


def load_jsonl_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_json_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dedupe_by_id(records: list[dict], prefer_later: bool = True) -> list[dict]:
    merged = {}
    order = []
    for record in records:
        item_id = record["id"]
        if item_id not in merged:
            order.append(item_id)
        if prefer_later or item_id not in merged:
            merged[item_id] = record
    return [merged[item_id] for item_id in order]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def compute_average_metrics(intermediate_records: list[dict]) -> dict[str, float]:
    sums = {}
    count = 0
    for record in intermediate_records:
        metric_score = (record.get("output") or {}).get("metric_score")
        if not metric_score:
            continue
        count += 1
        for key, value in metric_score.items():
            sums[key] = sums.get(key, 0.0) + float(value)
    if count == 0:
        return {}
    return {key: value / count for key, value in sums.items()}


def parse_records_txt(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    match = re.search(
        r"Total:\s*([0-9.]+)s\s*\|\s*Count:\s*([0-9]+)\s*\|\s*Avg per item:\s*([0-9.]+)s",
        text,
    )
    if not match:
        return None
    return {
        "total_seconds": float(match.group(1)),
        "count": int(match.group(2)),
        "avg_per_item_seconds": float(match.group(3)),
    }


def write_metric_score(path: Path, metrics: dict[str, float]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")


def write_records(path: Path, sources: list[Path]) -> None:
    parsed = [parse_records_txt(src / "records.txt") for src in sources]
    parsed = [x for x in parsed if x is not None]
    if not parsed:
        return
    total_seconds = sum(x["total_seconds"] for x in parsed)
    total_count = sum(x["count"] for x in parsed)
    avg = total_seconds / total_count if total_count else 0.0
    path.write_text(
        f"[Timing] Total: {total_seconds:.2f}s | Count: {total_count} | Avg per item: {avg:.4f}s\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-a", required=True)
    parser.add_argument("--src-b", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    src_a = Path(args.src_a)
    src_b = Path(args.src_b)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    traj_a = load_jsonl_records(src_a / "omnisearch_trajectories.jsonl")
    traj_b = load_jsonl_records(src_b / "omnisearch_trajectories.jsonl")
    merged_traj = dedupe_by_id(traj_a + traj_b, prefer_later=True)

    inter_a = load_json_records(src_a / "intermediate_data.json")
    inter_b = load_json_records(src_b / "intermediate_data.json")
    merged_inter = dedupe_by_id(inter_a + inter_b, prefer_later=True)

    write_jsonl(out_dir / "omnisearch_trajectories.jsonl", merged_traj)
    write_json(out_dir / "intermediate_data.json", merged_inter)
    write_metric_score(out_dir / "metric_score.txt", compute_average_metrics(merged_inter))
    write_records(out_dir / "records.txt", [src_a, src_b])

    if (src_a / "config.yaml").exists():
        shutil.copy2(src_a / "config.yaml", out_dir / "config.yaml")

    traj_ids = {x["id"] for x in merged_traj}
    inter_ids = {x["id"] for x in merged_inter}
    summary = {
        "source_dirs": [str(src_a), str(src_b)],
        "merged_trajectory_count": len(merged_traj),
        "merged_intermediate_count": len(merged_inter),
        "trajectory_only_ids": sorted(traj_ids - inter_ids),
        "intermediate_only_ids": sorted(inter_ids - traj_ids),
    }
    (out_dir / "merge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
