import json
from pathlib import Path


BASE_DIR = Path("/home/you/FlashRAG/exps/idea10/data/result/crag_mm_2026_03_31_14_06_experiment")
GROUPED_PATH = BASE_DIR / "label/trajectory_annotation_label_studio_grouped.json"
INTERMEDIATE_PATH = BASE_DIR / "intermediate_data.json"


def main() -> None:
    grouped = json.loads(GROUPED_PATH.read_text())
    intermediate = json.loads(INTERMEDIATE_PATH.read_text())

    answers_by_id = {item["id"]: item.get("answer") for item in intermediate}

    missing = []
    for group in grouped:
        sample_id = group.get("data", {}).get("sample_id")
        if sample_id not in answers_by_id:
            missing.append(sample_id)
            continue
        group["data"]["Golden Answers"] = answers_by_id[sample_id]

    if missing:
        raise KeyError(f"Missing answers for {len(missing)} sample_ids: {missing[:5]}")

    GROUPED_PATH.write_text(json.dumps(grouped, ensure_ascii=False, indent=2) + "\n")
    print(f"Updated {len(grouped)} groups in {GROUPED_PATH}")


if __name__ == "__main__":
    main()
