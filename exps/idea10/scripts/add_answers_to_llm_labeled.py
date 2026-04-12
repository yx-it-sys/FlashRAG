import json
from pathlib import Path


BASE_DIR = Path("/home/you/FlashRAG/exps/idea10/data/result/crag_mm_2026_03_31_14_06_experiment")
JSONL_PATH = BASE_DIR / "label/trajectory_annotation.llm_labeled.jsonl"
INTERMEDIATE_PATH = BASE_DIR / "intermediate_data.json"


def main() -> None:
    intermediate = json.loads(INTERMEDIATE_PATH.read_text())
    answers_by_id = {item["id"]: item.get("answer") for item in intermediate}

    updated_lines = []
    missing = []

    with open(JSONL_PATH) as f:
        for line in f:
            obj = json.loads(line)
            annotation_id = obj.get("annotation_id")
            if annotation_id not in answers_by_id:
                missing.append(annotation_id)
            else:
                obj["Answers"] = answers_by_id[annotation_id]
            updated_lines.append(json.dumps(obj, ensure_ascii=False))

    if missing:
        raise KeyError(f"Missing answers for {len(missing)} annotation_ids: {missing[:5]}")

    JSONL_PATH.write_text("\n".join(updated_lines) + "\n")
    print(f"Updated {len(updated_lines)} items in {JSONL_PATH}")


if __name__ == "__main__":
    main()
