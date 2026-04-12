#!/usr/bin/env python3

import json
from pathlib import Path


PORT = 8008
HOST = "127.0.0.1"

BASE_DIR = Path(__file__).resolve().parent
TASKS_PATH = BASE_DIR / "search_action_label_studio_grouped.json"
IMAGES_DIR = BASE_DIR.parents[2] / "datasets" / "crag_mm" / "images"


def main() -> None:
    tasks = json.loads(TASKS_PATH.read_text(encoding="utf-8"))
    missing = []

    for task in tasks:
        sample_id = task["data"]["sample_id"]
        image_path = IMAGES_DIR / f"{sample_id}.jpg"
        if not image_path.exists():
            missing.append(sample_id)
            continue
        task["data"]["image_url"] = f"http://{HOST}:{PORT}/{sample_id}.jpg"

    TASKS_PATH.write_text(
        json.dumps(tasks, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"updated {len(tasks) - len(missing)} tasks in {TASKS_PATH}")
    print(f"serve images with: python3 -m http.server {PORT} --directory {IMAGES_DIR}")
    if missing:
        print(f"missing images: {len(missing)}")
        for sample_id in missing[:20]:
            print(sample_id)


if __name__ == "__main__":
    main()
