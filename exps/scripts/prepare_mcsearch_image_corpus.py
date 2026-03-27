import argparse
import json
from pathlib import Path

from datasets import Dataset, Image


COMMON_SUFFIXES = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a CLIP-ready image corpus parquet from all_image_infos.json.")
    parser.add_argument(
        "--all-image-infos",
        type=Path,
        default=Path("/home/you/FlashRAG/exps/idea10/data/datasets/mcsearch/corpus/all_image_infos.json"),
        help="Path to all_image_infos.json.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("/home/you/FlashRAG/exps/idea10/data/datasets/infoseek_val/images"),
        help="Directory containing downloaded images named by image_id.",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=Path("/home/you/FlashRAG/exps/idea10/data/datasets/mcsearch/corpus/image_corpus.parquet"),
        help="Output parquet path.",
    )
    return parser.parse_args()


def find_image_path(image_dir: Path, image_id: str) -> Path | None:
    for suffix in COMMON_SUFFIXES:
        candidate = image_dir / f"{image_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    args = parse_args()

    with args.all_image_infos.open("r", encoding="utf-8") as f:
        infos = json.load(f)

    records = []
    missing = 0
    for item in infos:
        image_id = str(item["image_id"])
        title = str(item.get("title", "")).strip()
        image_path = find_image_path(args.image_dir, image_id)
        if image_path is None:
            missing += 1
            continue

        records.append(
            {
                "id": image_id,
                "title": title,
                "text": title,
                "contents": title,
                "image": str(image_path),
            }
        )

    ds = Dataset.from_list(records)
    ds = ds.cast_column("image", Image())

    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(args.output_parquet))

    print(f"input_items={len(infos)}")
    print(f"kept_with_local_image={len(ds)}")
    print(f"missing_local_image={missing}")
    print(f"output_parquet={args.output_parquet}")


if __name__ == "__main__":
    main()
