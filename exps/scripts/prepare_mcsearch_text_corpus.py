import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a FlashRAG text corpus jsonl from MC-Search KB all_docs.json."
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path("/home/you/FlashRAG/exps/idea10/data/datasets/MC-Search/data/KB/all_docs.json"),
        help="Path to MC-Search all_docs.json (actually JSONL).",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("/home/you/FlashRAG/exps/idea10/data/datasets/MC-Search/corpus/all_docs.jsonl"),
        help="Output FlashRAG corpus path.",
    )
    return parser.parse_args()


def iter_records(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    args = parse_args()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with args.output_jsonl.open("w", encoding="utf-8") as fout:
        for count, item in enumerate(iter_records(args.input_jsonl), start=1):
            title = str(item.get("title", "")).strip()
            fact = str(item.get("fact", "")).strip()
            contents = title if not fact else f"{title}\n{fact}" if title else fact

            out = {
                "id": str(item.get("snippet_id", count - 1)),
                "title": title,
                "text": fact,
                "contents": contents,
                "url": item.get("url", ""),
                "snippet_id": str(item.get("snippet_id", count - 1)),
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"output_jsonl={args.output_jsonl}")
    print(f"written_records={count}")


if __name__ == "__main__":
    main()
