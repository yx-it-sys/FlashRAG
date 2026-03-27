import io
import json
import time
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image
import requests


TRAIN_JSONL = Path("/home/you/FlashRAG/exps/idea10/data/datasets/mcsearch/train.jsonl")
IMAGE_INFO_JSON = Path("/home/you/FlashRAG/exps/idea10/data/datasets/mcsearch/corpus/all_image_infos.json")
OUTPUT_DIR = Path("/home/you/FlashRAG/exps/idea10/data/datasets/mcsearch/corpus/supporting_fact_images")
ID_LIST_PATH = Path("/home/you/FlashRAG/exps/idea10/data/datasets/mcsearch/corpus/supporting_fact_image_ids.txt")
STATS_PATH = Path("/home/you/FlashRAG/exps/idea10/data/datasets/mcsearch/corpus/supporting_fact_image_stats.json")
FAIL_LOG = Path("/home/you/FlashRAG/exps/idea10/data/datasets/mcsearch/corpus/supporting_fact_image_failures.jsonl")

TIMEOUT = 60
MAX_RETRIES = 5
WIKIMEDIA_MIN_INTERVAL = 1.0
DEFAULT_BACKOFF = 2.0
CONTACT_EMAIL = "replace-with-your-email@example.com"


def extract_supporting_fact_ids() -> list[str]:
    ids = set()
    with TRAIN_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            for item in obj.get("subqa_chain", []):
                if item.get("modality") != "image":
                    continue
                supporting_fact_id = str(item.get("supporting_fact_id", "")).strip()
                if supporting_fact_id.isdigit():
                    ids.add(supporting_fact_id)
    return sorted(ids)


def load_image_url_map() -> dict[str, str]:
    with IMAGE_INFO_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        str(item["image_id"]): item["imgUrl"]
        for item in data
        if item.get("image_id") is not None and item.get("imgUrl")
    }


def convert_to_jpg(image_bytes: bytes, out_path: Path) -> None:
    with Image.open(io.BytesIO(image_bytes)) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        else:
            img = img.copy()
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    img.save(tmp_path, format="JPEG", quality=95)
    tmp_path.replace(out_path)


def is_wikimedia_url(url: str) -> bool:
    return urlparse(url).netloc.lower() == "upload.wikimedia.org"


def get_retry_delay(response: requests.Response | None, attempt: int, is_wikimedia: bool) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(float(retry_after), WIKIMEDIA_MIN_INTERVAL if is_wikimedia else 0.0)
            except ValueError:
                pass
    base = WIKIMEDIA_MIN_INTERVAL if is_wikimedia else DEFAULT_BACKOFF
    return base * (2 ** (attempt - 1))


def download_one(session: requests.Session, image_id: str, img_url: str) -> tuple[bool, str, dict | None]:
    out_path = OUTPUT_DIR / f"{image_id}.jpg"
    wikimedia = is_wikimedia_url(img_url)

    if out_path.exists() and out_path.stat().st_size > 0:
        return True, f"skip {out_path.name}", None

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        response = None
        try:
            if wikimedia:
                time.sleep(WIKIMEDIA_MIN_INTERVAL)

            with session.get(img_url, timeout=TIMEOUT, stream=True) as response:
                response.raise_for_status()
                image_bytes = bytearray()
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        image_bytes.extend(chunk)
                convert_to_jpg(bytes(image_bytes), out_path)
                return True, f"ok {out_path.name}", None
        except requests.HTTPError as exc:
            last_error = exc
            time.sleep(get_retry_delay(exc.response, attempt, wikimedia))
        except Exception as exc:
            last_error = exc
            time.sleep(get_retry_delay(response, attempt, wikimedia))

    failure = {
        "image_id": image_id,
        "imgUrl": img_url,
        "reason": repr(last_error),
    }
    return False, f"failed {image_id}: {last_error}", failure


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FAIL_LOG.parent.mkdir(parents=True, exist_ok=True)

    supporting_fact_ids = extract_supporting_fact_ids()
    ID_LIST_PATH.write_text("\n".join(supporting_fact_ids) + "\n", encoding="utf-8")

    image_url_map = load_image_url_map()
    matched = [(image_id, image_url_map[image_id]) for image_id in supporting_fact_ids if image_id in image_url_map]
    missing = [image_id for image_id in supporting_fact_ids if image_id not in image_url_map]

    stats = {
        "total_unique_supporting_fact_ids": len(supporting_fact_ids),
        "matched_image_urls": len(matched),
        "missing_in_all_image_infos": len(missing),
        "missing_image_ids": missing,
        "id_list_path": str(ID_LIST_PATH),
        "output_dir": str(OUTPUT_DIR),
        "fail_log": str(FAIL_LOG),
    }
    STATS_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "FlashRAG-MCSearch-SupportingFact-Downloader/1.0 "
                f"(contact: {CONTACT_EMAIL})"
            ),
            "Accept": "image/*,*/*;q=0.8",
        }
    )

    success = 0
    failed = 0
    with FAIL_LOG.open("w", encoding="utf-8") as log_f:
        for idx, (image_id, img_url) in enumerate(matched, start=1):
            ok, message, failure = download_one(session, image_id, img_url)
            if ok:
                success += 1
            else:
                failed += 1
                log_f.write(json.dumps(failure, ensure_ascii=False) + "\n")
            if idx % 50 == 0 or not ok:
                print(f"[{idx}/{len(matched)}] {message}")

    print(
        f"done success={success} failed={failed} "
        f"total_unique_ids={len(supporting_fact_ids)} matched={len(matched)} missing={len(missing)}"
    )


if __name__ == "__main__":
    main()
