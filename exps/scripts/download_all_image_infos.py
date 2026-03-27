import json
import io
import time
from urllib.parse import urlparse
from pathlib import Path

from PIL import Image
import requests


SOURCE_JSON = Path("/home/you/FlashRAG/exps/idea10/data/datasets/mcsearch/corpus/all_image_infos.json")
OUTPUT_DIR = Path("/home/you/FlashRAG/exps/idea10/data/datasets/mcsearch/corpus/images_from_imgurl")
FAIL_LOG = Path("/home/you/FlashRAG/exps/idea10/data/datasets/mcsearch/corpus/download_failures.jsonl")
TIMEOUT = 60
MAX_RETRIES = 5
WIKIMEDIA_MIN_INTERVAL = 1.0
DEFAULT_BACKOFF = 2.0
CONTACT_EMAIL = "replace-with-your-email@example.com"


def load_items() -> list[dict]:
    with SOURCE_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {SOURCE_JSON}, got {type(data).__name__}")
    return data

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


def download_one(session: requests.Session, item: dict) -> tuple[bool, str, dict | None]:
    image_id = str(item["image_id"])
    img_url = item["imgUrl"]
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
            status = exc.response.status_code if exc.response is not None else None
            if status == 429:
                time.sleep(get_retry_delay(exc.response, attempt, wikimedia))
            else:
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
    items = load_items()
    FAIL_LOG.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "FlashRAG-MCSearch-Downloader/1.0 "
                f"(contact: {CONTACT_EMAIL})"
            ),
            "Accept": "image/*,*/*;q=0.8",
        }
    )

    success = 0
    failed = 0
    with FAIL_LOG.open("w", encoding="utf-8") as log_f:
        for idx, item in enumerate(items, start=1):
            ok, message, failure = download_one(session, item)
            if ok:
                success += 1
            else:
                failed += 1
                log_f.write(json.dumps(failure, ensure_ascii=False) + "\n")
            if idx % 100 == 0 or not ok:
                print(f"[{idx}/{len(items)}] {message}")

    print(f"done success={success} failed={failed} output_dir={OUTPUT_DIR} fail_log={FAIL_LOG}")


if __name__ == "__main__":
    main()
