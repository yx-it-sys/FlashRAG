import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from tqdm import tqdm

from flashrag.evaluator.metrics import GPTAcc
import os

for k in [
    "http_proxy", "https_proxy",
    "HTTP_PROXY", "HTTPS_PROXY",
    "all_proxy", "ALL_PROXY"
]:
    os.environ.pop(k, None)

DEFAULT_RESULT_DIR = Path("/home/you/FlashRAG/exps/idea10/data/result/2026_04_10_12_53_48_first_round_oracle_rewrite_experiment")
SAVE_NAME = "gpt_acc_score.json"
MAX_RETRIES = 5
RETRY_SLEEP_SECONDS = 10
SAVE_EVERY = 1
MAX_WORKERS = 10
METRIC_SCORE_NAME = "metric_score.txt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recompute GPTAcc from an existing intermediate_data.json without rerunning the pipeline."
    )
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--save-name", type=str, default=SAVE_NAME)
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    return parser.parse_args()


def extract_final_answer(text):
    if not isinstance(text, str):
        return ""
    if "Final Answer:" in text:
        return text.split("Final Answer:", 1)[1].split("</End>", 1)[0].strip()
    return text.strip()


def build_samples(items, metric):
    samples = []
    for item in items:
        golden_answers = item.get("answer_eval", item.get("answer", []))
        golden_answers = metric._canonicalize_answer_list(golden_answers)
        pred = extract_final_answer(item.get("output", {}).get("pred", ""))
        samples.append(
            {
                "id": item.get("id") or item.get("data_id"),
                "question": item.get("question", ""),
                "pred": pred,
                "golden_answers": golden_answers,
            }
        )
    return samples


def load_progress(save_path, items):
    if not save_path.exists():
        return [None] * len(items), items

    with open(save_path, "r", encoding="utf-8") as f:
        saved = json.load(f)

    if isinstance(saved, list):
        score_list = [None] * len(items)
        merged_items = items
        for idx, item in enumerate(saved[: len(items)]):
            merged_items[idx] = item
            score = (
                item.get("output", {})
                .get("metric_score", {})
                .get("gpt_acc")
            )
            score_list[idx] = score
        return score_list, merged_items

    saved_scores = saved.get("score_list", [])
    score_list = [None] * len(items)
    for idx, score in enumerate(saved_scores[: len(items)]):
        score_list[idx] = score
        if score is not None:
            items[idx].setdefault("output", {})
            items[idx]["output"].setdefault("metric_score", {})
            items[idx]["output"]["metric_score"]["gpt_acc"] = score
    return score_list, items


def dump_progress(save_path, items):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def dump_intermediate_data(data_path, items):
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def load_metric_summary(metric_score_path):
    summary = {}
    if not metric_score_path.exists():
        return summary
    with open(metric_score_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            try:
                summary[key] = float(value)
            except ValueError:
                summary[key] = value
    return summary


def dump_metric_score(metric_score_path, final_score):
    summary = load_metric_summary(metric_score_path)
    summary["gpt_acc"] = final_score
    with open(metric_score_path, "w", encoding="utf-8") as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")


def score_one(metric, pred, golden_answers):
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return metric.calculate_acc(pred, golden_answers)
        except Exception as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            print(f"[retry {attempt}/{MAX_RETRIES}] {type(exc).__name__}: {exc}")
            time.sleep(RETRY_SLEEP_SECONDS * attempt)
    raise last_error


def score_one_with_index(metric, idx, sample):
    score = score_one(metric, sample["pred"], sample["golden_answers"])
    return idx, score


def main():
    args = parse_args()
    result_dir = args.result_dir
    data_path = result_dir / "intermediate_data.json"
    config_path = args.config or (result_dir / "config.yaml")
    save_path = result_dir / args.save_name
    metric_score_path = result_dir / METRIC_SCORE_NAME

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    with open(data_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    metric = GPTAcc(config)
    samples = build_samples(items, metric)
    score_list, items = load_progress(save_path, items)

    progress = tqdm(total=len(samples), desc="Calculating GPTAcc")
    progress.update(sum(score is not None for score in score_list))

    pending_indices = [idx for idx, score in enumerate(score_list) if score is None]

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_idx = {
            executor.submit(score_one_with_index, metric, idx, samples[idx]): idx
            for idx in pending_indices
        }

        completed_since_save = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                _, score = future.result()
            except Exception as exc:
                dump_progress(save_path, items)
                progress.close()
                print(f"Stopped at index {idx} after repeated failures: {type(exc).__name__}: {exc}")
                raise

            score_list[idx] = score
            items[idx].setdefault("output", {})
            items[idx]["output"].setdefault("metric_score", {})
            items[idx]["output"]["metric_score"]["gpt_acc"] = score
            progress.update(1)
            completed_since_save += 1

            if completed_since_save >= SAVE_EVERY:
                dump_progress(save_path, items)
                dump_intermediate_data(data_path, items)
                completed_since_save = 0

    progress.close()
    dump_progress(save_path, items)
    dump_intermediate_data(data_path, items)

    completed_scores = [score for score in score_list if score is not None]
    final_score = sum(completed_scores) / len(completed_scores) if completed_scores else 0.0
    dump_metric_score(metric_score_path, final_score)
    print(json.dumps({"gpt_acc": final_score}, ensure_ascii=False))
    print(f"Saved scores to: {save_path}")
    print(f"Saved metric summary to: {metric_score_path}")


if __name__ == "__main__":
    main()
