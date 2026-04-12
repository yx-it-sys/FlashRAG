#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import yaml

from flashrag.dataset.dataset import Dataset
from flashrag.evaluator.evaluator import Evaluator


DEFAULT_INTERMEDIATE_PATH = Path(
    "/home/you/FlashRAG/exps/idea10/data/result/2026_04_10_12_53_48_first_round_oracle_rewrite_experiment/intermediate_data.json"
)
DEFAULT_CONFIG_PATH = Path("/home/you/FlashRAG/exps/idea10/configs/config.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill ROUGE-L for a FlashRAG intermediate_data.json file, "
            "write per-sample scores to output.metric_score, and print the "
            "overall ROUGE-L score."
        )
    )
    parser.add_argument("--intermediate", type=Path, default=DEFAULT_INTERMEDIATE_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def load_config(config_path: Path, save_dir: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config = dict(config)
    config["save_dir"] = str(save_dir)
    config["save_intermediate_data"] = True
    config["save_metric_score"] = True
    config["metrics"] = ["rouge-l"]
    return config


def load_data(intermediate_path: Path) -> list[dict]:
    with intermediate_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    intermediate_path = args.intermediate.resolve()
    config = load_config(args.config.resolve(), intermediate_path.parent)
    raw_data = load_data(intermediate_path)
    dataset = Dataset(config=config, data=raw_data)
    evaluator = Evaluator(config)
    result = evaluator.evaluate(dataset)
    rouge_l = result.get("rouge-l")
    if rouge_l is None:
        raise RuntimeError("Evaluator did not return rouge-l.")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Updated {intermediate_path}")
    print(f"Overall ROUGE-L: {rouge_l:.12f}")


if __name__ == "__main__":
    main()
