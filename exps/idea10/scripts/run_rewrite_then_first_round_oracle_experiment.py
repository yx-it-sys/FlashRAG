#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path
import os

for k in [
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
]:
    os.environ.pop(k, None)


SCRIPT_DIR = Path(__file__).resolve().parent
REWRITE_SCRIPT = SCRIPT_DIR / "rewrite_entity_ambiguous_queries_with_llm.py"
EXPERIMENT_SCRIPT = SCRIPT_DIR / "exp_oracle_v1" / "run_first_round_oracle_rewrite_experiment.py"

DEFAULT_REWRITE_INPUT = Path(
    "/home/you/FlashRAG/exps/idea10/data/result/crag_mm_2026_03_31_14_06_experiment/label/trajectory_annotation.llm_labeled.jsonl"
)
DEFAULT_REWRITE_OUTPUT = Path(
    "/home/you/FlashRAG/exps/idea10/data/result/crag_mm_2026_03_31_14_06_experiment/label/trajectory_annotation.llm_labeled.rewrite.jsonl"
)
DEFAULT_PROMPT_PATH = Path(
    "/home/you/FlashRAG/exps/idea10/prompts/rewrite_entity_ambiguous_query.toml"
)
DEFAULT_CONFIG = Path("/home/you/FlashRAG/exps/idea10/configs/config.yaml")
DEFAULT_BASELINE_RESULT_DIR = Path(
    "/home/you/FlashRAG/exps/idea10/data/result/crag_mm_2026_03_31_14_06_experiment"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run query rewriting first, then launch the first-round oracle rewrite experiment "
            "with the produced rewrite labels."
        )
    )
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to use.")

    parser.add_argument("--rewrite-input", type=Path, default=DEFAULT_REWRITE_INPUT)
    parser.add_argument("--rewrite-output", type=Path, default=DEFAULT_REWRITE_OUTPUT)
    parser.add_argument("--rewrite-prompt", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--rewrite-workers", type=int, default=2)

    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--baseline-result-dir", type=Path, default=DEFAULT_BASELINE_RESULT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sample-ids-path", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--max-turns", type=int, default=5)
    return parser.parse_args()


def run_command(cmd: list[str]) -> None:
    print(f"\n[RUN] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()

    rewrite_cmd = [
        args.python,
        str(REWRITE_SCRIPT),
        "--input",
        str(args.rewrite_input),
        "--output",
        str(args.rewrite_output),
        "--prompt",
        str(args.rewrite_prompt),
        "--workers",
        str(args.rewrite_workers),
    ]

    experiment_cmd = [
        args.python,
        str(EXPERIMENT_SCRIPT),
        "--config",
        str(args.config),
        "--baseline-result-dir",
        str(args.baseline_result_dir),
        "--rewrite-label-path",
        str(args.rewrite_output),
        "--seed",
        str(args.seed),
        "--max-turns",
        str(args.max_turns),
    ]

    if args.output_dir is not None:
        experiment_cmd.extend(["--output-dir", str(args.output_dir)])
    if args.sample_ids_path is not None:
        experiment_cmd.extend(["--sample-ids-path", str(args.sample_ids_path)])
    if args.max_samples is not None:
        experiment_cmd.extend(["--max-samples", str(args.max_samples)])
    if args.skip_eval:
        experiment_cmd.append("--skip-eval")

    run_command(rewrite_cmd)
    run_command(experiment_cmd)


if __name__ == "__main__":
    main()
