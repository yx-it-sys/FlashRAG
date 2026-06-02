#!/usr/bin/env python3
import gc
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT_RESULT_DIRS = [
    # Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_qwen3_vl_32b"),
    Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_Qwen3-vl-4B"),
    Path("/home/you/FlashRAG/exps/idea10/data/result/RefAmb_original_Qwen3-vl-8b"),
]
EXPERIMENT_SCRIPT = Path(
    "/home/you/FlashRAG/exps/idea10/scripts/exp_oracle_v1/run_first_round_disturb_rewrite_experiment.py"
)
REWRITE_LABEL_REL_PATH = Path(
    "label/deepseek/omnisearch_trajectories.entity_ambiguity_labeled.disturb_rewrite.jsonl"
)
OUTPUT_DIR_NAME = "first_round_disturb_rewrite_static_prefix_whole"
SLEEP_AFTER_CLEANUP_SECONDS = 3
PKILL_PATTERNS = [
    "vllm",
    "api_server",
    "openai.api_server",
]


def discover_source_dirs() -> list[Path]:
    source_dirs: list[Path] = []
    for root_dir in ROOT_RESULT_DIRS:
        if not root_dir.exists():
            raise FileNotFoundError(f"Root result dir not found: {root_dir}")
        for child in sorted(root_dir.iterdir()):
            if not child.is_dir() or not child.name.endswith("_stage"):
                continue
            config_path = child / "config.yaml"
            rewrite_label_path = child / REWRITE_LABEL_REL_PATH
            if config_path.exists() and rewrite_label_path.exists():
                source_dirs.append(child)

    if not source_dirs:
        raise RuntimeError(
            "No runnable source dirs found under ROOT_RESULT_DIRS; "
            "expected *_stage directories with config.yaml and disturb rewrite label file."
        )
    return source_dirs


def cleanup_gpu_processes() -> None:
    gc.collect()
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as exc:  # noqa: BLE001
        print(f"[Cleanup] torch cleanup skipped: {exc}", flush=True)

    for pattern in PKILL_PATTERNS:
        subprocess.run(
            ["pkill", "-f", pattern],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    time.sleep(SLEEP_AFTER_CLEANUP_SECONDS)


def run_one_source(source_dir: Path) -> None:
    config_path = source_dir / "config.yaml"
    rewrite_label_path = source_dir / REWRITE_LABEL_REL_PATH
    output_dir = source_dir / OUTPUT_DIR_NAME

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    if not rewrite_label_path.exists():
        raise FileNotFoundError(f"Missing disturb rewrite label file: {rewrite_label_path}")

    print(f"\n[Run] source_dir={source_dir}", flush=True)
    print(f"[Run] config={config_path}", flush=True)
    print(f"[Run] baseline_result_dir={source_dir}", flush=True)
    print(f"[Run] rewrite_label_path={rewrite_label_path}", flush=True)
    print(f"[Run] output_dir={output_dir}", flush=True)
    child_env = os.environ.copy()

    subprocess.run(
        [
            sys.executable,
            str(EXPERIMENT_SCRIPT),
            "--config",
            str(config_path),
            "--baseline-result-dir",
            str(source_dir),
            "--rewrite-label-path",
            str(rewrite_label_path),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        env=child_env,
    )


def main() -> None:
    source_dirs = discover_source_dirs()
    for idx, source_dir in enumerate(source_dirs, start=1):
        print(f"\n===== [{idx}/{len(source_dirs)}] Start Replay Experiment =====", flush=True)
        try:
            run_one_source(source_dir)
        finally:
            print("[Cleanup] Releasing vLLM / CUDA-related resources.", flush=True)
            cleanup_gpu_processes()
        print(f"===== [{idx}/{len(source_dirs)}] Done =====", flush=True)

    print(f"\nAll {len(source_dirs)} replay experiments completed.", flush=True)


if __name__ == "__main__":
    main()
