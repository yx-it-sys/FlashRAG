import argparse
import gc
import json
import os
from pathlib import Path
from typing import Iterable

for k in [
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
]:
    os.environ.pop(k, None)

PROJECT_ROOT = Path("/home/you/FlashRAG/exps/idea10")
DEFAULT_TRAIN_PATH = Path("/home/you/FlashRAG/exps/idea10/data/datasets/RefAmb/new/task_balanced.jsonl")
SOURCE_CHOICES = ["crag", "infoseek", "mcsearch", "oven"]
MODEL_TO_SOURCE_CONFIG = {
    "default": {
        "crag": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/gpt/enhanced/config_stage_crag.yaml",
        "infoseek": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/gpt/enhanced/config_stage_infoseek.yaml",
        "oven": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/gpt/enhanced/config_stage_oven.yaml",
        "mcsearch": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/gpt/enhanced/config_stage_mcsearch.yaml",
    },
    "gpt": {
        "crag": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/gpt/enhanced/config_stage_crag.yaml",
        "infoseek": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/gpt/enhanced/config_stage_infoseek.yaml",
        "oven": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/gpt/enhanced/config_stage_oven.yaml",
        "mcsearch": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/gpt/enhanced/config_stage_mcsearch.yaml",
    },
    "qwen2_5_7b": {
        "crag": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen2_5_7b/enhanced/sim/enhanced_thr_080/config_stage_crag.yaml",
        "infoseek": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen2_5_7b/enhanced/sim/enhanced_thr_080/config_stage_infoseek.yaml",
        "oven": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen2_5_7b/enhanced/sim/enhanced_thr_080/config_stage_oven.yaml",
        "mcsearch": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen2_5_7b/enhanced/sim/enhanced_thr_080/config_stage_mcsearch.yaml",
    },
    "intervl3_5_8b": {
        "crag": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/intervl3_5_8b/enhanced/config_stage_crag.yaml",
        "infoseek": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/intervl3_5_8b/enhanced/config_stage_infoseek.yaml",
        "oven": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/intervl3_5_8b/enhanced/config_stage_oven.yaml",
        "mcsearch": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/intervl3_5_8b/enhanced/config_stage_mcsearch.yaml",
    },
    "qwen3_vl_4b": {
        "crag": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen3_vl_4b/enhanced/config_stage_crag.yaml",
        "infoseek": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen3_vl_4b/enhanced/config_stage_infoseek.yaml",
        "oven": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen3_vl_4b/enhanced/config_stage_oven.yaml",
        "mcsearch": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen3_vl_4b/enhanced/config_stage_mcsearch.yaml",
    },
    "qwen3_vl_8b": {
        "crag": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen3_vl_8b/enhanced/config_stage_crag.yaml",
        "infoseek": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen3_vl_8b/enhanced/config_stage_infoseek.yaml",
        "oven": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen3_vl_8b/enhanced/config_stage_oven.yaml",
        "mcsearch": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen3_vl_8b/enhanced/config_stage_mcsearch.yaml",
    },
    "qwen3_vl_32b": {
        "crag": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen3_vl_32b/enhanced/config_stage_crag.yaml",
        "infoseek": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen3_vl_32b/enhanced/config_stage_infoseek.yaml",
        "oven": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen3_vl_32b/enhanced/config_stage_oven.yaml",
        "mcsearch": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen3_vl_32b/enhanced/config_stage_mcsearch.yaml",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IIGCR_Pipeline on RefAmb subset for one source.")
    parser.add_argument("--source", required=True, choices=SOURCE_CHOICES, help="RefAmb source to run.")
    parser.add_argument(
        "--model",
        default="default",
        choices=sorted(MODEL_TO_SOURCE_CONFIG.keys()),
        help="Config group / generator variant to use.",
    )
    parser.add_argument(
        "--train-path",
        default=DEFAULT_TRAIN_PATH,
        type=Path,
        help="Path to RefAmb subset jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional existing or target stage directory. When set, reuse this directory in place and do not create a new timestamped result subdirectory.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="Optional explicit config yaml path. Overrides the built-in model/source config mapping.",
    )
    parser.add_argument(
        "--save-note-suffix",
        default="",
        help="Optional suffix appended to save_note for distinguishing hyperparameter sweeps.",
    )
    parser.add_argument(
        "--disable-incremental",
        action="store_true",
        help="Do not skip items that already appear in the trajectory file.",
    )
    parser.add_argument(
        "--disable-auto-history-resume",
        action="store_true",
        help="Do not auto-scan historical result directories with the same save_note.",
    )
    parser.add_argument(
        "--history-root",
        action="append",
        default=None,
        help="Optional extra root to scan for historical result directories. Can be repeated.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of remaining items to run after filtering.",
    )
    parser.add_argument(
        "--disable-eval",
        action="store_true",
        help="Skip FlashRAG evaluation and metric writing; only run inference and trajectory logging.",
    )
    return parser.parse_args()


def resolve_config_path(model: str, source: str, explicit_config_path: Path | None = None) -> str:
    config_path = explicit_config_path or Path(MODEL_TO_SOURCE_CONFIG[model][source])
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return str(config_path)


def resolve_save_note(model: str, source: str, save_note_suffix: str = "") -> str:
    suffix = save_note_suffix.strip().replace("/", "_")
    if suffix:
        return f"refamb_eao_{source}_{model}_{suffix}_stage"
    return f"refamb_eao_{source}_{model}_stage"


def resolve_runtime_gpu_id() -> str | None:
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if gpu_id:
        return gpu_id
    gpu_id = os.environ.get("GPU_ID", "").strip()
    if gpu_id:
        return gpu_id
    return None


def import_flashrag_modules():
    from flashrag.config import Config
    from flashrag.dataset.dataset import Dataset
    from flashrag.utils import get_generator, get_retriever
    from flashrag.pipeline.iigcr_pipeline import IIGCR_Pipeline

    return Config, Dataset, get_generator, get_retriever, IIGCR_Pipeline


def load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    existing_ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            item_id = record.get("id")
            if item_id:
                existing_ids.add(item_id)
    return existing_ids


def iter_history_roots(extra_roots: list[str] | None) -> list[Path]:
    roots = [PROJECT_ROOT / "data/result"]
    if extra_roots:
        roots.extend(Path(item) for item in extra_roots if item)
    unique_roots = []
    seen = set()
    for root in roots:
        resolved = root.resolve()
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        unique_roots.append(resolved)
    return unique_roots


def discover_historical_result_dirs(
    save_note: str,
    roots: Iterable[Path],
    exclude_dirs: Iterable[Path] | None = None,
) -> list[Path]:
    exclude_resolved = {path.resolve() for path in (exclude_dirs or []) if path.exists()}
    matches: list[Path] = []
    suffix = f"_{save_note}"
    for root in roots:
        for candidate in root.rglob(f"*{suffix}"):
            if not candidate.is_dir():
                continue
            if candidate.resolve() in exclude_resolved:
                continue
            if not candidate.name.startswith("RefAmb_"):
                continue
            if not (candidate / "omnisearch_trajectories.jsonl").exists():
                continue
            matches.append(candidate)
    matches.sort()
    return matches


def load_source_subset(dataset_path: Path, source: str) -> list[dict]:
    rows = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("source") == source:
                rows.append(obj)
    return rows


def build_config(
    model: str,
    source: str,
    train_path: Path,
    Config,
    explicit_config_path: Path | None = None,
    save_note_suffix: str = "",
    output_dir: Path | None = None,
) -> "Config":
    config_path = resolve_config_path(model, source, explicit_config_path=explicit_config_path)
    override = {
        "source": source,
        "split": ["task_balanced_analysis_subset"],
        "save_note": resolve_save_note(model, source, save_note_suffix=save_note_suffix),
        "data_dir": str((PROJECT_ROOT / "data/datasets").resolve()),
        "save_dir": str((PROJECT_ROOT / "data/result").resolve()),
        "omni_prompt_version": "system_react_prompt",
        "omni_disturb_trace_filename": "disturb_trace_with_disturb.jsonl",
    }
    if output_dir is not None:
        resolved_output_dir = str(output_dir.resolve())
        override["output_dir"] = resolved_output_dir
        override["save_dir"] = resolved_output_dir
        override["save_new_dir"] = False
    runtime_gpu_id = resolve_runtime_gpu_id()
    if runtime_gpu_id is not None:
        override["gpu_id"] = runtime_gpu_id
    return Config(config_path, config_dict=override)


def cleanup_runtime_objects(pipeline=None, retriever=None, generator=None) -> None:
    if pipeline is not None and hasattr(pipeline, "_shutdown_roi_worker"):
        try:
            pipeline._shutdown_roi_worker()
        except Exception as exc:
            print(f"[Cleanup] ROI worker shutdown skipped: {exc}")

    for attr_owner, attr_name in (
        (pipeline, "generator"),
        (pipeline, "retriever"),
        (generator, "model"),
        (generator, "tokenizer"),
        (retriever, "model"),
        (retriever, "tokenizer"),
    ):
        if attr_owner is None or not hasattr(attr_owner, attr_name):
            continue
        try:
            delattr(attr_owner, attr_name)
        except Exception:
            pass

    del pipeline
    del retriever
    del generator
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as exc:
        print(f"[Cleanup] torch cleanup skipped: {exc}")


def main():
    args = parse_args()
    source = args.source
    model = args.model
    Config, Dataset, get_generator, get_retriever, IIGCR_Pipeline = import_flashrag_modules()
    config = build_config(
        model,
        source,
        args.train_path,
        Config,
        explicit_config_path=args.config_path,
        save_note_suffix=args.save_note_suffix,
        output_dir=args.output_dir,
    )

    if args.output_dir is not None:
        # Fixed output directories should only use the current trajectory file for incremental reruns.
        args.disable_auto_history_resume = True

    subset_rows = load_source_subset(args.train_path, source)
    if not subset_rows:
        print(f"No RefAmb train samples found for source `{source}` in {args.train_path}.")
        return

    trajectory_path = Path(config["save_dir"]) / "omnisearch_trajectories.jsonl"
    existing_ids = set()
    auto_history_ids = set()

    if not args.disable_auto_history_resume:
        history_dirs = discover_historical_result_dirs(
            save_note=resolve_save_note(model, source, save_note_suffix=args.save_note_suffix),
            roots=iter_history_roots(args.history_root),
            exclude_dirs=[Path(config["save_dir"])],
        )
        if history_dirs:
            for history_dir in history_dirs:
                auto_history_ids.update(load_existing_ids(history_dir / "omnisearch_trajectories.jsonl"))
            print(
                f"Auto history resume for `{source}`: found {len(auto_history_ids)} existing ids "
                f"across {len(history_dirs)} historical runs."
            )
        else:
            print(f"Auto history resume for `{source}`: no matching historical runs found.")

    if not args.disable_incremental:
        existing_ids = load_existing_ids(trajectory_path)
        if existing_ids:
            print(
                f"Incremental rerun for `{source}`: found {len(existing_ids)} existing ids "
                f"in {trajectory_path}."
            )
        else:
            print(f"Incremental rerun for `{source}`: no existing trajectory ids found.")

    skip_ids = existing_ids | auto_history_ids
    if skip_ids:
        original_count = len(subset_rows)
        subset_rows = [row for row in subset_rows if row.get("id") not in skip_ids]
        print(
            f"Incremental filtering for `{source}`: "
            f"{original_count - len(subset_rows)} items already have records, "
            f"{len(subset_rows)} items remain."
        )
    else:
        print(f"Incremental filtering for `{source}`: no existing ids found.")

    if not subset_rows:
        print("No remaining items to run. Exit without launching IIGCR_Pipeline.")
        return

    if args.limit is not None:
        if args.limit <= 0:
            print(f"`--limit` must be positive, got {args.limit}. Exit without launching IIGCR_Pipeline.")
            return
        original_count = len(subset_rows)
        subset_rows = subset_rows[: args.limit]
        print(
            f"Applied --limit={args.limit}: running {len(subset_rows)} of {original_count} remaining items."
        )

    test_data = Dataset(config=config, data=subset_rows)

    print("Begin loading generator...")
    generator = get_generator(config)
    print("Finished!")
    if "text_retriever_config" in config:
        print("Skip eager retriever loading; IIGCR_Pipeline will lazy-load configured retrievers.")
        retriever = None
    else:
        print("Begin loading retriever...")
        retriever = get_retriever(config)
        print("Finished!")

    pipeline = None
    try:
        pipeline = IIGCR_Pipeline(config=config, retriever=retriever, generator=generator)
        pipeline.run(test_data, do_eval=not args.disable_eval)
    finally:
        cleanup_runtime_objects(pipeline=pipeline, retriever=retriever, generator=generator)


if __name__ == "__main__":
    main()
