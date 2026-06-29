import argparse
import json
import os
from pathlib import Path
import yaml
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


REFAMB_TRAIN_PATH = Path("/home/you/FlashRAG/exps/idea10/data/datasets/RefAmb/new/task_balanced.jsonl")
PROJECT_ROOT = Path("/home/you/FlashRAG/exps/idea10")
MODEL_TO_SOURCE_CONFIG = {
    "default": {
        "crag": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen2_5_7b/config_stage_crag.yaml",
        "infoseek": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/config_stage_infoseek.yaml",
        "oven": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/config_stage_infoseek.yaml",
        "mcsearch": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/config_stage_mcsearch.yaml",
    },
    "qwen2_5_7b": {
        "crag": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen2_5_7b/config_stage_crag.yaml",
        "infoseek": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen2_5_7b/config_stage_infoseek.yaml",
        "oven": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen2_5_7b/config_stage_oven.yaml",
        "mcsearch": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen2_5_7b/config_stage_mcsearch.yaml",
    },
    "mmsearch_r1_7b": {
        "crag": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/mmsearch_r1_7b/config_stage_crag.yaml",
        "infoseek": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/mmsearch_r1_7b/config_stage_infoseek.yaml",
        "oven": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/mmsearch_r1_7b/config_stage_oven.yaml",
        "mcsearch": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/mmsearch_r1_7b/config_stage_mcsearch.yaml",
    },
    "qwen3_5_scaling": {
        "crag": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen3_5_scaling/config_stage_crag.yaml",
        "infoseek": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen3_5_scaling/config_stage_infoseek.yaml",
        "oven": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen3_5_scaling/config_stage_oven.yaml",
        "mcsearch": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen3_5_scaling/config_stage_mcsearch.yaml",
    },
    "llava15_7b": {
        "crag": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/llava1_5_7b/config_stage_crag_llava15_7b.yaml",
        "infoseek": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/llava1_5_7b/config_stage_infoseek_llava15_7b.yaml",
        "oven": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/llava1_5_7b/config_stage_infoseek_llava15_7b.yaml",
        "mcsearch": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/llava1_5_7b/config_stage_mcsearch_llava15_7b.yaml",
    },
    "intervl3_5_8b": {
        "crag": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/intervl3_5_8b/config_stage_crag.yaml",
        "infoseek": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/intervl3_5_8b/config_stage_infoseek.yaml",
        "oven": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/intervl3_5_8b/config_stage_oven.yaml",
        "mcsearch": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/intervl3_5_8b/config_stage_mcsearch.yaml",
    },
    "gpt": {
    "crag": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/gpt/config_stage_crag.yaml",
    "infoseek": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/gpt/config_stage_infoseek.yaml",
    "oven": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/gpt/config_stage_oven.yaml",
    "mcsearch": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/gpt/config_stage_mcsearch.yaml",
    },
    "qwen3_vl_2b": {
    "crag": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen3_vl_2b/config_stage_crag.yaml",
    "infoseek": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen3_vl_2b/config_stage_infoseek.yaml",
    "oven": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen3_vl_2b/config_stage_oven.yaml",
    "mcsearch": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen3_vl_2b/config_stage_mcsearch.yaml",
    },
    "qwen3_vl_4b": {
    "crag": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen3_vl_4b/config_stage_crag.yaml",
    "infoseek": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen3_vl_4b/config_stage_infoseek.yaml",
    "oven": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen3_vl_4b/config_stage_oven.yaml",
    "mcsearch": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen3_vl_4b/config_stage_mcsearch.yaml",
    },

    "qwen3_vl_32b": {
    "crag": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen3_vl_32b/config_stage_crag.yaml",
    "infoseek": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen3_vl_32b/config_stage_infoseek.yaml",
    "oven": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen3_vl_32b/config_stage_oven.yaml",
    "mcsearch": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen3_vl_32b/config_stage_mcsearch.yaml",
    },
    "qwen3_vl_8b": {
    "crag": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen3_vl_8b/config_stage_crag.yaml",
    "infoseek": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen3_vl_8b/config_stage_infoseek.yaml",
    "oven": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen3_vl_8b/config_stage_oven.yaml",
    "mcsearch": "/home/you/FlashRAG/exps/idea10/configs/configs_refamb/qwen3_vl_8b/config_stage_mcsearch.yaml",
    },

}
MODEL_TO_SOURCE_SAVE_NOTE = {
    "default": {
        "crag": "refamb_crag_stage",
        "infoseek": "refamb_infoseek_stage",
        "oven": "refamb_oven_stage",
        "mcsearch": "refamb_mcsearch_stage",
    },
    "qwen2_5_7b": {
        "crag": "refamb_crag_stage",
        "infoseek": "refamb_infoseek_stage",
        "oven": "refamb_oven_stage",
        "mcsearch": "refamb_mcsearch_stage",
    },
    "mmsearch_r1_7b": {
        "crag": "refamb_crag_mmsearch_r1_7b_stage",
        "infoseek": "refamb_infoseek_mmsearch_r1_7b_stage",
        "oven": "refamb_oven_mmsearch_r1_7b_stage",
        "mcsearch": "refamb_mcsearch_mmsearch_r1_7b_stage",
    },
    "qwen3_5_scaling": {
        "crag": "refamb_crag_72b_stage",
        "infoseek": "refamb_infoseek_72b_stage",
        "oven": "refamb_oven_72b_stage",
        "mcsearch": "refamb_mcsearch_72b_stage",
    },
    "llava15_7b": {
        "crag": "refamb_crag_llava15_7b_stage",
        "infoseek": "refamb_infoseek_llava15_7b_stage",
        "oven": "refamb_oven_llava15_7b_stage",
        "mcsearch": "refamb_mcsearch_llava15_7b_stage",
    },
    "intervl3_5_8b": {
        "crag": "refamb_crag_intervl3_5_8b_stage",
        "infoseek": "refamb_infoseek_intervl3_5_8b_stage",
        "oven": "refamb_oven_intervl3_5_8b_stage",
        "mcsearch": "refamb_mcsearch_intervl3_5_8b_stage",
    },
    "qwen3_vl_2b": {
    "crag": "refamb_crag_qwen3_vl_2b_stage",
    "infoseek": "refamb_infoseek_qwen3_vl_2b_stage",
    "oven": "refamb_oven_qwen3_vl_2b_stage",
    "mcsearch": "refamb_mcsearch_qwen3_vl_2b_stage",
    },
    "qwen3_vl_4b": {
    "crag": "refamb_crag_qwen3_vl_4b_stage",
    "infoseek": "refamb_infoseek_qwen3_vl_4b_stage",
    "oven": "refamb_oven_qwen3_vl_4b_stage",
    "mcsearch": "refamb_mcsearch_qwen3_vl_4b_stage",
    },

    "qwen3_vl_32b": {
    "crag": "refamb_crag_qwen3_vl_32b_stage",
    "infoseek": "refamb_infoseek_qwen3_vl_32b_stage",
    "oven": "refamb_oven_qwen3_vl_32b_stage",
    "mcsearch": "refamb_mcsearch_qwen3_vl_32b_stage",
    },
    "qwen3_vl_8b": {
    "crag": "refamb_crag_qwen3_vl_8b_stage",
    "infoseek": "refamb_infoseek_qwen3_vl_8b_stage",
    "oven": "refamb_oven_qwen3_vl_8b_stage",
    "mcsearch": "refamb_mcsearch_qwen3_vl_8b_stage",
    },

}
MODEL_TO_SOURCE_RESUME_DIRS = {
    "default": {
        "crag": "/home/you/FlashRAG/exps/idea10/data/result/RefAmb_Qwen2.5-vl-7B1/RefAmb_2026_04_29_13_07_refamb_crag_stage",
        "infoseek": "/home/you/FlashRAG/exps/idea10/data/result/RefAmb_Qwen2.5-vl-7B1/RefAmb_2026_04_29_18_47_refamb_infoseek_stage",
        "oven": "/home/you/FlashRAG/exps/idea10/data/result/RefAmb_Qwen2.5-vl-7B1/RefAmb_2026_04_30_10_09_refamb_oven_stage",
        "mcsearch": "/home/you/FlashRAG/exps/idea10/data/result/RefAmb_Qwen2.5-vl-7B1/RefAmb_2026_04_30_11_10_refamb_mcsearch_stage",
    },
    "qwen2_5_7b": {
        "crag": "/home/you/FlashRAG/exps/idea10/data/result/RefAmb_2026_05_02_14_02_refamb_crag_stage",
        "infoseek": "/home/you/FlashRAG/exps/idea10/data/result/RefAmb_2026_05_02_13_26_refamb_infoseek_stage",
        "oven": "/home/you/FlashRAG/exps/idea10/data/result/RefAmb_Qwen2.5-vl-7B1/RefAmb_2026_04_30_10_09_refamb_oven_stage",
        "mcsearch": "/home/you/FlashRAG/exps/idea10/data/result/RefAmb_Qwen2.5-vl-7B1/RefAmb_2026_04_30_11_10_refamb_mcsearch_stage",
    },
    "intervl3_5_8b": {},
    "llava15_7b": {},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OmniSearchPipeline on the RefAmb train subset for one source.")
    parser.add_argument(
        "--source",
        required=True,
        choices=sorted(MODEL_TO_SOURCE_CONFIG["default"].keys()),
        help="RefAmb source to run.",
    )
    parser.add_argument(
        "--model",
        default="default",
        choices=sorted(MODEL_TO_SOURCE_CONFIG.keys()),
        help="Config group / generator variant to use.",
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=REFAMB_TRAIN_PATH,
        help="Path to RefAmb train.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional existing or target stage directory. When set, reuse this directory in place and do not create a new timestamped result subdirectory.",
    )
    parser.add_argument(
        "--disable-incremental",
        action="store_true",
        help="Do not skip items that already appear in the trajectory file.",
    )
    parser.add_argument(
        "--disable-resume",
        action="store_true",
        help="Do not skip items that already appear in the configured historical result directory.",
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


def resolve_config_path(model: str, source: str) -> str:
    return MODEL_TO_SOURCE_CONFIG[model][source]


def normalize_model_tag(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_").replace(".", "_")


def resolve_save_note(model: str, source: str) -> str:
    if model == "gpt":
        config_path = Path(resolve_config_path(model, source))
        with config_path.open("r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}
        generator_model = config_data.get("generator_model")
        if not generator_model:
            raise KeyError(f"`generator_model` is missing in config: {config_path}")
        return f"refamb_{source}_{normalize_model_tag(str(generator_model))}_stage"
    return MODEL_TO_SOURCE_SAVE_NOTE[model][source]


def resolve_resume_dir(model: str, source: str) -> Path | None:
    raw_path = MODEL_TO_SOURCE_RESUME_DIRS.get(model, {}).get(source)
    if not raw_path:
        return None
    return Path(raw_path)


def preload_cuda_visible_devices(model: str, source: str) -> None:
    return


def import_flashrag_modules():
    from flashrag.config import Config
    from flashrag.dataset.dataset import Dataset
    from flashrag.utils import get_generator, get_retriever
    from flashrag.pipeline.omni_pipeline import OmniSearchPipeline
    from flashrag.pipeline.mmsearch_r1_pipeline import MMSearchR1Pipeline

    return Config, Dataset, get_generator, get_retriever, OmniSearchPipeline, MMSearchR1Pipeline


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


def load_existing_ids_from_result_dir(result_dir: Path) -> set[str]:
    trajectory_path = result_dir / "omnisearch_trajectories.jsonl"
    return load_existing_ids(trajectory_path)


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


def build_config(model: str, source: str, Config, output_dir: Path | None = None) -> "Config":
    config_path = resolve_config_path(model, source)
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    override = {
        "source": source,
        "split": ["task_balanced_analysis_subset"],
        "save_note": resolve_save_note(model, source),
        "data_dir": str((PROJECT_ROOT / "data/datasets").resolve()),
        "save_dir": str((PROJECT_ROOT / "data/result").resolve()),
        "roi_preprocess_config": {"enabled": False},
        # Keep GPU selection consistent with launcher script (GPU_ID/CUDA_VISIBLE_DEVICES).
        "gpu_id": gpu_id if gpu_id != "" else None,
    }
    if output_dir is not None:
        override["save_dir"] = str(output_dir.resolve())
        override["save_new_dir"] = False
    return Config(config_path, config_dict=override)


def main():
    args = parse_args()
    source = args.source
    model = args.model
    preload_cuda_visible_devices(model, source)
    Config, Dataset, get_generator, get_retriever, OmniSearchPipeline, MMSearchR1Pipeline = import_flashrag_modules()
    config = build_config(model, source, Config, output_dir=args.output_dir)

    if args.output_dir is not None:
        # When resuming in-place, only the current output directory should be consulted.
        args.disable_resume = True
        args.disable_auto_history_resume = True

    subset_rows = load_source_subset(args.train_path, source)
    if not subset_rows:
        print(f"No RefAmb train samples found for source `{source}` in {args.train_path}.")
        return

    trajectory_path = Path(config["save_dir"]) / "omnisearch_trajectories.jsonl"
    existing_ids = set()
    resume_ids = set()
    auto_history_ids = set()

    if not args.disable_resume:
        resume_dir = resolve_resume_dir(model, source)
        if resume_dir is None:
            print(f"Resume filter for `{source}`: no configured historical result directory.")
        else:
            resume_ids = load_existing_ids_from_result_dir(resume_dir)
            print(
                f"Resume filter for `{source}`: found {len(resume_ids)} existing ids "
                f"in {resume_dir / 'omnisearch_trajectories.jsonl'}."
            )

    if not args.disable_auto_history_resume:
        history_dirs = discover_historical_result_dirs(
            save_note=resolve_save_note(model, source),
            roots=iter_history_roots(args.history_root),
            exclude_dirs=[Path(config["save_dir"])],
        )
        if history_dirs:
            for history_dir in history_dirs:
                auto_history_ids.update(load_existing_ids_from_result_dir(history_dir))
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

    skip_ids = existing_ids | resume_ids | auto_history_ids
    if skip_ids:
        original_count = len(subset_rows)
        subset_rows = [row for row in subset_rows if row.get("id") not in skip_ids]
        print(
            f"Resume/incremental filtering for `{source}`: "
            f"{original_count - len(subset_rows)} items already have records, "
            f"{len(subset_rows)} items remain."
        )
    else:
        print(f"Resume/incremental filtering for `{source}`: no existing ids found.")

    if not subset_rows:
        print("No remaining items to run. Exit without launching OmniSearchPipeline.")
        return

    if args.limit is not None:
        if args.limit <= 0:
            print(f"`--limit` must be positive, got {args.limit}. Exit without launching OmniSearchPipeline.")
            return
        original_count = len(subset_rows)
        subset_rows = subset_rows[: args.limit]
        print(
            f"Applied --limit={args.limit}: running {len(subset_rows)} of {original_count} remaining items."
        )

    test_data = Dataset(config=config, data=subset_rows)

    if model == "mmsearch_r1_7b":
        print("Skip eager generator loading; MMSearchR1Pipeline loads the official model itself.")
        generator = None
    else:
        print("Begin loading generator...")
        generator = get_generator(config)
        print("Finished!")

    if "text_retriever_config" in config:
        print("Skip eager retriever loading; OmniSearchPipeline will lazy-load configured retrievers.")
        retriever = None
    else:
        print("Begin loading retriever...")
        retriever = get_retriever(config)
        print("Finished!")

    pipeline_cls = MMSearchR1Pipeline if model == "mmsearch_r1_7b" else OmniSearchPipeline
    pipeline = pipeline_cls(config=config, retriever=retriever, generator=generator)
    print(f"Using pipeline class: {pipeline.__class__.__module__}.{pipeline.__class__.__name__}")
    if model != "mmsearch_r1_7b" and (
        pipeline.__class__.__module__ != "flashrag.pipeline.omni_pipeline" or hasattr(pipeline, "_log_module_output")
    ):
        raise RuntimeError(
            "run_refamb_by_source.py must use the plain OmniSearchPipeline, but an EAO/VORS-like "
            f"pipeline was instantiated: {pipeline.__class__.__module__}.{pipeline.__class__.__name__}"
        )
    pipeline.run(test_data, do_eval=not args.disable_eval)


if __name__ == "__main__":
    main()
