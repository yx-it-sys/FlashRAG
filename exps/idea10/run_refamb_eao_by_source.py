import argparse
import json
import os
from pathlib import Path

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
SOURCE_CHOICES = ["crag", "infoseek", "mcsearch", "oven"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ATL_CI_Ablation on RefAmb subset for one source.")
    parser.add_argument("--source", required=True, choices=SOURCE_CHOICES, help="RefAmb source to run.")
    parser.add_argument("--model", required=True, help="Model tag for save_note naming.")
    parser.add_argument("--train-path", required=True, type=Path, help="Path to RefAmb subset jsonl.")
    parser.add_argument(
        "--config-dir",
        required=True,
        type=Path,
        help="Directory containing config_stage_<source>.yaml (typically a snapshot dir).",
    )
    parser.add_argument(
        "--disable-incremental",
        action="store_true",
        help="Do not skip items that already appear in the trajectory file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of remaining items to run after filtering.",
    )
    return parser.parse_args()


def resolve_config_path(config_dir: Path, source: str) -> str:
    config_path = config_dir / f"config_stage_{source}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return str(config_path)


def resolve_save_note(model: str, source: str) -> str:
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
    from flashrag.pipeline import ATL_CI_Ablation

    return Config, Dataset, get_generator, get_retriever, ATL_CI_Ablation


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


def build_config(model: str, source: str, train_path: Path, config_dir: Path, Config) -> "Config":
    config_path = resolve_config_path(config_dir, source)
    override = {
        "source": source,
        "split": ["task_balanced_analysis_subset"],
        "save_note": resolve_save_note(model, source),
        "data_dir": str((PROJECT_ROOT / "data/datasets").resolve()),
        "save_dir": str((PROJECT_ROOT / "data/result").resolve()),
        "omni_disturb_trace_filename": "disturb_trace_with_disturb.jsonl",
    }
    runtime_gpu_id = resolve_runtime_gpu_id()
    if runtime_gpu_id is not None:
        override["gpu_id"] = runtime_gpu_id
    return Config(config_path, config_dict=override)


def main():
    args = parse_args()
    source = args.source
    model = args.model
    Config, Dataset, get_generator, get_retriever, ATL_CI_Ablation = import_flashrag_modules()
    config = build_config(model, source, args.train_path, args.config_dir, Config)

    subset_rows = load_source_subset(args.train_path, source)
    if not subset_rows:
        print(f"No RefAmb train samples found for source `{source}` in {args.train_path}.")
        return

    trajectory_path = Path(config["save_dir"]) / "omnisearch_trajectories.jsonl"
    existing_ids = set()
    if not args.disable_incremental:
        existing_ids = load_existing_ids(trajectory_path)
        if existing_ids:
            print(
                f"Incremental rerun for `{source}`: found {len(existing_ids)} existing ids "
                f"in {trajectory_path}."
            )
        else:
            print(f"Incremental rerun for `{source}`: no existing trajectory ids found.")

    if existing_ids:
        original_count = len(subset_rows)
        subset_rows = [row for row in subset_rows if row.get("id") not in existing_ids]
        print(
            f"Incremental filtering for `{source}`: "
            f"{original_count - len(subset_rows)} items already have records, "
            f"{len(subset_rows)} items remain."
        )
    else:
        print(f"Incremental filtering for `{source}`: no existing ids found.")

    if not subset_rows:
        print("No remaining items to run. Exit without launching ATL_CI_Ablation.")
        return

    if args.limit is not None:
        if args.limit <= 0:
            print(f"`--limit` must be positive, got {args.limit}. Exit without launching ATL_CI_Ablation.")
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
        print("Skip eager retriever loading; ATL_CI_Ablation will lazy-load configured retrievers.")
        retriever = None
    else:
        print("Begin loading retriever...")
        retriever = get_retriever(config)
        print("Finished!")

    pipeline = ATL_CI_Ablation(config=config, retriever=retriever, generator=generator)
    if hasattr(pipeline, "warmup_roi_preprocess"):
        warmed_up = pipeline.warmup_roi_preprocess()
        if warmed_up:
            print("Persistent GroundingDINO ROI worker warmed up at startup.")
    pipeline.run(test_data, do_eval=True)


if __name__ == "__main__":
    main()
