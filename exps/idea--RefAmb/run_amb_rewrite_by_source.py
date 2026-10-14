import argparse
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

from run_refamb_by_source import (
    REFAMB_TRAIN_PATH,
    build_config,
    discover_historical_result_dirs,
    import_flashrag_modules,
    iter_history_roots,
    load_existing_ids,
    load_existing_ids_from_result_dir,
    load_source_subset,
    normalize_model_tag,
    preload_cuda_visible_devices,
    resolve_config_path,
    resolve_resume_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AmbRewritePipeline on the RefAmb train subset for one source.")
    parser.add_argument(
        "--source",
        required=True,
        choices=["crag", "infoseek", "mcsearch", "oven"],
        help="RefAmb source to run.",
    )
    parser.add_argument(
        "--model",
        default="default",
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
    parser.add_argument(
        "--prompt-opt",
        action="store_true",
        help="Use AmbRewritePipeline.prompt_opt_run() with the hard-coded query-grounded prompt baseline.",
    )
    return parser.parse_args()


def resolve_save_note(model: str, source: str, prompt_opt: bool = False) -> str:
    if model == "gpt":
        config_path = Path(resolve_config_path(model, source))
        import yaml

        with config_path.open("r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}
        generator_model = config_data.get("generator_model")
        if not generator_model:
            raise KeyError(f"`generator_model` is missing in config: {config_path}")
        save_note = f"refamb_{source}_{normalize_model_tag(str(generator_model))}_amb_rewrite_stage"
    else:
        save_note = f"refamb_{source}_{model}_amb_rewrite_stage" if model != "default" else f"refamb_{source}_amb_rewrite_stage"
    return f"{save_note}_prompt_opt" if prompt_opt else save_note


def build_amb_rewrite_config(model: str, source: str, Config, output_dir: Path | None = None, prompt_opt: bool = False):
    config = build_config(model, source, Config, output_dir=output_dir, prompt_opt=False)
    save_note = resolve_save_note(model, source, prompt_opt=prompt_opt)
    if hasattr(config, "final_config"):
        config.final_config["save_note"] = save_note
    try:
        config["save_note"] = save_note
    except Exception:
        pass
    return config


def main():
    args = parse_args()
    source = args.source
    model = args.model
    preload_cuda_visible_devices(model, source)
    Config, Dataset, get_generator, get_retriever, _, _ = import_flashrag_modules()
    from flashrag.pipeline.amb_rewrite_pipeline import AmbRewritePipeline

    config = build_amb_rewrite_config(model, source, Config, output_dir=args.output_dir, prompt_opt=args.prompt_opt)

    if args.output_dir is not None:
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
            save_note=resolve_save_note(model, source, prompt_opt=args.prompt_opt),
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
        print("No remaining items to run. Exit without launching AmbRewritePipeline.")
        return

    if args.limit is not None:
        if args.limit <= 0:
            print(f"`--limit` must be positive, got {args.limit}. Exit without launching AmbRewritePipeline.")
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
        print("Skip eager retriever loading; AmbRewritePipeline will lazy-load configured retrievers.")
        retriever = None
    else:
        print("Begin loading retriever...")
        retriever = get_retriever(config)
        print("Finished!")

    pipeline = AmbRewritePipeline(config=config, retriever=retriever, generator=generator)
    print(f"Using pipeline class: {pipeline.__class__.__module__}.{pipeline.__class__.__name__}")
    if args.prompt_opt:
        print("Running query-grounded prompt baseline via AmbRewritePipeline.prompt_opt_run().")
        pipeline.prompt_opt_run(test_data, do_eval=not args.disable_eval)
    else:
        pipeline.run(test_data, do_eval=not args.disable_eval)


if __name__ == "__main__":
    main()
