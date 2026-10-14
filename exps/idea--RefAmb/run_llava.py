import os
import json
from pathlib import Path

for k in [
    "http_proxy", "https_proxy",
    "HTTP_PROXY", "HTTPS_PROXY",
    "all_proxy", "ALL_PROXY"
]:
    os.environ.pop(k, None)

# vLLM defaults to fork in some paths, which breaks once CUDA was touched
# earlier in the parent process.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from flashrag.config import Config
from flashrag.utils import get_dataset, get_generator, get_retriever
from flashrag.pipeline import OmniSearchPipeline

EXISTING_TRAJECTORY_PATH = Path(
    "/home/you/FlashRAG/exps/idea10/data/result/crag_mm_2026_04_experiment/omnisearch_trajectories.jsonl"
)


def load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    existing_ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            item_id = record.get("id")
            if item_id:
                existing_ids.add(item_id)
    return existing_ids


def main():
    config = Config("/home/you/FlashRAG/exps/idea10/configs/config_llava15_7b.yaml")
    all_split = get_dataset(config)
    configured_splits = config["split"]
    test_data = None
    active_split = None
    for split_name in configured_splits:
        split_data = all_split.get(split_name)
        if split_data is not None:
            test_data = split_data
            active_split = split_name
            break
    if test_data is None:
        raise KeyError(
            f"None of the configured splits are available in dataset: {configured_splits}"
        )
    print(f"Using dataset split from config: {active_split}")
    existing_ids = load_existing_ids(EXISTING_TRAJECTORY_PATH)
    if existing_ids:
        original_count = len(test_data)
        test_data.data = [item for item in test_data.data if item.id not in existing_ids]
        print(
            f"Incremental rerun enabled: {original_count - len(test_data)} items already "
            f"exist in {EXISTING_TRAJECTORY_PATH}, {len(test_data)} items remain to run."
        )
    else:
        print("Incremental rerun enabled: no existing trajectory ids found, running full split.")

    if len(test_data) == 0:
        print("No remaining items to run. Exit without launching OmniSearchPipeline.")
        return

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
    pipeline = OmniSearchPipeline(config=config, retriever=retriever, generator=generator)
    pipeline.run(test_data, do_eval=True)


if __name__ == "__main__":
    main()
