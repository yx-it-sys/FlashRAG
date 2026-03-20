from flashrag.config import Config
from flashrag.utils import get_dataset, get_generator, get_retriever
import os
import sys
import multiprocessing as mp


def _load_config_with_safe_vllm_memory(config_file):
    from flashrag.config import Config

    config_override = {}

    safe_util_env = os.getenv("VLLM_GPU_MEMORY_UTILIZATION")
    if safe_util_env is not None:
        config_override["vllm_gpu_memory_utilization"] = float(safe_util_env)

    safe_max_model_len_env = os.getenv("VLLM_MAX_MODEL_LEN")
    if safe_max_model_len_env is not None:
        config_override["max_model_len"] = int(safe_max_model_len_env)

    return Config(
        config_file,
        config_dict=config_override,
    )

def run_omnisearch_pipeline(config_file):
    from flashrag.pipeline import OmniSearchPipeline
    from flashrag.utils import get_dataset, get_generator

    config = _load_config_with_safe_vllm_memory(config_file)
    generator = get_generator(config)
    test_data = get_dataset(config)["validation"]
    retriever = get_retriever(config)
    pipeline = OmniSearchPipeline(config=config, retriever=retriever, generator=generator)
    output_dataset = pipeline.run(test_data, do_eval=True)

def main():
    run_omnisearch_pipeline("/home/you/FlashRAG/exps/baseline/omnisearch/my_config.yaml")
    return 0

if __name__ == "__main__":
    # Single-GPU mode only; never auto-select across devices.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
