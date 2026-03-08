import argparse
import multiprocessing as mp
import os
import tomllib

from PIL import Image


def _cleanup_runtime(generator=None):
    try:
        if generator is not None:
            model = getattr(generator, "model", None)
            llm_engine = getattr(model, "llm_engine", None) if model is not None else None
            if llm_engine is not None:
                engine_core = getattr(llm_engine, "engine_core", None)
                if engine_core is not None and hasattr(engine_core, "shutdown"):
                    engine_core.shutdown()
                if hasattr(llm_engine, "__del__"):
                    llm_engine.__del__()
    except Exception:
        pass

    try:
        import gc
        import torch

        del generator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


def _load_config_with_safe_vllm_memory(config_file):
    from flashrag.config import Config

    safe_util = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.80"))
    safe_max_model_len = int(os.getenv("VLLM_MAX_MODEL_LEN", "4096"))
    return Config(
        config_file,
        config_dict={
            "vllm_gpu_memory_utilization": safe_util,
            "max_model_len": safe_max_model_len,
        },
    )


class OmniPromptTemplateAdapter:
    def __init__(self, config, prompt_path):
        self.config = config
        with open(prompt_path, "rb") as f:
            prompt_cfg = tomllib.load(f)
        self.system_prompt = prompt_cfg["system_prompts"]["multimodal_qa_omni"]

    def get_string(self, item, _config=None):
        question = item.question if getattr(item, "question", None) is not None else item.text
        image_id = getattr(item, "image_id", None)
        if image_id is None:
            image_id = item.id

        image_path = os.path.join(self.config["dataset_image_dir"], f"{image_id}.jpg")
        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Input Question: {question}"},
                    {"type": "image", "image": image},
                ],
            },
        ]
        return messages


def run_omni(
    config_file="phase4_config.yaml",
    split="validation",
    do_eval=True,
    uncertainty_type=None,
    prompt_path=None,
):
    from flashrag.utils import get_dataset, get_generator
    from flashrag.pipeline.omni_pipeline import OmniSearchPipeline

    generator = None
    try:
        config = _load_config_with_safe_vllm_memory(config_file)

        if prompt_path is None:
            prompt_path = config["omni_vqa_prompt_path"]

        prompt_template = OmniPromptTemplateAdapter(config, prompt_path)
        generator = get_generator(config)

        all_split = get_dataset(config)
        if split not in all_split:
            raise ValueError(f"Split '{split}' not found. Available splits: {list(all_split.keys())}")
        dataset = all_split[split]

        pipeline = OmniSearchPipeline(
            config=config,
            prompt_template=prompt_template,
            retriever=None,
            generator=generator,
        )

        return pipeline.run(
            dataset,
            do_eval=do_eval,
            pred_process_func=None)
    finally:
        _cleanup_runtime(generator)


def main():
    parser = argparse.ArgumentParser(description="Run OmniSearch Pipeline on infoseek_val")
    parser.add_argument("--config", default="phase4_config.yaml", help="Path to config yaml")
    parser.add_argument("--split", default="validation", help="Dataset split, e.g. validation")
    parser.add_argument("--no-eval", action="store_true", help="Disable evaluation")
    parser.add_argument(
        "--uncertainty-type",
        default=None,
        choices=["entropy", "ig_text"],
        help="Uncertainty type",
    )
    parser.add_argument(
        "--prompt-path",
        default=None,
        help="Path to omni prompt toml; defaults to config omni_vqa_prompt_path or baseline path",
    )
    args = parser.parse_args()

    run_omni(
        config_file=args.config,
        split=args.split,
        do_eval=not args.no_eval,
        uncertainty_type=args.uncertainty_type,
        prompt_path=args.prompt_path,
    )


if __name__ == "__main__":
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
