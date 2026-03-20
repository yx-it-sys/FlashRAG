import os
import sys
import multiprocessing as mp

def _cleanup_runtime(generator=None, force_shutdown=False):
    try:
        if generator is not None:
            model = getattr(generator, "model", None)
            llm_engine = getattr(model, "llm_engine", None) if model is not None else None
            if llm_engine is not None:
                engine_core = getattr(llm_engine, "engine_core", None)
                if engine_core is not None and hasattr(engine_core, "shutdown"):
                    if force_shutdown:
                        engine_core.shutdown()
                    else:
                        # Try soft shutdown first
                        try:
                            engine_core.shutdown()
                        except:
                            pass
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
            # More aggressive memory cleanup
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Additional cleanup
            torch.cuda.ipc_collect()
    except Exception:
        pass

    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass

    # Additional memory optimization
    try:
        import gc
        gc.collect()
    except Exception:
        pass


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

def _load_pointwise_bin_prompt_template(config):
    from flashrag.prompt import MMPromptTemplate

    user_prompt = """
Passage: {reference}\n
Text Query: {question}\n
Does the passage answer the multimodal query? Only answer 'Yes' or 'No'
"""
    return MMPromptTemplate(
        config,
        user_prompt=user_prompt
    )

def _load_pointwise_qlm_prompt_template(config):
    from flashrag.prompt import MMPromptTemplate
    user_prompt = """
Passage: {reference}\n
Please write a question based on this passage.\n
 {question}
"""
    return MMPromptTemplate(
        config,
        user_prompt=user_prompt
    )

def _load_listwise_prompt_template(config):
    from flashrag.prompt import MMPromptTemplate
    user_prompt = """
You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query.
I will provide several passages with identifiers [1]...[len(docs)].
Query: {reference}
"""
    return MMPromptTemplate(
        config,
        user_prompt=user_prompt
    )


def pointwise_reranking(
    config_file="my_config.yaml",
    retrieval_results_path="data/result/infoseek_v9/phase2_image_caption_retrieval/final_retrieval_results.jsonl",
    method="qlm",
):
    from flashrag.pipeline.mm_pipeline import BaselineMMPipeline
    from flashrag.utils import get_dataset, get_generator

    config = _load_config_with_safe_vllm_memory(config_file)
    pipeline = None
    retry_count = 0

    try:
        generator = get_generator(config)
        test_data = get_dataset(config)["validation"]

        if method == "bin":
            prompt_template = _load_pointwise_bin_prompt_template(config)
            pipeline = BaselineMMPipeline(config=config, generator=generator, prompt_template=prompt_template)
            return pipeline.pointwise_bin_reranking(
                dataset=test_data,
                retrieval_results_path=retrieval_results_path,
            )

        else:
            prompt_template = _load_pointwise_qlm_prompt_template(config)
            pipeline = BaselineMMPipeline(config=config, generator=generator, prompt_template=prompt_template)
            return pipeline.pointwise_qlm_reranking(
                dataset=test_data,
                retrieval_results_path=retrieval_results_path,
            )

    except Exception as e:
        retry_count += 1
        _cleanup_runtime(getattr(pipeline, "generator", None), force_shutdown=True)
        print(f"[run_baseline][ERROR] pointwise_reranking failed (method={method}): {e}")
        raise

    finally:
        # Only clean up if we're not going to retry
        _cleanup_runtime(getattr(pipeline, "generator", None))

def listwise_reranking(
    config_file="my_config.yaml",
    retrieval_results_path="data/result/infoseek_v9/phase2_image_caption_retrieval/final_retrieval_results.jsonl",        
):
    from flashrag.pipeline.mm_pipeline import BaselineMMPipeline
    from flashrag.utils import get_dataset, get_generator

    config = _load_config_with_safe_vllm_memory(config_file)
    pipeline = None
    retry_count = 0

    try:
        generator = get_generator(config)
        test_data = get_dataset(config)["validation"]

        prompt_template = _load_listwise_prompt_template(config)
        pipeline = BaselineMMPipeline(config=config, generator=generator, prompt_template=prompt_template)
        return pipeline.listwise_reranking(
            dataset=test_data,
            retrieval_results_path=retrieval_results_path,
        )

    except Exception as e:
        retry_count += 1
        _cleanup_runtime(getattr(pipeline, "generator", None), force_shutdown=True)
        print(f"[run_baseline][ERROR] listwise_reranking failed: {e}")
        raise

    finally:
        # Only clean up if we're not going to retry
        _cleanup_runtime(getattr(pipeline, "generator", None))

def pairwise_reranking(
    config_file="my_config.yaml",
    retrieval_results_path="data/result/infoseek_v9/phase2_image_caption_retrieval/final_retrieval_results.jsonl",        
):
    from flashrag.pipeline.mm_pipeline import BaselineMMPipeline
    from flashrag.utils import get_dataset, get_generator

    config = _load_config_with_safe_vllm_memory(config_file)
    pipeline = None
    retry_count = 0

    try:
        generator = get_generator(config)
        test_data = get_dataset(config)["validation"]

        pipeline = BaselineMMPipeline(config=config, generator=generator, prompt_template=None)
        return pipeline.pairwise_reranking(
            dataset=test_data,
            retrieval_results_path=retrieval_results_path,
        )

    except Exception as e:
        retry_count += 1
        _cleanup_runtime(getattr(pipeline, "generator", None), force_shutdown=True)
        print(f"[run_baseline][ERROR] pairwise_reranking failed: {e}")
        raise

    finally:
        # Only clean up if we're not going to retry
        _cleanup_runtime(getattr(pipeline, "generator", None))

def setwise_reranking(
    config_file="my_config.yaml",
    retrieval_results_path="data/result/infoseek_v9/phase2_image_caption_retrieval/final_retrieval_results.jsonl",        
):
    from flashrag.pipeline.mm_pipeline import BaselineMMPipeline
    from flashrag.utils import get_dataset, get_generator

    config = _load_config_with_safe_vllm_memory(config_file)
    pipeline = None
    retry_count = 0

    try:
        generator = get_generator(config)
        test_data = get_dataset(config)["validation"]

        pipeline = BaselineMMPipeline(config=config, generator=generator, prompt_template=None)
        return pipeline.setwise_reranking(
            dataset=test_data,
            retrieval_results_path=retrieval_results_path,
        )

    except Exception as e:
        retry_count += 1
        _cleanup_runtime(getattr(pipeline, "generator", None), force_shutdown=True)
        print(f"[run_baseline][ERROR] setwise_reranking failed: {e}")
        raise

    finally:
        # Only clean up if we're not going to retry
        _cleanup_runtime(getattr(pipeline, "generator", None))

def main():
    setwise_reranking()
    return 0

if __name__ == "__main__":
    # Single-GPU mode only; never auto-select across devices.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()