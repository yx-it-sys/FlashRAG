import os
import multiprocessing as mp
import tomllib


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


def _load_prompt_template(config, prompt_file_key):
    from flashrag.prompt import MMPromptTemplate

    with open(config[prompt_file_key], 'rb') as f:
        prompt_dict = tomllib.load(f)
        return MMPromptTemplate(
            config,
            system_prompt=prompt_dict['system_prompt'],
            user_prompt=prompt_dict['user_prompt'],
        )


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


def phase1_clue_mining(config_file='my_config.yaml'):
    from flashrag.utils import get_dataset, get_generator, get_retriever
    from flashrag.pipeline.mm_pipeline import MMCluePipeline

    generator = None
    try:
        config = _load_config_with_safe_vllm_memory(config_file)
        retriever = get_retriever(config)
        generator = get_generator(config)
        test_data = get_dataset(config)["validation"]
        visual_clue_prompt_template = _load_prompt_template(config, 'visual_clue_prompt_file')
        pipeline = MMCluePipeline(
            config=config,
            visual_clue_prompt_template=visual_clue_prompt_template,
            retriever=retriever,
            generator=generator,
        )
        return pipeline.get_clue(test_data)
    finally:
        _cleanup_runtime(generator)


def phase2_image_retrieval(config_file='my_config.yaml'):
    from flashrag.utils import get_dataset, get_retriever
    from flashrag.pipeline.mm_pipeline import MMCluePipeline

    config = _load_config_with_safe_vllm_memory(config_file)
    retriever = get_retriever(config)
    test_data = get_dataset(config)["validation"]
    visual_clue_prompt_template = _load_prompt_template(config, 'visual_clue_prompt_file')
    pipeline = MMCluePipeline(
        config=config,
        visual_clue_prompt_template=visual_clue_prompt_template,
        retriever=retriever,
        generator=None,
    )
    pipeline.img_retrieval(test_data)

def phase2_caption_retrieval(config_file='phase4_config.yaml', image_phase_dir="data/result/infoseek_v9/phase2_image_retrieval", caption_phase_dir="data/result/infoseek_v9/phase2_image_caption_retrieval"):
    from flashrag.utils import get_dataset, get_retriever, get_generator
    from flashrag.pipeline.mm_pipeline import MMCluePipeline

    config = _load_config_with_safe_vllm_memory(config_file)
    retriever = get_retriever(config)
    generator = get_generator(config)
    test_data = get_dataset(config)["validation"]
    visual_clue_prompt_template = _load_prompt_template(config, 'visual_clue_prompt_file')
    pipeline = MMCluePipeline(
        config=config,
        visual_clue_prompt_template=visual_clue_prompt_template,
        retriever=retriever,
        generator=generator,
    )
    pipeline.caption_retrieval(test_data, image_phase_dir, caption_phase_dir)

def phase3_reranking(
    config_file='my_config.yaml',
    clue_path="data/result/infoseek_v9/phase1_clue_mining/clue.jsonl",
    retrieval_results_path="data/result/infoseek_v9/phase2_image_caption_retrieval/final_retrieval_results.jsonl",
):
    from flashrag.utils import get_retriever, get_generator
    from flashrag.pipeline.mm_pipeline import MMCluePipeline

    config = _load_config_with_safe_vllm_memory(config_file)
    retriever = get_retriever(config)
    generator = get_generator(config)
    visual_clue_prompt_template = _load_prompt_template(config, 'visual_clue_prompt_file')
    pipeline = MMCluePipeline(
        config=config,
        visual_clue_prompt_template=visual_clue_prompt_template,
        retriever=retriever,
        generator=generator,
    )
    return pipeline.my_reranking_v1(clue_path, retrieval_results_path)
    # return pipeline.quadric_reranking(clue_path=clue_path, retrieval_results_path=retrieval_results_path)


def phase4_query_gen_retrieval(
    config_file='phase4_config.yaml',
    reranked_results_path="data/result/infoseek_v9/phase3_reranking_llm_compute_prune5/reranked_results.jsonl",
):
    from flashrag.utils import get_dataset, get_generator, get_retriever
    from flashrag.pipeline.mm_pipeline import MMCluePipeline

    generator = None
    try:
        config = _load_config_with_safe_vllm_memory(config_file)
        retriever = get_retriever(config)
        generator = get_generator(config)
        test_data = get_dataset(config)["validation"]
        
        query_generate_prompt_template = _load_prompt_template(config, 'query_generate_prompt_file')
        
        pipeline = MMCluePipeline(
            config=config,
            query_generate_prompt_template=query_generate_prompt_template,
            retriever=retriever,
            generator=generator,
        )
        return pipeline.query_gen_retrieval(test_data, reranked_results_path=reranked_results_path)
    finally:
        _cleanup_runtime(generator)

def phase5_rag_generation(dataset, config_file='phase4_config.yaml', reranked_results_path="data/result/infoseek_v9/phase3_reranking_llm_compute_prune5/reranked_results.jsonl"):
    from flashrag.utils import get_generator
    from flashrag.pipeline.mm_pipeline import MMCluePipeline

    generator = None
    try:
        config = _load_config_with_safe_vllm_memory(config_file)
        generator = get_generator(config)
        
        naive_prompt_template = _load_prompt_template(config, 'naive_prompt_file')
        rag_prompt_template = _load_prompt_template(config, 'rag_prompt_file')
        
        pipeline = MMCluePipeline(
            config=config,
            naive_prompt_template=naive_prompt_template,
            rag_prompt_template=rag_prompt_template,
            generator=generator,
        )
        return pipeline.rag_run(dataset, reranked_results_path=reranked_results_path)
    finally:
        _cleanup_runtime(generator)

def main():
    # phase3_reranking()
    run_phase1 = False
    run_phase2 = False
    run_phase3 = True
    run_phase4 = False

    if run_phase1:
        phase1_clue_mining()

    if run_phase2:
        # phase2_image_retrieval()
        phase2_caption_retrieval()

    if run_phase3:
        phase3_reranking()

    if run_phase4:
        dataset = phase4_query_gen_retrieval()
        phase5_rag_generation(dataset, config_file='phase4_config.yaml', reranked_results_path="data/result/infoseek_v9/phase3_reranking_llm_compute_prune5/reranked_results.jsonl")

    return 0

if __name__ == "__main__":
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()