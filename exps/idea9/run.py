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

def main():
    from flashrag.prompt import MMPromptTemplate
    from flashrag.config import Config
    from flashrag.utils import get_dataset, get_generator
    from flashrag.pipeline.mm_pipeline import MMCluePipeline

    generator = None
    try:
        config = Config("my_config.yaml")
        all_split = get_dataset(config)
        test_data = all_split["validation"]

        with open(config['visual_clue_prompt_file'], 'rb') as f:
            prompt_dict = tomllib.load(f)
            sys_prompt = prompt_dict['system_prompt']
            usr_prompt = prompt_dict['user_prompt']
        generator = get_generator(config)
        visual_clue_prompt_template = MMPromptTemplate(config, system_prompt=sys_prompt, user_prompt=usr_prompt)
        pipeline = MMCluePipeline(config=config, visual_clue_prompt_template=visual_clue_prompt_template, generator=generator)
        
        # Phase I: Clue Mining
        clue_list = pipeline.get_clue(test_data)
        # # Phase II: Image Retrieval
        # pipeline.img_retrieval(test_data)
        # # Phase III: Reranking
        # output_dir = config['output_dir'] if 'output_dir' in config and config['output_dir'] else config['save_dir']
        # clue_path = os.path.join(output_dir, "clue.jsonl")
        # retrieval_results_path = os.path.join(output_dir, "retrieval_results.jsonl")
        # pipeline.reranking(clue_path=clue_path, retrieval_results_path=retrieval_results_path)
        return clue_list
    finally:
        _cleanup_runtime(generator)

if __name__ == "__main__":
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()