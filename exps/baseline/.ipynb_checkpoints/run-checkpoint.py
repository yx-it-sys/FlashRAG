import tomllib
import json
import os
from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import OmniSearchQAPipeline, IterativePipeline, SelfAskPipeline, SelfRAGPipeline, FLAREPipeline
from flashrag.prompt import MMPromptTemplate, PromptTemplate
from flashrag.uncertainty import DisturbImage

def load_prompt(prompt_name: str) -> str:
    with open("omni_prompt.toml", "rb") as f:
        prompts_data = tomllib.load(f)
    return prompts_data["system_prompts"][prompt_name]

def main():
    config_dict = {
        "dataset_path": "data/datasets/hotpotqa",
        "image_path": "data/datasets/okvqa/images/val2014",
        "index_path": "data/indexes/e5/e5_flat_inner.index",
        "corpus_path": "data/indexes/wiki18_100w.jsonl",
        "generator_model_path": "data/models/Qwen2.5-7B-Instruct",
        "retrieval_method": "e5",
        "metrics": ["em", "f1", "acc"],
        "retrieval_topk": 2,
        "save_intermediate_data": True,
    }

    config = Config("my_config.yaml", config_dict=config_dict)
    all_split = get_dataset(config)
    test_data = all_split["dev"]
    

    pipeline = IterativePipeline(config)
    # pipeline = OmniSearchQAPipeline(config=config)
    # pipeline = SelfRAGPipeline(config)
    # pipeline = SelfAskPipeline(config)
    output_dataset = pipeline.run(test_data, do_eval=True)

    # uncertainty = DisturbImage(config, prompt_templete, method="cluster", threshold=0.7)
    # chunk_size = 10

    # # blur_path = './results_disturb_images/gausian_blur/output.jsonl'
    # blur_path = './results_disturb_images/pepper_salt/output.jsonl'

    # if os.path.exists(blur_path):
    #     with open(blur_path, 'r') as f:
    #         done = {json.loads(line)["id"] for line in f}
    #     data = [d for d in data if str(d.id) not in done]
    
    # for i in range(0, len(data), chunk_size):
    #     chunk = data[i: i+chunk_size]
    #     uncertainty.generate_answers(chunk, step=5)

    
if __name__ == "__main__":    
    main()

