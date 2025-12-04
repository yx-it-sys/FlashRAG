import tomllib
import json
import os
from flashrag.config import Config
from flashrag.utils import get_dataset, get_retriever
from detective_agent import Detective 
from flashrag.prompt import MMPromptTemplate, PromptTemplate
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

def load_prompt(prompt_name: str) -> str:
    with open("omni_prompt.toml", "rb") as f:
        prompts_data = tomllib.load(f)
    return prompts_data["system_prompts"][prompt_name]

def main():
    config_dict = {
        "dataset_path": "data/datasets/okvqa",
        "image_path": "data/datasets/okvqa/images",
        "index_path": "data/indexes/e5/e5_flat_inner.index",
        "corpus_path": "data/indexes/wiki18_100w.jsonl",
        "generator_model_path": "data/models/Qwen2.5-VL-7B-Instruct",
        "retrieval_method": "e5",
        "metrics": ["em", "f1", "acc"],
        "retrieval_topk": 5,
        "save_intermediate_data": True,
    }

    config = Config("my_config.yaml", config_dict=config_dict)
    all_split = get_dataset(config)
    test_data = all_split["dev"]
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    retriever = get_retriever(config)
    
    pipeline = Detective(model=model, processor=processor, max_loop=5, retriever=retriever, config=config)
    output_dataset = pipeline.run(test_data, do_eval=True)

    
if __name__ == "__main__":    
    main()

