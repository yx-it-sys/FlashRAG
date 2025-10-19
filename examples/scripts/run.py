import argparse
import tomllib
import json
from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import OmniSearchIGPipeline, OmniSearchPipeline
from flashrag.prompt import MMPromptTemplate
from flashrag.uncertainty import DisturbImage

def load_prompt(prompt_name: str) -> str:
    with open("plain_prompt.toml", "rb") as f:
        prompts_data = tomllib.load(f)
    return prompts_data["system_prompts"][prompt_name]

def main(args):
    config_dict = {
        "data_dir": "data/",
        "image_path": "data/images/val2014",
        "index_path": "indexes/bm25",
        "corpus_path": "indexes/bm25/corpus.jsonl",
        "generator_model": "Qwen2.5-VL-7B-Instruct",
        "generator_model_path": args.model_path,
        "retrieval_method": "bm25",
        "metrics": ["em", "f1", "acc"],
        "retrieval_topk": 1,
        "save_intermediate_data": True,
    }

    config = Config("my_config.yaml", config_dict=config_dict)
    all_split = get_dataset(config)
    test_data = all_split["dev"]

    base_sys_prompt = load_prompt("multimodal_qa")
    
    prompt_templete = MMPromptTemplate(
        config=config,
        system_prompt=base_sys_prompt,
        user_prompt= "This is the input image. Now, please start following your instructions to answer the original question: {input_question}",
    )

    prediction_list = []
    with open("result/mnt/data/okvqa_dummy_entropy_plain_run/output.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            prediction_list.append(data.get("prediction"))

    # pipeline = OmniSearchPipeline(config, prompt_template=prompt_templete)
    # pipeline = OmniSearchIGPipeline(config, prompt_templete)
    # output_dataset = pipeline.naive_run(test_data, do_eval=False, generated_answers_list=prediction_list)
    # output_dataset = pipeline.run(test_data, do_eval=True, prompt_answer_path="result/mnt/data/okvqa_dummy_entropy_omni_run/output1.jsonl")

    uncertainty = DisturbImage(config, prompt_templete, method="cluster", threshold=0.7)
    uncertainty.generate_answers(test_data, step=5)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the OmniSearch pipeline.")
    parser.add_argument("--model_path", type=str, help="Path to config file")
    args = parser.parse_args()
    
    main(args)

