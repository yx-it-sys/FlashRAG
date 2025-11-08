import argparse
from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import DFAQAPipeline
from flashrag.prompt import MMPromptTemplate, PromptTemplate

def main():
    config_dict = {
        "dataset_path": "data/datasets/2wikimultihopqa",
        "image_path": "data/datasets/okvqa/images/val2014",
        "index_path": "data/indexes/e5_flat_inner.index",
        "corpus_path": "data/indexes/wiki18_100w.jsonl",
        "model2path": {"e5": "data/models/e5-base-v2"},
        "generator_model_path": "data/models/Qwen2.5-7B-Instruct",
        "retrieval_method": "e5",
        "metrics": ["em", "f1", "acc"],
        "retrieval_topk": 1,
        "save_intermediate_data": True,
    }

    config = Config("my_config.yaml", config_dict=config_dict)
    all_split = get_dataset(config)
    test_data = all_split["test"]
    
    # vqa_prompt_templete = MMPromptTemplate(
    #     config=config
    # )

    qa_prompt_pipeline = PromptTemplate(
        config=config
    )

    # pipeline = DFAVQAPipeline(config, prompt_template=vqa_prompt_templete)
    pipeline = DFAQAPipeline(config, prompt_template=qa_prompt_pipeline)
    output_dataset = pipeline.run(test_data, do_eval=True)

    
if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Run the OmniSearch pipeline.")
    # parser.add_argument("--model_path", type=str, help="Path to config file")
    # args = parser.parse_args()   
    main()

