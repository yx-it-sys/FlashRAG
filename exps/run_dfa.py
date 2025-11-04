import argparse
from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import NFAPipeline
from flashrag.prompt import MMPromptTemplate

def main():
    config_dict = {
        "data_dir": "lmms-lab/OK-VQA",
        # "image_path": "data/",
        "index_path": "indexes/bm25",
        "corpus_path": "indexes/bm25/corpus.jsonl",
        "generator_model": "Qwen2.5-VL-7B-Instruct",
        # "generator_model_path": "data",
        "retrieval_method": "bm25",
        "metrics": ["em", "f1", "acc"],
        "retrieval_topk": 1,
        "save_intermediate_data": True,
    }

    config = Config("my_config.yaml", config_dict=config_dict)
    all_split = get_dataset(config)
    test_data = all_split["dev"]
    data = list(test_data)
    
    prompt_templete = MMPromptTemplate(
        config=config
    )

    pipeline = NFAPipeline(config, prompt_template=prompt_templete)
    output_dataset = pipeline.run(test_data, do_eval=True)

    
if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Run the OmniSearch pipeline.")
    # parser.add_argument("--model_path", type=str, help="Path to config file")
    # args = parser.parse_args()   
    main()

