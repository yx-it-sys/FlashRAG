import tomllib
from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import OmniSearchPipeline, IterativePipeline

def main():
    config_dict = {
        "dataset_path": "/mnt/data/you/datasets/infoseek_val",
        "image_path": "/mnt/data/you/datasets/infoseek_val/infoseek_val_images",
        "index_path": "/mnt/data/you/datasets/infoseek_val/indexes/bge-large-en-v1.5_Flat.index",
        "corpus_path": "/mnt/data/you/datasets/infoseek_val/corpus.jsonl",
        "generator_model_path": "/mnt/data/you/modelscope/Qwen2.5-VL-7B-Instruct",
        "retrieval_method": "bge",
        "metrics": ["em", "f1", "acc"],
        "retrieval_topk": 2,
        "save_intermediate_data": False,
    }

    config = Config("my_config.yaml", config_dict=config_dict)
    all_split = get_dataset(config)
    test_data = all_split["validation"]
    

    # pipeline = IterativePipeline(config)
    pipeline = OmniSearchPipeline(config=config)
    # pipeline = SelfRAGPipeline(config)
    # pipeline = SelfAskPipeline(config)
    output_dataset = pipeline.run(test_data, do_eval=True)
    
if __name__ == "__main__":    
    main()

