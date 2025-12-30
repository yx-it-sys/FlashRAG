from flashrag.config import Config
from flashrag.utils import get_dataset
from utils import CRAGSearch, Qwen2Generator
from flashrag.pipeline import MMSequentialPipeline

def main():
    config_dict = {
        "dataset_path": "data/datasets/crag",
        "image_path": "data/datasets/crag/images",
        "index_path": "data/indexes/e5/e5_flat_inner.index",
        "corpus_path": "data/indexes/wiki18_100w.jsonl",
        "generator_model_path": "Qwen/Qwen2.5-VL-7B-Instruct",
        "retrieval_method": "e5",
        "metrics": ["em", "f1", "acc"],
        "retrieval_topk": 5,
        "save_intermediate_data": True,
    }
    config = Config("my_config.yaml", config_dict=config_dict)
    all_split = get_dataset(config)
    test_data = all_split["test"]

    generator = Qwen2Generator()
    retriever = CRAGSearch(top_k=config['retrieval_topk'])
    pipeline = MMSequentialPipeline(config=config, retriever=retriever, generator=generator)
    output_dataset = pipeline.self_run(test_data, search_type="image", do_eval=True)


    
if __name__ == "__main__":    
    main()

