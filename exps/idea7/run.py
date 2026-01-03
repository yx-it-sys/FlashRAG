from flashrag.config import Config
from flashrag.utils import get_dataset
from utils import CRAGSearch
from instructor_agent import Instructor
from student_agent import Student
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

def main():
    config_dict = {
        "dataset_path": "data/datasets/crag",
        "image_path": "data/datasets/crag/images",
        "index_path": "data/indexes/e5/e5_flat_inner.index",
        "corpus_path": "data/indexes/wiki18_100w.jsonl",
        "generator_model_path": "Qwen/Qwen2.5-VL-7B-Instruct",
        "retrieval_method": "e5",
        "metrics": ["em", "f1", "acc"],
        "retrieval_topk": 10,
        "save_intermediate_data": True,
    }
    config = Config("my_config.yaml", config_dict=config_dict)
    all_split = get_dataset(config)
    test_data = all_split["test"]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    retriever = CRAGSearch(top_k=config['retrieval_topk'])
    student = Student(model=model, processor=processor,retriever=retriever, config=config)
    pipeline = Instructor(config=config, model=model, processor=processor, student=student, retriever=retriever)
    output_dataset = pipeline.run(test_data, do_eval=True)
    toten_usage = generator.get_total_usage()
    with open("records.txt", "a", encoding="utf-8") as f:
        f.write(f"Total Usage: Input={toten_usage['total_input']}, Output={toten_usage['total_output']}, All={toten_usage['total_all']}\n")
        
if __name__ == "__main__":    
    main()

