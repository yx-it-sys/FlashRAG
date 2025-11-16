from flashrag.evaluator import Evaluator
from flashrag.config import Config
from flashrag.utils import get_dataset
from executor import DFAExecutor

def evaluate(evaluator, dataset, do_eval=True, pred_process_func=None):
    """The evaluation process after finishing overall generation"""

    if pred_process_func is not None:
        dataset = pred_process_func(dataset)

    if do_eval:
        eval_result = evaluator.evaluate(dataset)
        print(eval_result)
    return 

def main():
    config_dict = {
        "dataset_path": "data/datasets/hotpotqa",
        "image_path": "data/datasets/okvqa/images/val2014",
        "index_path": "data/indexes/e5/e5_flat_inner.index",
        "corpus_path": "data/indexes/wiki18_100w.jsonl",
        "generator_model_path": "data/models/Qwen2.5-7B-Instruct",
        "retrieval_method": "e5",
        "metrics": ["em", "f1", "acc"],
        "retrieval_topk": 5,
        "save_intermediate_data": True,
    }
    config = Config("my_config.yaml", config_dict=config_dict)
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    prompt_path = "prompts/meta_plan.toml"
    
    all_split = get_dataset(config)
    test_data = all_split["dev"]
    
    evaluator = Evaluator(config)
    executor = DFAExecutor(prompt_path=prompt_path, config=config)
    
    prediction_list = []
    for item in test_data:
        print(f"Question:{item.question}")
        print(f"Golden Answer: {item.golden_answers}")
        pred = executor.serial_execute(item)
        print(f"prediction:{pred}")
        prediction_list.append(pred)
    
    test_data.update_output("pred", prediction_list)
    test_data = evaluate(evaluator, test_data, do_eval=True, pred_process_func=None)                 



    
if __name__ == "__main__":    
    main()

