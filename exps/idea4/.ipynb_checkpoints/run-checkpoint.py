from flashrag.evaluator import Evaluator
from flashrag.config import Config
from flashrag.utils import get_dataset
from executor import DFAExecutor
import json

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
        "retrieval_topk": 10,
        "save_intermediate_data": True,
    }
    config = Config("my_config.yaml", config_dict=config_dict)
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    prompt_path = "prompts/draft_plan.toml"
    
    all_split = get_dataset(config)
    test_data = all_split["dev"]
    
    evaluator = Evaluator(config)
    executor = DFAExecutor(config=config, prompts_path=prompt_path, model_name=model_name)
    
    output_file_path = "output.jsonl"

    prediction_list = []

    existing_keys = set()
    with open(output_file_path, "r", encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            if "question" in data:
                existing_keys.add(data["question"])

    with open(output_file_path, "a", encoding='utf-8') as f:
        for item in test_data:
            if item.question in existing_keys:
                continue
            pred, logs = executor.serial_execute(item)
            # prediction_list.append(pred)
            data = {'question': item.question, 'golden_answer': item.golden_answers, 'prediction': pred, 'logs': logs}
            f.write(json.dumps(data, ensure_ascii=False)+'\n')
    
    all_prediction_map = {}
    with open(output_file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            all_prediction_map[data['question']] = data['prediction']
    
    prediction_list = [all_prediction_map.get(item.question) for item in test_data]
    if None in prediction_list:
        print("警告：部分测试数据在输出文件中未找到对应的预测结果。")
    test_data.update_output("pred", prediction_list)
    test_data = evaluate(evaluator, test_data, do_eval=True, pred_process_func=None)                 



    
if __name__ == "__main__":    
    main()

