import json
import string
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from rouge import Rouge


cached_scores = {}
def calculate_rouge(scorer, pred, golden_answers):
        if (pred, tuple(golden_answers)) in cached_scores:
            return cached_scores[(pred, tuple(golden_answers))]
        output = {}
        for answer in golden_answers:
            if pred == "":
                pred = "I can't answer."
            scores = scorer.get_scores(pred, answer)
            for key in ["rouge-1", "rouge-2", "rouge-l"]:
                if key not in output:
                    output[key] = []
                output[key].append(scores[0][key]["f"])
        for k, v in output.items():
            output[k] = max(v)

        cached_scores[(pred, tuple(golden_answers))] = output
        return output
    
def main():
    intermediate_logs = [
        "baseline/omnisearch/result/crag_qwen_test200/intermediate_data.json"
        # "baseline/naiveRAG/crag_whole_naive/intermediate_data.json",
        # "baseline/mRAG/crag_whole_mRAG/intermediate_data.json",
        # "baseline/standardRAG/result/crag_text_retrieval/intermediate_data.json",
        # "baseline/standardRAG/result/crag_image_retrieval/intermediate_data.json",
        # "baseline/omnisearch/result/crag_2025_12_17_16_06_experiment/intermediate_data.json",
        # "idea7/result/crag_instructor_whole/intermediate_data.json",
    ]
    for log in intermediate_logs:
        print(f"Type: {log}")
        scorer = Rouge()
        with open(log, 'r', encoding='utf-8') as f:
            data = json.load(f)
            pred_list = []
            golden_answers_list = []
            for item in tqdm(data, desc="Begin evaluating"):
                pred = item['output']['pred']
                golden_answers_list.append(item['golden_answers'])
                pred_list.append(pred)
        
            metric_score_list = [
                calculate_rouge(scorer, pred, golden_answers)["rouge-l"]
                for pred, golden_answers in zip(pred_list, golden_answers_list)
            ]
            score = sum(metric_score_list) / len(metric_score_list)
        
        print(f"rouge-l: {score}")

if __name__ == "__main__":
    main()

