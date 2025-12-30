import json
import string
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

model_name = "Qwen/Qwen2.5-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
sys_prompt = f"You are an expert evaluator for question answering systems. Your task is to determine if a Prediction correctly answers a question based on the Ground Truth.\n\nRules:\n1. The prediction is correct if it captures all the key information from the ground truth.\n2. The prediction is correct even if phrased differently as long as the meaning is the same.\n3. The prediction is incorrect if it contains incorrect information or is missing essential details.\nYour response should ONLY be yes or no."

# intermediate_logs = "baseline/omnisearch/result/crag_2025_12_17_16_06_experiment/intermediate_data.json"
# intermediate_logs = "baseline/mRAG/crag_whole_mRAG/intermediate_data.json"
# intermediate_logs = "baseline/naiveRAG/crag_whole_naive/intermediate_data.json"
import json
import string
from tqdm import tqdm

intermediate_logs = "baseline/omnisearch/result/crag_qwen_test200/intermediate_data.json"
print(f"Type: {intermediate_logs}")

with open(intermediate_logs, 'r', encoding='utf-8') as f:
    data = json.load(f)

total_score = 0.0

for item in tqdm(data, desc="Begin evaluating"):
    current_item_score = 0.0
    
    g_ans = "; ".join(item['golden_answers'])
    pred = item['output']['pred']
    metric_score = item['output']['metric_score']

    if pred == "":
        current_item_score = 0.0
    else:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Ground Truth:\n{g_ans}\n\nPrediction:{pred}\n\nYour judge response:"}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        raw_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        response = raw_response.lower().strip().strip(string.punctuation).strip()
        
        if "yes" == response:
            current_item_score = 1.0
        elif "no" == response:
            current_item_score = 0.0
        else:
            print(f"ERROR judge with GPT! Response cannot be parsed: {response}")
            current_item_score = 0.0
            
    metric_score['gpt'] = current_item_score
    item['output']['metric_score'] = metric_score
    total_score += current_item_score

with open(intermediate_logs, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"Evaluation finished. Scores saved to {intermediate_logs}. Total Score: {total_score}")
print(f"gpt: {total_score/len(data)}")

