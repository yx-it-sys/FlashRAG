import json
import string
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ================= 配置区域 =================
model_name = "Qwen/Qwen2.5-7B-Instruct"
# 请在此处修改 test.jsonl 的实际路径
test_file_path = "idea7/data/datasets/crag/test.jsonl" 

# intermediate_logs = "baseline/omnisearch/result/crag_2025_12_17_16_06_experiment/intermediate_data.json"
intermediate_logs = "idea7/result/crag_instructor_220/intermediate_logs1.jsonl"
# ===========================================

# 1. 加载模型
print(f"Loading model: {model_name}...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
sys_prompt = f"You are an expert evaluator for question answering systems. Your task is to determine if a Prediction correctly answers a question based on the Ground Truth.\n\nRules:\n1. The prediction is correct if it captures all the key information from the ground truth.\n2. The prediction is correct even if phrased differently as long as the meaning is the same.\n3. The prediction is incorrect if it contains incorrect information or is missing essential details.\nYour response should ONLY be yes or no."

print(f"Type: {intermediate_logs}")

# 2. 预加载 test.jsonl (用于后续补充缺失的 golden_answers)
golden_answers_map = {}
if os.path.exists(test_file_path):
    print(f"Loading golden answers from {test_file_path}...")
    with open(test_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                gid = obj.get('interaction_id', obj.get('id'))
                g_ans_list = obj.get('golden_answers', obj.get('golden_answer'))
                
                if gid is not None and g_ans_list is not None:
                    golden_answers_map[str(gid)] = g_ans_list
            except json.JSONDecodeError:
                pass
else:
    print(f"Warning: {test_file_path} not found. Evaluation may fail if logs rely on external golden answers.")

# 3. 加载待评估数据 (修复点：针对 .jsonl 逐行读取)
data = []
print(f"Reading data from {intermediate_logs}...")
with open(intermediate_logs, 'r', encoding='utf-8') as f:
    # 判断文件扩展名，如果是 .jsonl 则逐行读取，否则尝试一次性读取
    if intermediate_logs.endswith('.jsonl'):
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Skipping invalid json line: {line[:50]}...")
    else:
        # 兼容旧的 json 列表格式
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            # 如果 json.load 失败，尝试回退到逐行读取（防止后缀名不准）
            f.seek(0)
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))

total_score = 0.0
evaluated_count = 0

# 4. 开始评估循环
for item in tqdm(data, desc="Begin evaluating"):
    current_item_score = 0.0
    
    # --- 检查并获取 golden_answers ---
    golden_answers = item.get('golden_answers') or item.get('golden_answer')
    
    if golden_answers is None:
        current_id = str(item.get('interaction_id', item.get('id')))
        if current_id in golden_answers_map:
            golden_answers = golden_answers_map[current_id]
            item['golden_answers'] = golden_answers # 回填
        else:
            # 如果没有标准答案，跳过评估，给 0 分或不做处理
            # 这里选择 continue，不计入 total_score 的分母
            # print(f"Error: Missing golden_answers for ID {current_id}, skipping.")
            # item['gpt'] = 0.0 # 也可以标记为 0
            continue 

    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    g_ans = "; ".join(golden_answers)
    
    # 获取预测结果
    pred = item.get('prediction', item.get('output', {}).get('pred', ""))
    
    if not pred:
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
        
        if "yes" in response.split(): 
            current_item_score = 1.0
        elif "yes" == response:
            current_item_score = 1.0
        elif "no" in response.split() or "no" == response:
            current_item_score = 0.0
        else:
            if response.startswith("yes"):
                current_item_score = 1.0
            elif response.startswith("no"):
                current_item_score = 0.0
            else:
                print(f"ERROR judge with GPT! Response cannot be parsed: {response}")
                current_item_score = 0.0
            
    # 保存分数
    item['gpt'] = current_item_score
    total_score += current_item_score
    evaluated_count += 1

# 5. 保存结果 (修复点：保持与输入文件一致的格式 .jsonl)
print(f"Saving results to {intermediate_logs}...")
with open(intermediate_logs, 'w', encoding='utf-8') as f:
    if intermediate_logs.endswith('.jsonl'):
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        # 如果是 .json 结尾，保存为列表格式
        json.dump(data, f, ensure_ascii=False, indent=4)

print(f"Evaluation finished. Total valid items: {evaluated_count}. Total Score: {total_score}")
if evaluated_count > 0:
    print(f"gpt: {total_score/evaluated_count}")
else:
    print("gpt: 0.0 (No valid data evaluated)")