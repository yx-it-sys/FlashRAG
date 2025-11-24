from typing import List
from transformers import pipeline
import re
import json
import json_repair

def general_generate(messages, model, tokenizer):

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    generated_ids = [
        output_ids[len(input_ids)] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

def parse_json(text: str):
    pattern = r"\{[\s\S]*\}"
    match = re.search(pattern, text)
    if not match:
        print("❌ 未找到任何大括号 {} 包裹的内容")
        return None
    json_str = match.group(0)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        print(f"⚠️ 标准 JSON 解析失败，正在调用 json_repair 尝试修复...\n{json_str}")
        try:
            return json_repair.loads(json_str)
        except Exception as e:
            print(f"❌ json_repair 解析失败: {e}")
            return None