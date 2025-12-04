import json
import re
import ast
from json_repair import repair_json
from transformers import AutoTokenizer, AutoModelForCausalLM

def extract_json(response: str) -> dict | None:
    pattern = r"JSON:\s*(\{[\s\S]+\})"
    match = re.search(pattern, response)
    if match:
        json_string = match.group(1)
        try:
            parsed_json = json.loads(json_string)
        except json.JSONDecodeError as e:
            print("⚠️ LLM output is not valid JSON. Attempting to repair...")
            print(json_string)        
            try:
                repaired_json_string = repair_json(response)
                parsed_json = json.loads(repaired_json_string)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"❌ Failed to parse JSON even after repair. Error: {e}")
                return None
        return parsed_json
    else:
        print(f"❌ Failed to parse JSON: {response}")
        return None
                    
def extract_json_for_assessment(llm_output: str):
    start_index = llm_output.find('{')
    end_index = llm_output.rfind('}')

    if start_index == -1 or end_index == -1 or end_index < start_index:
        print("Error: Could not find a JSON block enclosed in {} braces.")
        return None

    json_str_candidate = llm_output[start_index : end_index + 1]

    try:
        parsed_json = json.loads(json_str_candidate)
        return parsed_json
    except json.JSONDecodeError as e:
        print(f"Standard `json.loads` failed: {e}")
        print("Attempting to repair with `json_repair` library...")

        try:
            repaired_json_string = repair_json(json_str_candidate)
            parsed_json = json.loads(repaired_json_string)
            print("Successfully repaired and parsed JSON.")
            return parsed_json
        except Exception as e:
            print(f"Error: JSON repair also failed. The string might be too malformed. Details: {e}")
            return None

def parse_response(response: str):
    if "JSON:" not in response:
        return []
    list_part = response.split("JSON:")[1].strip()
    lines = list_part.splitlines()
    parsed_items = [re.sub(r"^\d+\.\s*", "", line).strip() for line in lines if line.strip()]

    return parsed_items


def parse_action(response: str):
    pattern = r"^\s*Next Action:\s*(\w+):\s*(.*)$"
    match = re.search(pattern, response, re.MULTILINE | re.IGNORECASE)

    if match:
        action_type = match.group(1)
        action_value = match.group(2)

        return action_type, action_value
    else:
        print(f"Fail to parse planning! Return None.")
        return None, None

def extract_refine(response: str):
    result = {
        "entities": [],
        "reformed_query": ""
    }

    # 1. 提取 Entities
    # 使用正则查找 Entities: 后的 [...] 部分
    # re.DOTALL 允许 . 匹配换行符，防止列表跨行时匹配失败
    entities_pattern = r"Entities:\s*(\[.*?\])"
    entities_match = re.search(entities_pattern, response, re.IGNORECASE | re.DOTALL)
    
    if entities_match:
        raw_list_str = entities_match.group(1)
        try:
            # ast.literal_eval 可以安全地将字符串形式的 Python 列表转换为实际列表
            # 它比 json.loads 更宽容，支持单引号和双引号
            result["entities"] = ast.literal_eval(raw_list_str)
        except (ValueError, SyntaxError):
            print("Warning: Failed to parse entities list syntax.")
            result["entities"] = []
            # 如果解析失败，可以在这里通过正则硬提取作为备选方案

    # 2. 提取 Reformed Query
    # 查找 Reformed Query: 后的引号内容
    # 兼容双引号 "..." 和单引号 '...'
    query_pattern = r"Reformed Query:\s*([\"'])(.*?)\1"
    query_match = re.search(query_pattern, response, re.IGNORECASE | re.DOTALL)
    
    if query_match:
        # group(2) 是引号内部的实际内容
        result["reformed_query"] = query_match.group(2)

    return result["entities"], result["reformed_query"]
    
def chat_with_qwen(model, tokenizer, messages, type, enable_thinking=True):
        if type == "qwen2":
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=32768)
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            return {"thinking_content": None, "content": response}
        
        elif type == "qwen3":
            enable_thinking = True if enable_thinking else False
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking
            )
            model_inputs = tokenizer([inputs], return_tensors="pt").to(model.device)
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=32768
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

            try:
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0

            thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            return {"thinking_content": thinking_content, "content": content}