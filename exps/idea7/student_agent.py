from utils import qwen2_generate
from flashrag.pipeline import BasicPipeline
from flashrag.utils import get_retriever
import tomllib
import re
import json

class Student(BasicPipeline):
    def __init__(self, model, processor, config, retriever=None):
        super().__init__(config)
        if retriever is None:
            retriever = get_retriever(config)
        self.retriever = retriever
        self.model = model
        self.processor = processor
        with open('prompts/student.toml', 'rb') as f:
            self.prompt = tomllib.load(f)
        with open('prompts/query_rewrite.toml', 'rb') as f:
            self.query_rewrite_prompt = tomllib.load(f)

    def query_rewrite(self, image, action_content):
        # 1. 构造 Prompt
        # 注意：请确保 self.query_rewrite_prompt['system_prompt'] 中的文本
        # 已经更新为要求输出 JSON 格式，并且包含了你在问题中提供的 JSON 示例。
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.query_rewrite_prompt['system_prompt']},
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.query_rewrite_prompt['user_prompt'].format(query=action_content)}
                ]
            }
        ]

        # 2. 调用模型生成
        response, entropy = qwen2_generate(self.model, self.processor, messages)
        print(f"Model Raw Response: {response}")
                
        json_data = None
        json_str = None
        
        # 策略 A: 优先尝试提取 Markdown 代码块 ```json ... ```
        # re.DOTALL 让 . 可以匹配换行符
        markdown_pattern = r"```json\s*(.*?)\s*```"
        match = re.search(markdown_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if match:
            json_str = match.group(1)
        else:
            # 策略 B: 如果没写 markdown，尝试寻找最外层的花括号 { ... }
            brace_pattern = r"\{.*\}"
            match = re.search(brace_pattern, response, re.DOTALL)
            if match:
                json_str = match.group(0)
                
        # 默认兜底：如果解析失败，使用原始 query 放入列表
        results = [action_content] 

        if json_str:
            try:
                # 尝试解析 JSON
                # strict=False 允许字符串中包含控制字符（提高容错率）
                data = json.loads(json_str, strict=False)
                
                # 1. 打印思维链（如果有的话），用于调试
                if "reasoning" in data:
                    print(f"[Rewrite Reasoning]: {data['reasoning']}")
                
                # 2. 提取 rewritten_query
                if "rewritten_query" in data:
                    extracted_queries = data["rewritten_query"]
                    
                    # 类型检查：确保它是一个列表
                    if isinstance(extracted_queries, list):
                        results = extracted_queries
                    elif isinstance(extracted_queries, str):
                        # 如果模型偶尔只生成了一个字符串，将其包裹为列表
                        results = [extracted_queries]
                else:
                    print("Warning: JSON parsed but key 'rewritten_query' missing.")
                    
            except json.JSONDecodeError as e:
                print(f"JSON Parsing Error: {e}")
                print(f"Problematic string: {json_str}")
        else:
            print("No JSON object found in response, using original query.")
        print(f"Rewrite Results: {results}")
        return results
    
    def generate(self, question, image, plan=None):
        if plan is None:
            plan = question
        logs = []
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt['system_prompt'].format(original_question=question, current_task=plan)},
                ],
            }
        ]
        response, entropy = qwen2_generate(self.model, self.processor, messages)
        print(f"Student first Response: {response}")
        logs.append({"student_first_response": response})
        messages.append({
            "role": "assistant",
            "content": [{
                "type": "text", "text": response
            }]
        })
        action = self.parse_action(response)
        action_type = action['type']
        action = action['content']
        conversation_num, max_turns = 0, 3
        final_answer = "I can't answer."
        while conversation_num < max_turns:
            # print("="*30)
            # print(f"STUDENT Prompt Contexts:\n{messages}")
            # print("="*30)
            if action_type == "Image Retrieval":
                print("<Image Retrieval>")
                search_text = self.retriever.search_by_image(image)
                retrieval_content = "\n\n".join([f"Doc{i+1}:\n{text}" for i, text in enumerate(search_text)])
                logs.append({"image_ret": retrieval_content[:200]})
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Here are retrieved information:{retrieval_content}."}
                    ]
                })
                response, entropy = qwen2_generate(self.model, self.processor, messages)
                print(f"Student response after image retrieval: {response}")
                logs.append({"student_response": response})
                action = self.parse_action(response)
                action_type = action['type']
                action = action['content']
                messages.append({
                    "role": "assistant",
                    "content": [{
                        "type": "text", "text": f"{action_type}: {action}"
                    }]
                })
                conversation_num += 1
                
            elif action_type == "Text Retrieval":
                print("<Text Retrieval>")
                # query_list = self.query_rewrite(image, action)
                query_list = [action]
                logs.append({"retriever_queries": query_list})
                retrieval_content = []
                seen_blocks = set()
                
                # 建议：如果 query_list 本身就有重复词，最好先在这里去重，节省 API 调用
                query_list = list(dict.fromkeys(query_list)) 
                
                for query in query_list:
                    search_text = self.retriever.search_by_text(query)
                    
                    block_str = "\n\n".join([f"Doc{i+1}:\n{text}" for i, text in enumerate(search_text)])
                    
                    if block_str and block_str not in seen_blocks:
                        retrieval_content.append(block_str)
                        seen_blocks.add(block_str)
                retrieval_content = "\n\n".join(retrieval_content)
                print(f"Retrieval Content: {retrieval_content}")
                logs.append({"text_ret": retrieval_content[:200]})
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Here are retrieved information:{retrieval_content}."}
                    ]
                })
                response, entropy = qwen2_generate(self.model, self.processor, messages)
                print(f"Student response after text retrieval: {response}")
                logs.append({"student_response": response})
                action = self.parse_action(response)
                action_type = action['type']
                action = action['content']
                messages.append({
                    "role": "assistant",
                    "content": [{
                        "type": "text", "text": f"{action_type}: {action}"
                    }]
                })
                conversation_num += 1
            elif action_type == "Final Answer":
                final_answer = action
                logs.append({"current_final_answer": final_answer})
                break
            else:
                break
        print(f"<answer>\n{final_answer}\n</answer>")

        return final_answer, logs


    def parse_action(self, text):
        """
        解析 LLM 生成的文本，提取 Action。
        兼容关键词被 markdown 符号包裹的情况 (如 **Text Retrieval**: )。
        """
        result = {
            "type": None,
            "content": text
        }
        
        if not text:
            return result

        # --- 策略：使用 \W* 忽略关键词前后的符号 ---
        # \W* 匹配零个或多个"非单词字符" (即忽略 *, [, ], ", ', 空格 等)
        # [::：] 匹配英文冒号或中文冒号

        # 1. 优先匹配 Final Answer (支持多行)
        # 匹配逻辑：找到 "Final Answer"，忽略后面的符号(如 **)，直到遇到冒号，然后捕获后面所有内容
        fa_match = re.search(r"Conclusion\W*[:：]\s*(.*)", text, re.IGNORECASE | re.DOTALL)
        if fa_match:
            result["type"] = "Final Answer"
            result["content"] = fa_match.group(1).strip()
            return result

        # 2. 匹配 Text Retrieval (单行)
        # 匹配逻辑：同上，找到 "Text Retrieval" 后紧接冒号的内容
        tr_match = re.search(r"Text Retrieval\W*[:：]\s*(.*)", text, re.IGNORECASE)
        if tr_match:
            result["type"] = "Text Retrieval"
            result["content"] = tr_match.group(1).strip()
            return result

        # 3. 匹配 Image Retrieval (通常无参数)
        # 匹配逻辑：只要文本中包含 "Image Retrieval" 这个短语（允许周围有符号），就命中
        # 使用 re.search 会自动忽略前面的符号（如 "**Image Retrieval"）
        if re.search(r"Image Retrieval", text, re.IGNORECASE):
            result["type"] = "Image Retrieval"
            result["content"] = None
            return result
        
        print("ERROR: No action type matched.")
        return result        

    def parse_action_json(self, text):
        """
        解析文本中的 JSON 输出，提取 Action 类型（Text/Image Retrieval 或 Final Answer）。
        兼容标准 JSON (null) 和 Python 字典格式 (None)。
        """
        if not text:
            return {"type": None, "content": None, "raw_line": ""}
    
        result = {
            "type": None,
            "content": None,
            "raw_line": ""
        }
    
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
        
        if json_match:
            json_str = json_match.group(1)
        else:
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = text[start_idx : end_idx + 1]
            else:
                print(f"Error: No JSON-like structure found.")
                result['type'] = "Final Answer"
                result["content"] = text.strip().split('\n')[-1]
                result["raw_line"] = text.strip().split('\n')[-1]
                return result
    
        result["raw_line"] = json_str
    
        # 2. 解析逻辑 (增强版)
        data = None
        
        fixed_json_str = json_str.replace("{{", "{").replace("}}", "}")
        
        # 使用正则精准替换值部分的 None (避免误伤文本中的单词)
        fixed_json_str = re.sub(r':\s*None\b', ': null', fixed_json_str)
        fixed_json_str = re.sub(r':\s*True\b', ': true', fixed_json_str)
        fixed_json_str = re.sub(r':\s*False\b', ': false', fixed_json_str)
    
        try:
            data = json.loads(fixed_json_str)
        except json.JSONDecodeError:
            # --- 尝试 2: 如果标准 JSON 解析失败，尝试作为 Python 字面量解析 ---
            # ast.literal_eval 可以完美处理 {'key': None} 这种 Python 格式
            try:
                # ast.literal_eval 对双花括号敏感，确保使用原始字符串或简单清理后的
                clean_str = json_str.replace("{{", "{").replace("}}", "}")
                data = ast.literal_eval(clean_str)
            except Exception as e:
                print(f"Parsing failed. Raw string: {json_str}\nError: {e}")
                return result
    
        # 3. 映射字段 (保持不变)
        if data:
            if "final_answer" in data:
                result["type"] = "Final Answer"
                result["content"] = data["final_answer"]
    
            elif "tool_call" in data:
                tool_info = data["tool_call"]
                tool_name = tool_info.get("tool")
                query = tool_info.get("query")
    
                if tool_name == "Text Retrieval":
                    result["type"] = "Text Retrieval"
                    result["content"] = query
                
                elif tool_name == "Image Retrieval":
                    result["type"] = "Image Retrieval"
                    result["content"] = query
                else:
                    result["type"] = "Final Answer"
                    result["content"] = "I can't answer"
    
        print(f"Result: {result}")
        return result