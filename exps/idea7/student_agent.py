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
        response = qwen2_generate(self.model, self.processor, messages)
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
    
    def generate(self, question, image, plan):
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
        response = qwen2_generate(self.model, self.processor, messages)
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
        conversation_num, max_turns = 0, 5
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
                        {"type": "text", "text": f"Here are retrieved information:{retrieval_content}"}
                    ]
                })
                response = qwen2_generate(self.model, self.processor, messages)
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
                
            elif action_type == "Text Retrieval":
                print("<Text Retrieval>")
                query_list = self.query_rewrite(image, action)
                retrieval_content = []
                for query in query_list:
                    search_text = self.retriever.search_by_text(query)
                    retrieval_content.append("\n\n".join([f"Doc{i+1}:\n{text}" for i, text in enumerate(search_text)]))
                retrieval_content = "\n\n".join(retrieval_content)
                print(f"Retrieval Content: {retrieval_content}")
                logs.append({"text_ret": retrieval_content[:200]})
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Here are retrieved information:{retrieval_content}"}
                    ]
                })
                response = qwen2_generate(self.model, self.processor, messages)
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
            elif action_type == "Final Answer":
                final_answer = action
                logs.append({"current_final_answer": final_answer})
                break
        print(f"<answer>\n{final_answer}\n</answer>")

        return final_answer, logs

    def parse_action(self, text):
        """
        解析文本中的 JSON 输出，提取 Action 类型（Text/Image Retrieval 或 Final Answer）。
        """
        if not text:
            return {"type": None, "content": None, "raw_line": ""}

        result = {
            "type": None,
            "content": None,
            "raw_line": ""
        }

        # 1. 尝试提取 JSON 字符串
        # 策略 A: 优先匹配 Markdown 代码块 ```json ... ``` 或 ``` ... ```
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
        
        if json_match:
            json_str = json_match.group(1)
        else:
            # 策略 B: 如果没有代码块，尝试寻找最外层的花括号 { ... }
            # 利用非贪婪匹配或寻找第一个 { 和最后一个 }
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = text[start_idx : end_idx + 1]
            else:
                # 无法提取到类似 JSON 的结构
                print(f"Error: No JSON found in text: {text[-100:]}...") # 打印末尾部分用于调试
                result["raw_line"] = text.strip().split('\n')[-1] # 保留最后一行作为原始记录
                return result

        result["raw_line"] = json_str

        # 2. 解析 JSON 并映射字段
        try:
            # 清理可能存在的双大括号（如果是 prompt template 格式残留）
            # 如果确定模型输出是标准 JSON，可以去掉 replace
            if "{{" in json_str:
                json_str = json_str.replace("{{", "{").replace("}}", "}")
                
            data = json.loads(json_str)

            # --- Case 3: Final Answer ---
            if "final_answer" in data:
                result["type"] = "Final Answer"
                result["content"] = data["final_answer"]

            # --- Case 1 & 2: Tool Calls ---
            elif "tool_call" in data:
                tool_info = data["tool_call"]
                tool_name = tool_info.get("tool")
                query = tool_info.get("query")

                if tool_name == "Text Retrieval":
                    result["type"] = "Text Retrieval"
                    result["content"] = query
                
                elif tool_name == "Image Retrieval":
                    result["type"] = "Image Retrieval"
                    # Image Retrieval 的 query 为 None，这里可以设为 None 或空字符串，视下游需求而定
                    result["content"] = None 

        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {e}")
            # 解析失败，保持 type 为 None
        except Exception as e:
            print(f"Unexpected Error parsing action: {e}")

        print(f"Result: {result}")
        return result 