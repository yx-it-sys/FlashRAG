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
        解析文本的最后一行，判断是否包含指定关键词，并提取 Text Retrieval 的查询内容。
        """
        if not text:
            return {"type": None, "content": None, "raw_line": ""}

        # 1. 提取最后一行
        # strip() 用于去除整个文本末尾可能存在的空行或空白字符
        lines = text.strip().split('\n')
        # 获取最后一行并去除首尾空格
        last_line = lines[-1].strip()

        result = {
            "type": None,          # 匹配到的类型
            "content": None,         # 提取到的 query (仅 Text Retrieval 有效)
            "raw_line": last_line  # 原始的最后一行文本
        }

        # 2. 判断关键词并执行提取逻辑
        
        # --- Case 1: Text Retrieval ---
        if "Text Retrieval" in last_line:
            result["type"] = "Text Retrieval"
            
            # 使用正则匹配冒号后面的内容
            # pattern 解释: 
            #   Text Retrieval  : 匹配关键词
            #   \s*:\s*         : 匹配冒号及其前后任意数量的空格
            #   (.*)            : 捕获组，匹配冒号后的所有内容
            match = re.search(r"Text Retrieval.*?[:：]\s*(.*)", last_line, re.IGNORECASE)
            
            if match:
                result["content"] = match.group(1).strip()
            else:
                # 如果存在关键词但没有冒号，这里视情况处理，或者设为空字符串
                result["content"] = ""

        # --- Case 2: Image Retrieval ---
        elif "Image Retrieval" in last_line:
            result["type"] = "Image Retrieval"

        # --- Case 3: Final Answer ---
        elif "Final Answer" in last_line:
            result["type"] = "Final Answer"
            match = re.search(r"Final Answer.*?[:：]\s*(.*)", last_line, re.IGNORECASE)
            
            if match:
                result["content"] = match.group(1).strip()
            else:
                # 如果存在关键词但没有冒号，这里视情况处理，或者设为空字符串
                result["content"] = ""

        return result
