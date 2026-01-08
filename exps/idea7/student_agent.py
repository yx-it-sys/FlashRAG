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

        response, entropy = qwen2_generate(self.model, self.processor, messages)
             
        json_data = None
        json_str = None
        
        markdown_pattern = r"```json\s*(.*?)\s*```"
        match = re.search(markdown_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if match:
            json_str = match.group(1)
        else:
            brace_pattern = r"\{.*\}"
            match = re.search(brace_pattern, response, re.DOTALL)
            if match:
                json_str = match.group(0)
                
        results = [action_content] 

        if json_str:
            try:
                data = json.loads(json_str, strict=False)
                
                # # 1. 打印思维链（如果有的话），用于调试
                # if "reasoning" in data:
                #     print(f"[Rewrite Reasoning]: {data['reasoning']}")
                
                if "rewritten_query" in data:
                    extracted_queries = data["rewritten_query"]
                    
                    if isinstance(extracted_queries, list):
                        results = extracted_queries
                    elif isinstance(extracted_queries, str):
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
                    {"type": "text", "text": self.prompt['system_prompt'].format(current_task=plan)},
                ],
            }
        ]
        response, entropy = qwen2_generate(self.model, self.processor, messages)
        print(f"Student first Response: \n{response}")
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
            if action_type == "Image Retrieval":
                print("<Image Retrieval>")
                search_text = self.retriever.search_by_image(image)
                retrieval_content = "\n\n".join([f"Doc{i+1}:\n{text}" for i, text in enumerate(search_text)])
                print(f"Retrieval Content: {retrieval_content[:300]}")
                print("</Image Retrieval>")
                logs.append({"image_ret": retrieval_content[:300]})
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Here are retrieved information:{retrieval_content}."}
                    ]
                })
                response, entropy = qwen2_generate(self.model, self.processor, messages)
                print(f"Student response after image retrieval: \n{response}")
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
                print("<Rewrite>")
                query_list = self.query_rewrite(image, action)
                print("</Rewrite>")
                # query_list = [action]
                logs.append({"retriever_queries": query_list})
                retrieval_content = []
                seen_blocks = set()
                
                query_list = list(dict.fromkeys(query_list)) 
                
                for query in query_list:
                    search_text = self.retriever.search_by_text(query)
                    
                    block_str = "\n\n".join([f"Doc{i+1}:\n{text}" for i, text in enumerate(search_text)])
                    
                    if block_str and block_str not in seen_blocks:
                        retrieval_content.append(block_str)
                        seen_blocks.add(block_str)
                retrieval_content = "\n\n".join(retrieval_content)
                print(f"Retrieval Content: \n{retrieval_content[:300]}...")
                print("</Text Retrieval>")
                logs.append({"text_ret": retrieval_content[:300]})
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Here are retrieved information:{retrieval_content}."}
                    ]
                })
                response, entropy = qwen2_generate(self.model, self.processor, messages)
                print(f"Student response after text retrieval: \n{response}")
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
        result = {
            "type": None,
            "content": text
        }
        
        if not text:
            return result
            
        tr_match = re.search(r"Text Retrieval\W*[:：]\s*(.*)", text, re.IGNORECASE) 
        if tr_match:
            result["type"] = "Text Retrieval"
            result["content"] = tr_match.group(1).strip()
            return result
            
        if re.search(r"Image Retrieval", text, re.IGNORECASE):
            result["type"] = "Image Retrieval"
            result["content"] = None
            return result
            
        fa_match = re.search(r"Conclusion\W*[:：]\s*(.*)", text, re.IGNORECASE | re.DOTALL)
        if fa_match:
            result["type"] = "Final Answer"
            result["content"] = fa_match.group(1).strip()
            return result

        print("ERROR: No action type matched.")
        return result        