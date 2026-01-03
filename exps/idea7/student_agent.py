from utils import qwen2_generate
from flashrag.pipeline import BasicPipeline
from flashrag.utils import get_retriever
import tomllib
import re

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
        response = qwen2_generate(self.model, self.processor, messages)
        print(f"Rewritten Query: {response}")
        pattern = r"<rewritten_query>(.*?)</rewritten_query>"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            results = match.group(1).strip()
        else:
            print("No rewritten query found, using original.")
            results = [action_content]
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
            print("="*30)
            print(f"STUDENT Prompt Contexts:\n{messages}")
            print("="*30)
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
                query_list = self.query_rewrite(action)
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
                print("<Final Answer>")
                final_answer = action
                logs.append({"current_final_answer": final_answer})
                break

        return final_answer, logs
    
    def parse_action(self, response):
        response = response.strip()
    
        # --- 核心修改 ---
        # 1. (?:^|\n) : 锚定行首（防止匹配到句子中间提到的关键词）。
        # 2. (?:\*\*)? : 兼容 markdown 加粗。
        # 3. (Final Answer|...) : 捕获动作类型 (Group 1)。
        # 4. (?: ... )? : 关键修改！整个“冒号+内容”部分变成了可选组。
        #    这意味着如果后面没有冒号（例如纯 "Image Retrieval"），正则依然能匹配成功。
        pattern = r"(?:^|\n)\s*(?:\*\*)?(Final Answer|Image Retrieval|Text Retrieval)(?:\*\*)?(?:\s*[:：]\s*(.*))?"
    
        # 使用 re.DOTALL 确保 (.*) 能抓取换行符后的内容
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
    
        if match:
            action_type = match.group(1).strip()
            
            # group(2) 是冒号后面的内容。
            # 如果没有冒号（比如你的 case），group(2) 会是 None，我们需要处理这种情况。
            content = match.group(2)
            if content:
                content = content.strip()
            else:
                content = None
    
            # --- 特殊逻辑处理 ---
            
            # 1. Image Retrieval 通常不需要参数，或者参数就是图片本身
            if "Image Retrieval" in action_type:
                return {"type": "Image Retrieval", "content": None}
            
            # 2. Text Retrieval 理论上必须有参数
            # 如果 content 是 None，说明模型只输出了 "Text Retrieval" 但没给查询词
            # 这里视你的业务逻辑而定，可以报错，也可以返回 None 让 Agent 决定
            if "Text Retrieval" in action_type and not content:
                print("WARNING: Text Retrieval detected but no query provided.")
                return {"type": "Text Retrieval", "content": ""} # 或者 None
    
            # 3. Final Answer
            return {"type": action_type, "content": content}
    
        else:
            # Fallback
            print(f"WARNING! No standard action format found. Treating full text as Final Answer.")
            return {"type": "Final Answer", "content": response}