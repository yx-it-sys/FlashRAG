import tomllib
from typing import List
from utils import chat_with_qwen
import re

class MetaPlan():
    def __init__(self, model, tokenizer, device, prompts_path):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        with open(prompts_path, "rb") as f:
            self.plan_prompt = tomllib.load(f)
        self.history = []

    def reset_prompt(self):
        self.plan_prompt = self._load_plan_prompts(self.prompts_path)
        
    def generate(self, question: str) -> List:
        messages = [
            {"role": "system", "content": self.plan_prompt['system_prompt']},
            {"role": "user", "content": self.plan_prompt['user_prompt'].format(user_query=question)}
        ]
        response = chat_with_qwen(model=self.model, tokenizer=self.tokenizer, messages=messages, type="qwen2", enable_thinking=False)
        response = response['content']
        pattern = r"<draft_plan>\s*(.*?)\s*</draft_plan>"
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)

        if not match:
            print("ERROR: No <draft_plan> tags found.")
            return []
        
        plan_content = match.group(1)
        step_pattern = r"(?:^|\n)\s*(\d+)\.\s*(.*?)(?=(?:\n\s*\d+\.)|$)"
        raw_steps = re.findall(step_pattern, plan_content, re.DOTALL)

        structured_plan = []
        for step_num, step_text in raw_steps:
            structured_plan.append({
                "step_id": int(step_num),
                "task": step_text.strip()
            })
        return structured_plan
    
