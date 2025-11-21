import tomllib
from typing import List
from utils import extract_json, chat_with_qwen
import torch

class MetaPlan():
    def __init__(self, model, tokenizer, device, prompts_path):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.prompts_path = prompts_path
        self.plan_prompt = self._load_plan_prompts(self.prompts_path)
        self.history = []
        
    def _load_plan_prompts(self, prompts_path):
        with open(prompts_path, "rb") as f:
            data = tomllib.load(f)
            system_content = data['system_prompt']
            user_prompt = data['user_prompt']
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_prompt}
            ]
        return messages

    def reset_prompt(self):
        self.plan_prompt = self._load_plan_prompts(self.prompts_path)
        
    def generate(self, question: str, context) -> List:
        current_plan_prompt = [p.copy() for p in self.plan_prompt]
        # fill in the user's question into the plan prompt
        current_plan_prompt[1]["content"] = current_plan_prompt[1]["content"].format(query=question)
        # print(f"Current Plan Prompt: {current_plan_prompt}")
        if context is not None:
            context_str = "\n".join([item for item in context if item is not None])
            current_plan_prompt.append({"role": "user", "content": context_str})
        
        response = chat_with_qwen(mode=self.model, tokenizer=self.tokenizer, messages=current_plan_prompt, type="qwen3", mode="thinking")
        self.plan_prompt.append({"role": "assistant", "content": response['content']})
        print(f"thinking: {response['thinking']}")
        print(f"response: {response['content']}")
        return response
    
