import tomllib
from typing import List
from utils import extract_json
import torch

class MetaPlan():
    def __init__(self, model, tokenizer, device, prompts_path):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.plan_prompt = self._load_plan_prompts(prompts_path)
        
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
                
    def generate(self, question: str, context) -> List:
        current_plan_prompt = [p.copy() for p in self.plan_prompt]
        # fill in the user's question into the plan prompt
        current_plan_prompt[1]["content"] = current_plan_prompt[1]["content"].format(query=question)
        if context is not None:
            context_str = "\n".join(context)
            current_plan_prompt.append({"role": "user", "content": context_str})
        
        inputs = self.tokenizer.apply_chat_template(
            current_plan_prompt,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(**inputs, max_new_tokens=2048)
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        print(f"response: {response}")
        return response
    
