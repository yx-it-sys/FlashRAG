import tomllib
from typing import List
from utils import extract_json

class MetaPlan():
    def __init__(self, prompts_path, generator):
        self.generator = generator
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
        response = self.generator.generate([current_plan_prompt])[0]
        print(f"response: {response}")
        return response
    
