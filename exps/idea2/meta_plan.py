import tomllib
from typing import List
import re

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
                
    def generate_plan(self, question: str) -> List:
        current_plan_prompt = [p.copy() for p in self.plan_prompt]
        # fill in the user's question into the plan prompt
        current_plan_prompt[1]["content"] = current_plan_prompt[1]["content"].format(query=question)
        
        response = self.generator.generate([current_plan_prompt])[0]
        print(f"response: {response}")
        planning_list = self.parse_response(response)
        print(f"Planning List: {planning_list}")
        return planning_list
    
    def parse_response(self, response: str):
        if "List:" not in response:
            return []
        list_part = response.split("List:")[1].strip()
        lines = list_part.splitlines()
        parsed_items = [re.sub(r"^\d+\.\s*", "", line).strip() for line in lines if line.strip()]

        return parsed_items

