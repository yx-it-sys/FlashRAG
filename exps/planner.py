import tomllib
from utils import general_generate
import re

class Planner():
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        with open('prompts/planner.toml', 'rb')as f:
            self.prompt = tomllib.load(f)
        
    def generate(self, query: str):
        system_prompt = self.prompt['system_prompt']
        user_prompt = self.prompt['user_prompt'].format(user_query=query)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        result = general_generate(messages, self.model, self.tokenizer)
        draft_plan = self.extract(result)
        return draft_plan
    
    def extract(self, result: str):
        clean_text = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()
        step_pattern = re.compile(r'^\s*(\s+)\.\s*(.*?)(?=^\s*\d+\.|\Z)', re.DOTALL|re.MULTILINE)
        matches = step_pattern.findall(clean_text)
        parsed_plan = []
        for step_num, step_content in matches:
            parsed_plan.append({
                "step": int(step_num),
                "content": step_content.strip()
            })

        return parsed_plan