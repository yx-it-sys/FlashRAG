import tomllib
from utils import general_generate, search, compare, conclude
import re
from typing import List
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
    
    def extract(self, result: str) -> List[str]:
        pattern = r"```(?:python|Python)?\s*(.*?)```"
        match = re.search(pattern, result, re.DOTALL)

        if not match:
            raw_text = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()
        else:
            raw_text = match.group(1)
        
        lines = raw_text.split('\n')

        executable_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                continue
            executable_lines.append(line)

        return executable_lines
    