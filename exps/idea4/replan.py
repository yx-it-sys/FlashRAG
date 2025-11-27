import re
import tomllib
from utils import chat_with_qwen, extract_json

class ActionPlanner():
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        with open("prompts/process_and_replan.toml", "rb") as f:
            self.prompts = tomllib.load(f)
        self.history = []
    
    def generate(self, user_query: str, draft_plan: str, historical_information: str):
        messages = [
            {"role": "system", "content": self.prompts['system_prompt']},
            {"role": "user", "content": self.prompts['user_prompt'].format(
                user_query=user_query,
                draft_plan=draft_plan,
                historical_infomation=historical_information
            )}
        ]
        response = chat_with_qwen(model=self.model, tokenizer=self.tokenizer, messages=messages, type="qwen2", enable_thinking=False).strip()
        action = extract_json(response)
        return action