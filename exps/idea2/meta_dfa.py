import tomllib
from typing import List
from utils import extract_json

class MetaDFA():
    def __init__(self, prompts_path, generator):
        self.generator = generator
        self.dfa_prompt = self._load_dfa_prompts(prompts_path)
        
    def _load_dfa_prompts(self, prompts_path):
        with open(prompts_path, "rb") as f:
            data = tomllib.load(f)
            system_content = data['system_prompt']['sys']
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": "Here is the user's question, now start your work:\n{initial_query}"}
            ]
        return messages
                
    def generate_dfa(self, question: str) -> List:
        current_dfa_prompt = [p.copy() for p in self.dfa_prompt]
        # fill in the user's question into the dfa prompt
        current_dfa_prompt[1]["content"] = current_dfa_prompt[1]["content"].format(initial_query=question)
        
        response = self.generator.generate([current_dfa_prompt])[0]
        # text = self.tokenizer.apply_chat_template(
        #     current_dfa_prompt,
        #     tokenize=False,
        #     add_generation_prompt=True
        # )

        # inputs = self.tokenizer(
        #     [text],
        #     return_tensors="pt",
        # ).to(self.model.device)

        # generated_ids = self.model.generate(
        #     **inputs,
        #     max_new_tokens=1024
        # )
        # generated_ids = [
        #     output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        # ]

        # response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"response: {response}")
        parsed_json = extract_json(response)
        return parsed_json