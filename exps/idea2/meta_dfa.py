import tomllib
from typing import List
import json
from json_repair import repair_json

class MetaDFA():
    def __init__(self, prompts_path, generator):
        self.generator = generator
        self.prompt = self._load_prompts_from_toml(prompts_path)

    def _load_prompts_from_toml(self, prompts_path):
        with open(prompts_path, "rb") as f:
            data = tomllib.load(f)
            system_content = data['system_prompt']['sys']
            user_content = data['user_prompt']['user']
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
        return messages

    def extract_json(self, response: str) -> dict | None:
        marker = "**json:**"
        try:
            index = response.find(marker)
            
            if index == -1:
                print("Error: Marker '**json**' not found in the response.")
                return None
            payload_part = response[index + len(marker):]
        
            first_brace_index = payload_part.find('{')
            last_brace_index = payload_part.rfind('}')
            
            if first_brace_index == -1 or last_brace_index == -1 or last_brace_index < first_brace_index:
                print("Error: Could not find a valid JSON object starting with '{' and ending with '}'.")
                return None
                
            json_string = payload_part[first_brace_index : last_brace_index + 1]
            
            parsed_json = json.loads(json_string)
            return parsed_json

        except json.JSONDecodeError as e:
            print("⚠️ LLM output is not valid JSON. Attempting to repair...")
            print(json_string)        
            try:
                repaired_json_string = repair_json(response)
                parsed_json = json.loads(repaired_json_string)
                return parsed_json
            except (json.JSONDecodeError, ValueError) as e:
                print(f"❌ Failed to parse JSON even after repair. Error: {e}")
                return None
            
    def generate_dfa(self, question: str) -> List:
        current_prompt = [p.copy() for p in self.prompt]
        current_prompt[1]["content"] = current_prompt[1]["content"].format(initial_query=question)
        
        response_dict = self.generator.generate([current_prompt])
        response = response_dict[0]
        # text = self.tokenizer.apply_chat_template(
        #     current_prompt,
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
        parsed_json = self.extract_json(response)
        return parsed_json