import tomllib
from typing import List
import json
from json_repair import repair_json
from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.utils import get_generator
import time
class MetaDFA():
    def __init__(self, prompts_path, generator):
        self.generator = generator
        self.classify_prompt = self._load_classify_prompts(prompts_path['classify'])
        self.dfa_prompt = self._load_dfa_prompts(prompts_path['dfa'])
    
    def _load_classify_prompts(self, prompts_path):
        with open(prompts_path, "rb") as f:
            data = tomllib.load(f)
            system_content = data['system_prompt']['sys']
            user_content = data['user_prompt']['user']
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
        return messages
    
    def _load_dfa_prompts(self, prompts_path):
        with open(prompts_path, "rb") as f:
            data = tomllib.load(f)
            system_content = data['system_prompt']['sys']
            user_content = data['user_prompt']['user']
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
        return messages
    
    def validate_and_fix_graph(self, graph_data: dict) -> dict:
        states = graph_data['states']
        if 'q_final' in states and list(states.keys())[-1] != 'q_final':
            del states['q_final']
        if not states or list(states.keys())[-1] != 'q_final':
            states['q_final'] = {}
        
        state_keys = list(states.keys())
        if len(state_keys) < 2:
            return graph_data
        second_last_key = state_keys[-2]
        second_last_state = states[second_last_key]

        if second_last_state.get('transitions', {}).get('*') != 'q_final':
            second_last_state.setdefault('transitions', {})['*'] = 'q_final'

        return graph_data
    
    def check_json(self, response: str) -> dict | None:
        marker = "JSON:"
        try:
            index = response.find(marker)
            
            if index == -1:
                print("Error: Marker 'JSON:' not found in the response.")
                return None
            payload_part = response[index + len(marker):]
        
            first_brace_index = payload_part.find('{')
            last_brace_index = payload_part.rfind('}')
            
            if first_brace_index == -1 or last_brace_index == -1 or last_brace_index < first_brace_index:
                print("Error: Could not find a valid JSON object starting with '{' and ending with '}'.")
                return None
                
            json_string = payload_part[first_brace_index : last_brace_index + 1]
            
            parsed_json = json.loads(json_string)
            automaton = self.validate_and_fix_graph(parsed_json)
            return automaton

        except json.JSONDecodeError as e:
            print("⚠️ LLM output is not valid JSON. Attempting to repair...")
            print(json_string)        
            try:
                repaired_json_string = repair_json(response)
                parsed_json = json.loads(repaired_json_string)
                # 寻找state最后一个状态是否为q_final，如果不是，就加一个
                automaton = self.validate_and_fix_graph(parsed_json)
                return automaton
            except (json.JSONDecodeError, ValueError) as e:
                print(f"❌ Failed to parse JSON even after repair. Error: {e}")
                return None
            
    def generate_dfa(self, question: str) -> List:
        current_dfa_prompt = [p.copy() for p in self.dfa_prompt]
        current_dfa_prompt[1]["content"] = current_dfa_prompt[1]["content"].format(initial_query=question)
        response_dict = self.generator.generate([current_dfa_prompt])
        response = response_dict[0]
        print(f"Response:{response}")
        parsed_json = self.check_json(response)
        return parsed_json


def main():
    config_dict = {
        "dataset_path": "data/datasets/hotpotqa",
        "image_path": "data/datasets/okvqa/images/val2014",
        "index_path": "data/indexes/bm25",
        "corpus_path": "data/indexes/wiki18_100w.jsonl",
        "generator_model_path": "data/models/Qwen2.5-7B-Instruct",
        "retrieval_method": "bm25",
        "metrics": ["em", "f1", "acc"],
        "retrieval_topk": 2,
        "save_intermediate_data": True,
    }
    config = Config("my_config.yaml", config_dict=config_dict)
    prompts_path = {
        "classify": "prompts/question_classify.toml",
        "dfa": "prompts/meta_plan.toml"
    }
    
    all_split = get_dataset(config)
    test_data = all_split["dev"]
    
    generator = get_generator(config)
    meta_dfa = MetaDFA(prompts_path, generator)

    for item in test_data:
        question = item.question
        print(f"Question: {question}")
        automaton = meta_dfa.generate_dfa(question)
        
    



    
if __name__ == "__main__":    
    main()

