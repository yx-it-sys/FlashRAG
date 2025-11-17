from collections import deque
import os
import json
from pipeline import Pipeline
from meta_plan import MetaPlan
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import parse_action
import torch

class DFAExecutor():
    def __init__(self, config, prompts_path, model_name="Qwen/Qwen2.5-7B-Instruct"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16).to(self.device)
        self.pipeline = Pipeline(config, self.model, self.tokenizer, self.device, max_loops=2, ret_thresh=0.7)
        self.plan_generator = MetaPlan(self.model, self.tokenizer, self.device, prompts_path)

    def _execute_node(self, node_id: str, dependency_results: dict):
        prev_answers = {dep_id: result[0] for dep_id, result in dependency_results.items()}
        print(f"prev_answers: {prev_answers}")
        prev_contexts = [result[1] for dep_i, result in dependency_results.items()]
        print(f"prev_contexts: {prev_contexts}")    
        formatted_question = self._format_sub_question(node_id, prev_answers)
        print(f"formatted question: {formatted_question}")
        if not formatted_question:
            final_aggregated_answer =  f"Final aggregation of results: {list(dependency_results.values())}"
            return final_aggregated_answer, prev_contexts
        
        answer, new_context = self.pipeline.run_with_question_only(question=formatted_question)
        print(f"Answer: {answer}")
        all_contexts = prev_contexts + [new_context]

        return answer, all_contexts
     
    def _write_results(self, final_result, all_contexts, item):
        result_data = {}
        result_data["id"] = item.id
        result_data["question"] = item.question
        result_data["ans_full"] = item.golden_answers
        result_data["prediction"] = final_result
        result_data["context"] = all_contexts
        
        file_path = os.path.join(self.config["save_dir"], "output.jsonl")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_data, ensure_ascii=False) + "\n")
    
    def run(self, dataset, do_eval=True, pred_process_fun=None):
        for data in dataset:
            self.execute(data)
        ### Continue coding...
            
    def serial_execute(self, item):
        question = item.question
        max_loop = 3
        loop = 0
        context = []
        final_action = "I can't answer."
        while loop < max_loop:
            print(f"Loop {loop}:")
            planning = self.plan_generator.generate(question, context)
            action_type, current_action = parse_action(planning)
            # print(f"Action Type: {action_type}")
            # print(f"Current Action: {current_action}")    
            if action_type.lower() == "search":
                loop += 1
                answer = self.pipeline.run_with_question_only(current_action, context)
                if answer is None:
                    return final_answer, context
                context.append(self.pipeline.run_with_question_only(current_action, context))
            elif action_type.lower() == "reason":
                loop += 1
                context.append(current_action)
            elif action_type.lower() == "conclude":
                final_action = current_action
                break
        return final_action, context
        
        
            
