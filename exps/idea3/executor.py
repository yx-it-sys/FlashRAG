import os
import json
from pipeline import Pipeline
from meta_plan import MetaPlan
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import parse_action
import torch
import tomllib
from gliner import GLiNER

class DFAExecutor():
    def __init__(self, config, prompts_path, model_name="Qwen/Qwen2.5-7B-Instruct"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        with open("prompts/final_answer.toml", "rb") as f:
            self.final_answer_prompt = tomllib.load(f)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16).to(self.device)
        self.entity_extractor = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
        self.pipeline = Pipeline(config, self.model, self.tokenizer, self.entity_extractor, self.device, max_loops=2, ret_thresh=0.7)
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
                
    def serial_execute(self, item):
        question = item.question
        max_loop = 5
        loop = 0
        context = []
        final_answer = "I can't answer."
        logs = []
        while loop < max_loop:
            print(f"Loop {loop}:")
            planning = self.plan_generator.generate(question, context)
            logs.append({"meta_plan": planning})
            action_type, current_action = parse_action(planning)
            if action_type is None:
                action_type = "reason"
            # print(f"Action Type: {action_type}")
            # print(f"Current Action: {current_action}")    
            if action_type.lower() == "search":
                loop += 1
                answer, log = self.pipeline.run_with_question_only(current_action)
                logs.append({"meta_state": "search", "logs": log})
                
                if answer is None:
                    return final_answer, logs

                context.append(f"{current_action}: {answer}")
            elif action_type.lower() == "reason":
                loop += 1
                context.append(current_action)
                logs.append({"meta_state": "reason", "logs": current_action})
            elif action_type.lower() == "conclude":
                conclusion = current_action
                final_answer = self.generate_final_answer(initial_question=question, conclusion=conclusion)
                print(f"final_answer: {final_answer}")
                logs.append({"meta_state": "conclude", "logs": conclusion})
                break
        # 计划过于长，仍未得出结论，进入Replan
        # 先展示出Trajectory，便于后续制定Replan策略
        print(f"Meta Plan: {logs[0]['meta_plan']}")
        for log in logs[1:]:
            print(f"Meta State: {log['meta_state']}")
            print(f"Logs: {log['logs']}")

        self.plan_generator.reset_prompt()  
        return final_answer, logs

    
    def generate_final_answer(self, initial_question: str, conclusion: str) -> str:
            messages = [                
                {"role": "system", "content": self.final_answer_prompt['system_prompt']},
                {"role": "user", "content": self.final_answer_prompt['user_prompt'].format(initial_question=initial_question, conclusion=conclusion)}
            ]
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            outputs = self.model.generate(**inputs, max_new_tokens=2048)
            response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

            return response
        
            
