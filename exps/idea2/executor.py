from concurrent.futures import ThreadPoolExecutor
import re
from collections import defaultdict, deque
import os
import json
from pipeline import Pipeline
from meta_plan import MetaPlan
from flashrag.utils import get_generator
from transformers import AutoTokenizer, AutoModelForCausalLM

class DFAExecutor():
    def __init__(self, config, model_name="Qwen/Qwen2.5-7B-Instruct"):
        self.pipeline = Pipeline(config, model_name=model_name, max_loops=3, ret_thresh=0.7)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
    
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
        planning = self.plan_generator.generate_plan(question)
        stepwise_answer = []
        contexts = []
        for plan, context in zip(planning_list, contexts):
            

        in_degree = {node_id: len(deps) for node_id, deps in self.dependencies.items()}
        
        queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])

        execution_order = []

        while queue:
            node_id = queue.popleft()
            execution_order.append(node_id)

            successors = self.dependencies.get(node_id, [])
            for successor_id in successors:
                in_degree[successor_id] -= 1
                if in_degree[successor_id] == 0:
                    queue.append(successor_id)

        if len(execution_order) != len(self.graph):
            raise ValueError(f"图执行错误：检测到循环依赖。执行顺序: {execution_order}, 所有节点: {list(self.graph.keys())}")
        
        self.results = {}
        all_contexts_collected = []

        for node_id in execution_order:
            if node_id == "q_final":
                continue

            dep_results = {
                dep_id: self.results[dep_id][0]
                for dep_id in self.dependencies[node_id]
            }

            answer, context = self._execute_node(node_id, dep_results)

            self.results[node_id] = (answer, context)
            all_contexts_collected.append(context)

            return self.results