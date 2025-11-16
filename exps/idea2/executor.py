from concurrent.futures import ThreadPoolExecutor
import re
from collections import defaultdict, deque
import os
import json
from pipeline import Pipeline
from meta_dfa import MetaDFA
from flashrag.utils import get_generator

class DFAExecutor():
    def __init__(self, prompt_path, config):
        self.config = config
        self.generator = get_generator(config)
        self.pipeline = Pipeline(config, model_name="Qwen/Qwen2.5-7B-Instruct", max_loops=3, ret_thresh=0.7)
        self.meta_dfa = MetaDFA(prompt_path, generator=self.generator)
        self.graph = None   # 将会在_parse_graph()方法中修改
        self.dependencies = None    # 将会在_parse_graph()方法中修改
        self.futures = {}

    def _parse_graph(self, graph_def: dict):
        """
        Parse forward DAG, flip the forward DAG to reverse dependenciy reflection.
        """
        self.graph = graph_def.get("states", {})
        self.dependencies = defaultdict(list)
        reverse_map = defaultdict(list)

        for source_node, details in self.graph.items():
            transitions = details.get("transistions", {})
            for _, dest_node in transitions.items():
                reverse_map[dest_node].append(source_node)
        
        for node_id in self.graph:
            self.dependencies[node_id] = sorted(reverse_map.get(node_id, []))

    def _format_sub_question(self, node_id: str, prev_answers: dict) -> str:
        sub_question_template = self.graph[node_id].get("sub_question", "") # 带有[answer_for_xx]的sub_question
        def replace_func(match):
            dep_id = match.group(1)
            return str(prev_answers.get(dep_id, match.group(0)))
        formatted_question = re.sub(r"\[answer_from_(.*?)\]", replace_func, sub_question_template)
        return formatted_question
    
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

    def parallel_execute(self, item):
        question = item.question
        automaton = self.meta_dfa.generate_dfa(question)
        self._parse_graph(automaton)
        self.futures = {}
        with ThreadPoolExecutor() as executor:
            nodes_to_run = set(self.graph.keys())

            while nodes_to_run:
                ready_nodes = {
                    node for node in nodes_to_run
                    if all(dep in self.futures for dep in self.dependencies[node])
                }
                print(f"Ready Nodes: {ready_nodes}")
                if not ready_nodes:
                    if nodes_to_run:
                      raise ValueError(f"图执行错误：可能存在循环依赖。剩余节点: {nodes_to_run}")
                    break

                for node_id in ready_nodes:
                    dep_results = {
                        dep_id: self.futures[dep_id].result()
                        for dep_id in self.dependencies[node_id]
                    } 
                    future = executor.submit(self._execute_node, node_id, dep_results)
                    self.futures[node_id] = future

                nodes_to_run -= ready_nodes
            
            final_node = "q_final" 
            if final_node in self.futures:
                final_answer, all_contexts = self.futures[final_node].result()
                self._write_results(final_answer, all_contexts, item)
                return final_answer
            else:
                print("警告：图中未找到 'q_final' 节点。")
                return None
            
    def serial_execute(self, item):
        question = item.question
        automaton = self.meta_dfa.generate_dfa(question)
        print(f"automaton: {automaton}")
        self._parse_graph(automaton)

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