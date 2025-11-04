from typing import List
from flashrag.prompt import MMPromptTemplate
from flashrag.pipeline import BasicMultiModalPipeline
from flashrag.utils import get_retriever, get_generator
import re
import os
import json
import tomllib

class NFAPipeline(BasicMultiModalPipeline):
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        super().__init__(config, prompt_template)
        
        self.config = config
        prompt_path = self.config['dfa_prompt_path']
        with open(prompt_path, "rb") as f:
            self.prompt = tomllib.load(f)

        self.generator = get_generator(config) if generator is None else generator
        self.retriever = get_retriever(config) if retriever is None else retriever

        if prompt_template is None:
            prompt_template = MMPromptTemplate(config)
        self.prompt_template = prompt_template

    def dfa_print(self, record: List[dict]):
        """Pretty print the state transitions and their results."""
        if not record:
            return
            
        last_record = record[-1]
        state = last_record["state"].upper()
        print(f"\n{'='*20} {state} STATE {'='*20}")
        
        if "response" in last_record:
            print(f"Response:\n{last_record['response']}\n")
            
        if "result" in last_record:
            print("Result:")
            if isinstance(last_record["result"], dict):
                for k, v in last_record["result"].items():
                    print(f"- {k}: {v}")
            else:
                print(f"- {last_record['result']}")
        
        print("="*50 + "\n")

    def _state_initial(self, run_state: dict) -> str:
        return "S_Plan"
    
    def _parse_plan(self, response: str):
        pattern = re.compile(r"<sub-question>(.*?)</sub-question>\s*<need_search>(.*?)</need_search>", re.DOTALL)

        match = pattern.search(response)
        if match:
            sub_question = match.group(1).strip()
            need_search = match.group(2).strip()
            return sub_question, need_search
        else:
            print("未找到匹配项。")
            return None, None

    def _state_plan(self, run_state: dict) -> str:
        input_prompt = self.prompt_template.get_string_for_nfa(config=self.config, prompt=self.prompt, state_type="plan", run_state=run_state)
        response_dict = self.generator.generate([input_prompt])
        response = response_dict["output_text"][0]
        sub_question, need_search = self._parse_plan(response)

        run_state["plan"] = {"sub_question": sub_question, "need_search": need_search}
        run_state["record"].append({"state": "plan", "response": response, "result": run_state["plan"]})
        self.dfa_print(run_state['record'])

        if need_search:
            return "S_Retrieve"
        else:
            return "S_generate"
    
    def _state_retrieve(self, run_state: dict) -> str:
        query = run_state["plan"]["sub_question"]

        search_text = self.retriever.search(query, 2)
        search_text = search_text[0]["contents"]
        run_state["retrieved_docs"] = f"Contents of retrieved documents:\n{' '.join(search_text)}"
        run_state["record"].append({"state": "retrieve", "result": run_state["retrieved_docs"]})
        self.dfa_print(run_state["record"])

        return "S_Assess"
    
    def _parse_assess(self, response: str):
        if "pass" in response:
            return "pass"
        elif response.startswith("fail:"):
            parts = response.split(':', 1)
            if len(parts) > 1:
                return parts[1].strip()
        else:
            print("未找到匹配项。")
            return None


    def _state_assess(self, run_state: dict) -> str:
        input_prompt = self.prompt_template.get_string_for_nfa(config=self.config, prompt=self.prompt, state_type="assess", run_state=run_state)
        response_dict = self.generator.generate([input_prompt])
        response = response_dict["output_text"][0]
        assess, reason = self._parse_assess(response)
        
        run_state["record"].append({"state": "assess", "response": response, "result": {"assessment_result": assess, "reason": reason}})
        self.dfa_print(run_state["record"])

        if assess == 'pass':
            run_state['assessment_result'] = assess
            return "S_Generate"
        else:
            run_state['assessment_result'] = assess
            run_state['reason'] = reason
            return "S_Refine"
        
    def _state_refine(self, run_state: dict) -> str:
        input_prompt = self.prompt_template.get_string_for_nfa(config=self.config, prompt=self.prompt, state_type="refine", run_state=run_state)
        response_dict = self.generator.generate([input_prompt])
        refined_query = response_dict['output_text'][0]
        run_state['current_query'] = refined_query
        run_state["record"].append({"state": "refine", "response": refined_query, "result": refined_query})
        self.dfa_print(run_state['record'])

        return "S_Retrieve"
    
    def _parse_generate(self, response: str):
        pattern = re.compile(
            r"<(?P<tag>Final_Answer|Further_Analysis)>(?P<content>.*?)</(?P=tag)>", re.DOTALL
        )
        match = pattern.search(response.strip())

        if match:
            tag_name = match.group('tag')
            content = match.group('content').strip()
            return tag_name, content
        else:
            print("未找到匹配项。")
            return None, None

    def _state_generate(self, run_state: dict) -> str:
        input_prompt = self.prompt_template.get_string_for_nfa(config=self.config, prompt=self.prompt, state_type="generate", run_state=run_state)
        response_dict = self.generator.generate([input_prompt])
        response = response_dict['output_text'][0]

        mark, answer = self._parse_generate(response)
        run_state["record"].append({"state": "generate", "response": response, "result": {"mark": mark, "answer": answer}})
        self.dfa_print(run_state['record'])

        if mark == "Final Answer":
            run_state['final_answer'] = answer
            return "S_Final"
        elif mark == "Further Analysis":
            run_state['further_analysis'] = answer
            return "S_Plan"

    def _state_final(self, run_state: dict) -> str:
        result_data = {
            "id": run_state['id'],
            "question": run_state['initial_query'],
            "prediction": run_state['final_answer'],
            "record": run_state['record']
        }
        return result_data
    
    def safe_write(self, file_path: str, data: dict):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    
    def run(self, dataset, do_eval=True, pred_process_func=None, uncertainty_type=None):
        data_items_list = []
        for item in dataset:
            data_items_list.append({
                "id": item.id,
                "question": item.question,
                "answers": item.golden_answers[0],
                }
            )
        pred_answer_list = []
        for item in data_items_list:
            run_state = {
                "initial_query": item.question,
                "current_query": item.question,
                "id": item.id,
                "plan": None,
                "retrieved_docs": [],
                "assessment_result": None,
                'reason': None,
                "final_answer": None,
                "further_analysis": None,
                "record": [],
            }

            state_handlers = {
                "S_initial": self._state_initial,
                "S_Plan": self._state_plan,
                "S_Retrieve": self._state_retrieve,
                "S_Assess": self._state_assess,
                "S_Refine": self._state_refine,
                "S_Generate": self._state_generate
            }

            current_state = "S_Initial"
            while current_state != "S_Final":
                handler = state_handlers[current_state]
                next_state = handler(run_state)
                current_state = next_state
            
            result_data = self._state_final(run_state)
            result_data['golden_answers'] = item.golden_answers
            pred_answer_list.append(result_data['prediction'])           
            file_path = os.path.join(self.config["save_dir"], "output.jsonl")
            self.safe_write(file_path, result_data)
        
        dataset.update_output("pred", pred_answer_list)
        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_func=pred_process_func)                 
        
        return dataset
       