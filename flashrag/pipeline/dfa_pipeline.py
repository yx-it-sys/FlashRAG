from typing import List
from flashrag.prompt import MMPromptTemplate, PromptTemplate
from flashrag.pipeline import BasicMultiModalPipeline
from flashrag.utils import get_retriever, get_generator
import re
import os
import json
from json_repair import repair_json
import tomllib
from mimetypes import guess_type
import base64
import time
from io import BytesIO
import requests
from serpapi import GoogleSearch

class DFAQAPipeline(BasicMultiModalPipeline):    
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        super().__init__(config, prompt_template)        
        self.config = config
        self.API_KEY = "7aa8ce7907a044e9eb0a7d7ced6eaeb176da938af233512d15ecf7d11832617c"
        self.retry_attempt = 3

        prompt_path = self.config['dfa_qa_prompt_path']
        with open(prompt_path, "rb") as f:
            self.prompt = tomllib.load(f)

        self.generator = get_generator(config) if generator is None else generator
        self.retriever = get_retriever(config) if retriever is None else retriever

        if prompt_template is None:
            prompt_template = PromptTemplate(config)
        self.prompt_template = prompt_template
    
    def _fine_search(self, query):
        params = {
            "engine": "google",
            "q": query,
            "api_key": self.API_KEY,
            "num": 2,
        }
        for i in range(self.retry_attempt):
            try:
                search = GoogleSearch(params)
                results = search.get_dict()
                results =  results.get("organic_results", [])
            except Exception as e:
                print(f"Attempt {i+1} failed: {e}")
                if i < self.retry_attempt - 1:
                    time.sleep(2)
                else:
                    print("All retries failed.")
                    results = []

        if results is not None:
            texts = [f"{item.get('title','')}" + item.get("snippet","") for item in results]
        else:
            texts = "No Retrieval Results."
        return texts

    def _parse_output_json(self, response: str):
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        print(f"{response}\nDirect parsing failed, attempting repairing...")
        try:
            repaired_json = repair_json(response)
            return json.loads(repaired_json)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"'json-repair' 修复失败: {e}")
            return None
        
    def _parse_plan(self, response: str):
        pattern = re.compile(
            r"<reasoning>(?P<reasoning>.*?)\s*"
            r"<sub-question>(?P<sub_question>.*?)\s*"
            r"<need_search>(?P<need_search>.*)", 
            re.DOTALL
        )

        match = pattern.search(response.strip())
        if match:
            reasoning = match.group('reasoning').strip()
            sub_question = match.group('sub_question').strip()
            need_search_raw = match.group('need_search').strip()
            end_tag_pos = need_search_raw.rfind('</')
            if end_tag_pos != -1:
                need_search = need_search_raw[:end_tag_pos].strip()
            else:
                need_search = need_search_raw
            
            return reasoning, sub_question, need_search
        else:
            print(f"_parse_plan: 未找到匹配项。{response}")
            return None, None, None
    
    def _parse_assess(self, response: str):
        if response.startswith("complete:"):
            return "complete", None
        elif response.startswith("incomplete:"):
            parts = response.split(':', 1)
            if len(parts) > 1:
                return parts[0].strip(), parts[1].strip()
        else:
            print(f"_parse_assess: 未找到匹配项。{response}")
            return None, None        

    def dfa_print(self, record: List[dict]):
        """Pretty print the state transitions and their results."""
        if not record:
            return
            
        last_record = record[-1]
        state = last_record["state"].upper()
        print(f"\n{'='*20} {state} STATE {'='*20}")
        
        if "response" in last_record:
            print(f"Response:\n{last_record['response']}\n")

        if "further_analysis" in last_record:
            print(f"Further Analysis:\n {last_record['further_analysis']}")

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
    
    def _state_plan(self, run_state: dict) -> str:
        input_prompt = self.prompt_template.get_string_for_dfa(config=self.config, prompt=self.prompt, state_type="plan", run_state=run_state)
        # print(f"input_prompt: {input_prompt}")
        response_dict = self.generator.generate_for_dfa_qa([input_prompt])
        # print(f"Response Dict: {response_dict}")
        output = response_dict[0]
        response = self._parse_output_json(output)
        
        if response is None:
            run_state["record"].append({"state": "plan", "response": output, "result": "Fail to parse llm's output!"})
            self.dfa_print(run_state["record"])
            return "S_Fail"
        
        reasoning = response.get('reasoning', '')
        need_search = response.get('need_search', '')
        sub_question = response.get('sub_question', '')

        plan = {"reasoning": reasoning, "need_search": need_search, "sub_question": sub_question}
        run_state["record"].append({"state": "plan", "response": output, "further_analysis": run_state['further_analysis'], "result": plan})
        self.dfa_print(run_state['record'])
        
        run_state['current_query'] = sub_question

        if need_search.lower() == "true":
            return "S_Retrieve"
        
        elif need_search.lower() == "false":
            return "S_Generate"
        
        else:
            run_state["record"].append({"state": "plan", "response": output, "result": "Fail to parse llm's output!"})
            self.dfa_print(run_state["record"])
            return "S_Fail"

    def _state_retrieve(self, run_state: dict) -> str:
        query = run_state['current_query']
        search_results = self.retriever.search(query, 3)
        # search_results = self._fine_search(query)
        search_text_list = []

        for result in search_results:
            search_text_list.append(result['contents'])
            # search_text_list.append(result)

        search_texts = "\n\n".join(search_text_list)
        run_state["retrieved_docs"] = f"Contents of retrieved documents:\n{search_texts}"
        run_state["record"].append({"state": "retrieve", "result": run_state["retrieved_docs"]})
        self.dfa_print(run_state["record"])

        return "S_Assess"
    
    def _state_judge(self, run_state: dict) -> str:
        if run_state['generate_plan_loop_counter'] > 5:
            return "S_Fail"
        else:
            run_state['generate_plan_loop_counter'] += 1
            input_prompt = self.prompt_template.get_string_for_dfa(config=self.config, prompt=self.prompt, state_type="judge", run_state=run_state)
            response_dict = self.generator.generate_for_dfa_qa([input_prompt])
            output = response_dict[0]
            judgement, reason = self._parse_assess(output)
            
            run_state["record"].append({"state": "judge", "response": output, "further_analysis": run_state['further_analysis'], "result": {"judgement_result": judgement, "reason": reason if reason is not None else ""}})
            self.dfa_print(run_state["record"])

            if judgement == 'complete':
                run_state['judgement_result'] = judgement
                return "S_Final"
            else:
                run_state['judgement_result'] = judgement
                run_state['s_judgement_reason'] = reason
                return "S_Plan"

            

    def _state_assess(self, run_state: dict) -> str:
        if run_state['retrieval_assess_refine_loop_counter'] > 5:
            return "S_Fail"
        else:
            run_state['retrieval_assess_refine_loop_counter'] += 1
            input_prompt = self.prompt_template.get_string_for_dfa(config=self.config, prompt=self.prompt, state_type="assess", run_state=run_state)
            response_dict = self.generator.generate_for_dfa_qa([input_prompt])
            response = response_dict[0]
            assess, reason = self._parse_assess(response)
            
            run_state["record"].append({"state": "assess", "response": response, "result": {"assessment_result": assess, "reason": reason if reason is not None else ""}})
            self.dfa_print(run_state["record"])

            if assess == 'complete':
                run_state['assessment_result'] = assess
                return "S_Generate"
            else:
                run_state['assessment_result'] = assess
                run_state['s_assessment_reason'] = reason
                return "S_Refine"
        
    def _state_refine(self, run_state: dict) -> str:
        input_prompt = self.prompt_template.get_string_for_dfa(config=self.config, prompt=self.prompt, state_type="refine", run_state=run_state)
        response_dict = self.generator.generate_for_dfa_qa([input_prompt])
        refined_query = response_dict[0]
        run_state['current_query'] = refined_query
        run_state["record"].append({"state": "refine", "response": refined_query, "result": refined_query})
        self.dfa_print(run_state['record'])

        return "S_Retrieve"
    
    def _parse_generate(self, response: str):
        pattern = re.compile(
            r"<reasoning>(?P<reasoning_content>.*?)(?:</reasoning>|$)\s*<Response>(?P<response_content>.*?)(?:</Response>|$)",
            re.DOTALL            
        )
        match = pattern.search(response.strip())

        if match:
            reasoning_content = match.group('reasoning_content').strip()
            response_content = match.group('response_content').strip()
            return reasoning_content, response_content
        else:
            print(f"_parse_generate: 未找到匹配项。{response}")
            return None, response

    def _state_generate(self, run_state: dict) -> str:
        input_prompt = self.prompt_template.get_string_for_dfa(config=self.config, prompt=self.prompt, state_type="generate", run_state=run_state)
        response_dict = self.generator.generate_for_dfa_qa([input_prompt])
        output = response_dict[0]
        response = self._parse_output_json(output)
        if response is None:
            run_state["record"].append({"state": "generate", "response": output, "result": "Fail to parse llm's output!"})
            self.dfa_print(run_state["record"])
            return "S_Fail"
        
        reasoning_content = response.get('reasoning', '')
        response_content = response.get('response', '')

        if (reasoning_content is None) or (response_content is None):
            run_state["record"].append({"state": "generate", "response": output, "result": "Fail to parse llm's output!"})
            self.dfa_print(run_state["record"])
            return "S_Fail"

        run_state["record"].append({"state": "generate", "response": output, "result": {"reasoning": reasoning_content, "response": response_content}})
        self.dfa_print(run_state['record'])

        run_state['further_analysis'].append({"sub_question": run_state['current_query'], "reasoning": reasoning_content, "answer": response_content})
        print(f"further_analysis: {run_state['further_analysis']}")
        
        return "S_Judge"
        
    def _state_final(self, run_state: dict) -> str:
        input_prompt = self.prompt_template.get_string_for_dfa(config=self.config, prompt=self.prompt, state_type="final_answer", run_state=run_state)
        response_dict = self.generator.generate_for_dfa_qa([input_prompt])
        output = response_dict[0]
        response = self._parse_output_json(output)
        final_answer = response.get('final_answer', '')

        if (response is None) or (final_answer == ''):
            run_state["record"].append({"state": "final", "response": output, "further_analysis": run_state["further_analysis"], "result": "Fail to parse llm's output!"})
            self.dfa_print(run_state["record"])
            return "S_Fail"
        
        run_state["final_answer"] = final_answer
        result = {'reasoning': run_state.get('reasoning', ''), 'final_answer': run_state.get('final_answer', '')}
        run_state["record"].append({"state": "final", "response": output, "result": result})
        self.dfa_print(run_state["record"])

        result_data = {
            "id": run_state['id'],
            "question": run_state['initial_query'],
            "prediction": run_state['final_answer'],
            "record": run_state['record']
        }
        return result_data
    
    def _state_fail(self, run_state: dict) -> str:
        if run_state['s_generate_response'] is None:
            prediction = "I can't answer."
        else:
            prediction = f"The Answer maybe {run_state['s_generate_response']}."
        result_data = {
            "id": run_state['id'],
            "question": run_state['initial_query'],
            "prediction": prediction,
            "record": run_state['record']
        }
        return result_data
    
    def safe_write(self, file_path: str, data: dict):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    
    def run(self, dataset, do_eval=True, pred_process_func=None, uncertainty_type=None):
        data_items_list = []
        for i, item in enumerate(dataset):
            data_items_list.append({
                "id": item.id,
                "question": item.question,
                "golden_answers": item.golden_answers,
                }
            )
        pred_answer_list = []
        for item in data_items_list:
            run_state = {
                "id": item['id'],   # id
                "initial_query": item['question'],  # User Initial Query
                "current_query": item['question'],  # sub-question from S_Plan
                "retrieved_docs": [],   # retrieved results from S_Retrieve
                "assessment_result": None,  #
                's_assessment_reason': None,
                "judgement_result": None,
                "s_judgement_reason": None,
                "final_answer": None,
                'further_analysis': [],
                "s_generate_response": None,
                "record": [],
                "generate_plan_loop_counter": 0,
                "retrieval_assess_refine_loop_counter": 0
            }

            state_handlers = {
                "S_Initial": self._state_initial,
                "S_Plan": self._state_plan,
                "S_Retrieve": self._state_retrieve,
                "S_Assess": self._state_assess,
                "S_Refine": self._state_refine,
                "S_Generate": self._state_generate,
                "S_Fail": self._state_fail,
                "S_Judge": self._state_judge,
                "S_Final": self._state_final,
            }

            current_state = "S_Initial"
            while current_state != "S_Final" and current_state != "S_Fail":
                handler = state_handlers[current_state]
                next_state = handler(run_state)
                current_state = next_state
            if current_state == "S_Final":
                result_data = self._state_final(run_state)
            elif current_state == "S_Fail":
                result_data = self._state_fail(run_state)
            else:
                print("ERROR! current state is not any of 'S_Fail' or 'S_Final'!")
            
            result_data['golden_answers'] = item['golden_answers']
            pred_answer_list.append(result_data['prediction'])           
            file_path = os.path.join(self.config["save_dir"], "output.jsonl")
            self.safe_write(file_path, result_data)
        
        dataset.update_output("pred", pred_answer_list)
        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_func=pred_process_func)                 
        
        return dataset

       