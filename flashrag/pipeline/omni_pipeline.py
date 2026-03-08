from flashrag.utils import get_retriever, get_generator
from flashrag.pipeline import BasicMultiModalPipeline
import re
import os
import json
import torch

class OmniSearchPipeline(BasicMultiModalPipeline):
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        super().__init__(config, prompt_template)
        self.config = config
        self.generator = get_generator(config) if generator is None else generator
        self.retriever = get_retriever(config) if retriever is None else retriever
    
    def iterative_infer(self, input_prompt):
        response_dict = self.generator.generate([input_prompt])
        response = response_dict[0]
        print(f"First Response: {response}")
        input_prompt["input_prompt"].append({'role': 'assistant', 'content': response})
        
        conversation_num, max_turns = 0, 5
        while conversation_num < max_turns:
            if "Final Answer" in response or "<Final Answer>" in response:
                break
            need_txt_ret = "Text Retrieval" in response
            if need_txt_ret:
                print("Start text retrieval...")
                pattern = r'Text Retrieval[:\s"]*(.*?)(?=<|$)'
                match = re.search(pattern, response, re.DOTALL)
                query_txt = ""
                if match:
                    query_txt = match.group(1).strip()
                    print(f"Query Text: {query_txt}")
                if query_txt == "":
                    print(f"Query_txt is None")
                    search_text = self.retriever.search(input_prompt["question"], 2)
                    search_text = search_text[0]["contents"]
                    print(f"Retrieval result: {search_text}")
                else:
                    search_text = self.retriever.search([query_txt], 1)
                    search_text = search_text[0]["contents"]
                    print(f"Retrieval result: {search_text}")

                contents = []
                if search_text:
                    contents.append({'type': 'text', 'text': f"Contents of retrieved documents:\n{' '.join(search_text)}"})
                else:
                    contents.append({'type': 'text', 'text': "No relevant information found."})    

                input_prompt["input_prompt"].append({'role': 'user', 'content': contents})

                try:
                    response_dict = self.generator.generate([input_prompt])
                    response = response_dict[0]
                    print(f"response: {response}")
                    input_prompt["input_prompt"].append({"role":"assistant", "content": response})
                except Exception as e:
                    print("Inference error, hidden states ignored:", e)
                    return response_dict, input_prompt["input_prompt"]
            else:
                conversation_num += 1
                break
            conversation_num += 1
        
        pattern = r'(?:<Final Answer>|Final Answer:)\s*(.*?)(?=<|$)'
        final_answer_match = re.search(pattern, response, re.DOTALL)

        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            final_answer = final_answer.replace('\n', '')
            print(f"Final Answer: {final_answer}")
            return final_answer, response_dict, input_prompt
        else:
            print(f"Warning: reached end of agent loop for item {conversation_num} without a 'Final Answer'. returning last response")
            return response, response_dict, input_prompt
      
    def safe_write(self, file_path: str, data: dict):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            
    def run(self, dataset, do_eval=True, pred_process_func=None):
        data_items_list = []
        data_list = list(dataset)
        for item in dataset:
            data_items_list.append({
                "id": item.id,
                "question": item.question,
                "answers": item.golden_answers,
                "input_prompt": self.prompt_template.get_string(item, self.config)
                }
            )
        pred_answer_list = []
        context_list = []

        for i, input_prompt in enumerate(data_items_list):
            answer, response_dict, context = self.iterative_infer(input_prompt)
            remove_image_context = context['input_prompt'][2:]
            pred_answer_list.append(answer)
            context_list.append(remove_image_context)
            # print(f"Answer: {answer}")
        result_data = {}
        for i, item in enumerate(data_list):
            result_data["id"] = item.id
            result_data["question"] = item.question
            result_data["image_id"] = item.image_id
            result_data["ans_full"] = item.golden_answers
            result_data["prediction"] = pred_answer_list[i]
            result_data["context"] = context_list[i]
            file_path = os.path.join(self.config["save_dir"], "output.jsonl")
            print(f"Saving to {file_path}")
            self.safe_write(file_path, result_data)
        
        dataset.update_output("pred", pred_answer_list)
        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_func=pred_process_func)                 
        return dataset

    def naive_run(self, dataset, do_eval=True, pred_process_func=None):        
        input_prompts = []
        items_list = list(dataset)
        for item in dataset:
            input_prompts.append({
                "id": item.id,
                "question": item.question,
                "answers": item.golden_answers[0],
                "input_prompt": self.prompt_template.get_string(item, self.config)
                }
            )
        pred_answer_list = []

        for i, input_prompt in enumerate(input_prompts):
            pred_answer = self.generator.generate([input_prompt])[0]
            pred_answer_list.append(pred_answer)
        
        result_data = {}
        for i, item in enumerate(items_list):
            result_data["id"] = item.id
            result_data["question"] = item.question
            result_data["image_id"] = item.image_id
            result_data["ans_full"] = item.golden_answers
            result_data["prediction"] = pred_answer_list[i]
            file_path = os.path.join(self.config["save_dir"], "output.jsonl")
            print(f"Saving to {file_path}")
            self.safe_write(file_path, result_data)
        
        dataset.update_output("pred", pred_answer_list)
        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_func=pred_process_func)                 
        return dataset
