from flashrag.evaluator import Evaluator
from flashrag.utils import get_retriever, get_generator
from flashrag.uncertainty import integrated_gradient_process
from transformers import AutoTokenizer, AutoProcessor, AutoModelForVision2Seq
from accelerate import Accelerator
import re
import os
import json
import torch
from PIL import Image

class BasicMultiModalPipeline:
    """Base object of all multimodal pipelines. A pipeline includes the overall process of RAG.
    If you want to implement a pipeline, you should inherit this class.
    """

    def __init__(self, config, prompt_template=None):
        from flashrag.prompt import MMPromptTemplate
        self.config = config
        self.device = config["device"]
        self.retriever = None
        self.evaluator = Evaluator(config)
        if prompt_template is None:
            prompt_template = MMPromptTemplate(config)
        self.prompt_template = prompt_template

    def run(self, dataset, pred_process_fun=None):
        """The overall inference process of a RAG framework."""
        pass

    def evaluate(self, dataset, do_eval=True, pred_process_func=None):
        """The evaluation process after finishing overall generation"""

        if pred_process_func is not None:
            dataset = pred_process_func(dataset)

        if do_eval:
            # evaluate & save result
            eval_result = self.evaluator.evaluate(dataset)
            print(eval_result)

        return 


class MMSequentialPipeline(BasicMultiModalPipeline):
    PERFORM_MODALITY_DICT = {
        'text': ['text'],
        'image': ['image']
    }
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        super().__init__(config, prompt_template)
        self.generator = get_generator(config) if generator is None else generator
        self.retriever = get_retriever(config) if retriever is None else retriever
    
    def naive_run(self, dataset, do_eval=True, pred_process_func=None):
        input_prompts = [
            self.prompt_template.get_string(item) for item in dataset
        ]
        
        dataset.update_output("prompt", input_prompts)

        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred", pred_answer_list)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_func=pred_process_func)

        return dataset
    
    def run(self, dataset, do_eval=True, perform_modality_dict=PERFORM_MODALITY_DICT, pred_process_func=None):
        if None not in dataset.question:
            text_query_list = dataset.question
        else:
            text_query_list = dataset.text
        image_query_list = dataset.image

        # perform retrieval
        retrieval_result = []
        for modal in perform_modality_dict.get('text', []):
            retrieval_result.append(
                self.retriever.batch_search(text_query_list, target_modal=modal)
            )
        for modal in perform_modality_dict.get('image', []):
            retrieval_result.append(
                self.retriever.batch_search(image_query_list, target_modal=modal)
           )
        retrieval_result = [sum(group, []) for group in zip(*retrieval_result)]

        dataset.update_output("retrieval_result", retrieval_result)

        input_prompts = [
            self.prompt_template.get_string(item) for item in dataset
        ]
        
        dataset.update_output("prompt", input_prompts)

        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred", pred_answer_list)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_func=pred_process_func)

        return dataset
    
class OmniSearchPipeline(BasicMultiModalPipeline):
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        super().__init__(config, prompt_template)
        self.config = config
        self.generator = get_generator(config) if generator is None else generator
        self.retriever = get_retriever(config) if retriever is None else retriever
    
    def iterative_infer(self, input_prompt, get_hidden_states=False, uncertainty_type=None):
        response_dict = self.generator.generate([input_prompt], get_hidden_states=get_hidden_states, uncertainty_type=uncertainty_type)[0]
        response = response_dict["output_text"][0]
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
                    response_dict = self.generator.generate([input_prompt], get_hidden_states=get_hidden_states, uncertainty_type=uncertainty_type)[0]
                    response = response_dict["output_text"][0]
                    print(f"response: {response}")
                    input_prompt["input_prompt"].append({"role":"assistant", "content": response})
                except Exception as e:
                    print("Inference error, hidden states ignored:", e)
                    return response_dict, input_prompt["input_prompt"]
            else:
                conversation_num += 1
                break
            conversation_num += 1
        
        if get_hidden_states == True:
            dict_to_save = response_dict.copy()
            dict_to_save.pop('output_text', None)
            file_path = f"{self.config['save_dir']}/hidden_states"
            os.makedirs(file_path, exist_ok=True)
            filename = f"{file_path}/hidden_states_{input_prompt['id']}.pth"
            torch.save(dict_to_save, filename)
            print(f"字典已成功保存到: {filename}")
        else:

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
            
    def run(self, dataset, do_eval=True, pred_process_func=None, uncertainty_type=None):
        data_items_list = []
        data_list = list(dataset)
        for item in dataset:
            data_items_list.append({
                "id": item.id,
                "question": item.question,
                "answers": item.golden_answers[0],
                "input_prompt": self.prompt_template.get_string(item, self.config)
                }
            )
        pred_answer_list = []
        context_list = []
        uncertainty_score_list = []

        for i, input_prompt in enumerate(data_items_list):
            answer, response_dict, context = self.iterative_infer(input_prompt, get_hidden_states=False, uncertainty_type=uncertainty_type)
            remove_image_context = context['input_prompt'][2:]
            pred_answer_list.append(answer)
            if uncertainty_type == "entropy":
                uncertainty_score_list.append(response_dict["generation_entropy"])
            elif uncertainty_type == "ig_text":
                uncertainty_score_list.append(response_dict["ig_score"])
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
            if uncertainty_type == "entropy":
                result_data["generation_entropy"] = uncertainty_score_list[i]
            elif uncertainty_type == "ig_text":
                result_data["ig_text_entropy"] = uncertainty_score_list[i]
            file_path = os.path.join(self.config["save_dir"], "output.jsonl")
            print(f"Saving to {file_path}")
            self.safe_write(file_path, result_data)
        
        dataset.update_output("pred", pred_answer_list)
        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_func=pred_process_func)                 
        return dataset

    def naive_run(self, dataset, do_eval=True, pred_process_func=None, uncertainty_type=None):        
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
        uncertainty_score_list = []

        for i, input_prompt in enumerate(input_prompts):
            pred_answer = self.generator.generate([input_prompt], uncertainty_type=uncertainty_type)
            pred_answer_list.append(pred_answer[0]['output_text'][0])
            if uncertainty_type == "entropy":
                uncertainty_score_list.append(pred_answer[0]["generation_entropy"])
            elif uncertainty_type == "ig_text":
                uncertainty_score_list.append(pred_answer[0]["ig_score"])
        
        result_data = {}
        for i, item in enumerate(items_list):
            result_data["id"] = item.id
            result_data["question"] = item.question
            result_data["image_id"] = item.image_id
            result_data["ans_full"] = item.golden_answers
            result_data["prediction"] = pred_answer_list[i]
            if uncertainty_type == "entropy":
                result_data["generation_entropy"] = uncertainty_score_list[i]
            elif uncertainty_type == "ig_text":
                result_data["ig_text_entropy"] = uncertainty_score_list[i]
            file_path = os.path.join(self.config["save_dir"], "output.jsonl")
            print(f"Saving to {file_path}")
            self.safe_write(file_path, result_data)
        
        dataset.update_output("pred", pred_answer_list)
        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_func=pred_process_func)                 
        return dataset

        
class OmniSearchIGPipeline(BasicMultiModalPipeline):
    def __init__(self, config, prompt_template=None):
        super().__init__(config, prompt_template)
        self.config = config
        self.model = AutoModelForVision2Seq.from_pretrained(
            config["generator_model_path"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(config["generator_model_path"], local_files_only=True)
        accelerator = Accelerator()
        self.model, self.processor = accelerator.prepare(self.model, self.processor)
        if hasattr(self.processor, "tokenizer"):
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(config["generator_model_path"], local_files_only=True)        
    
    def naive_run(self, dataset, generated_answers_list=None, do_eval=True, pred_process_func=None):        
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
        uncertainty_score_list = []

        for i, input_prompt in enumerate(input_prompts):
            if generated_answers_list is None:
                golden_answer = input_prompt["answers"]
            else:
                golden_answer = generated_answers_list[i]
            attributions = integrated_gradient_process(self.model, self.processor, self.tokenizer, input_prompt, golden_answer)
            
            uncertainty_score_list.append(attributions)
        
        result_data = {}
        for i, item in enumerate(items_list):
            result_data["id"] = item.id
            result_data["question"] = item.question
            result_data["image_id"] = item.image_id
            result_data["ans_full"] = item.golden_answers
            result_data["ig_text_entropy"] = uncertainty_score_list[i]
            file_path = os.path.join(self.config["save_dir"], "output.jsonl")
            print(f"Saving to {file_path}")
            self.safe_write(file_path, result_data)
        
        # dataset.update_output("pred", pred_answer_list)
        # dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_func=pred_process_func)                 
        # return dataset

    def run(self, dataset, do_eval=True, pred_process_func=None, prompt_answer_path=None):
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
        messages_prompt_list = []
        uncertainty_score_list = []
        id_list = []

        with open(prompt_answer_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                pred_answer_list.append(data.get("prediction"))
                messages_prompt_list.append(data.get("context"))
                id_list.append(data.get("id"))

        for _id, (messages, pred_answer) in zip(id_list, zip(messages_prompt_list, pred_answer_list)):
            image_path = os.path.join(self.config["image_path"], f"{_id}.jpg")
            image = Image.open(image_path).convert('RGB')
            attributions = integrated_gradient_process(self.model, self.processor, self.tokenizer, messages, pred_answer)
            uncertainty_score_list.append(attributions)

        result_data = {}
        for i, item in enumerate(items_list):
            result_data["id"] = item.id
            result_data["question"] = item.question
            result_data["image_id"] = item.image_id
            result_data["ans_full"] = item.golden_answers
            result_data["prediction"] = pred_answer_list[i]
            result_data["context"] = messages_prompt_list[i]
            result_data["ig_entropy"] = uncertainty_score_list[i]
            file_path = os.path.join(self.config["save_dir"], "output.jsonl")
            print(f"Saving to {file_path}")
            self.safe_write(file_path, result_data)
        
        dataset.update_output("pred", pred_answer_list)
        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_func=pred_process_func)                 
        return dataset
    
    def safe_write(self, file_path: str, data: dict):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


    