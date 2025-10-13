from flashrag.evaluator import Evaluator
from flashrag.utils import get_retriever, get_generator
import re
import os
import json

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
    
    def iterative_infer(self, input_prompt, query):
        response_dict = self.generator.generate([input_prompt], return_dict=True)
        response = response_dict["responses"][0]
        print(f"First Response: {response_dict}")
        input_prompt.append({'role': 'assistant', 'content': response})
        conversation_num, max_turns = 0, 5

        while conversation_num < max_turns:
            if "Final Answer" in response or "<Final Answer>" in response:
                break
            need_txt_ret = "Text Retrieval" in response
            if need_txt_ret:
                print("Start text retrieval...")
                query_txt = (
                    response.split("Text Retrieval")[-1]
                    .replace(":", "")
                    .replace('"', "")
                    .replace(">", "")
                )
                if query_txt is not None:
                    search_text = self.retriever.search([query_txt], 1)
                    search_text = search_text[0]["contents"]
                    print(f"Retrieval result: {search_text}")
                else:
                    search_text = self.retriever.search([query], 2)
                    print(f"Retrieval result: {search_text}")
                contents = []
                if search_text:
                    contents.append({'type': 'text', 'text': f"Contents of retrieved documents:\n{' '.join(search_text)}"})
                else:
                    contents.append({'type': 'text', 'text': "No relevant information found."})    

                input_prompt.append({'role': 'user', 'content': contents})

                try:
                    response_dict = self.generator.generate([input_prompt], return_dict=True)
                    response = response_dict["responses"][0]
                    print(f"response dict: {response_dict}")
                    input_prompt.append({"role":"assistant", "content": response})
                except Exception as e:
                    print("Inference error:", e)
                    return response, input_prompt
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
            return final_answer, input_prompt
        else:
            print(f"Warning: reached end of agent loop for item {conversation_num} without a 'Final Answer'. returning last response")
            return response, input_prompt
      
    def safe_write(self, file_path: str, data: dict):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            
    def run(self, dataset, do_eval=True, pred_process_func=None):
        input_prompts = []
        query_list = []
        items_list = list(dataset)
        for item in dataset:
            input_prompts.append(
                self.prompt_template.get_string(item, self.config)
            )
            query_list.append(
                item.question
            )

        pred_answer_list = []
        context_list = []
        for i, input_prompt in enumerate(input_prompts):
            answer, context = self.iterative_infer(input_prompt, query_list[i])
            remove_image_context = context[2:]
            pred_answer_list.append(answer)
            context_list.append(remove_image_context)
            # print(f"Answer: {answer}")

        for i, item in enumerate(items_list):
            result_data = {
                "id": item.id,
                "question": item.question,
                "image_id": item.image_id,
                "ans_full": item.golden_answers,
                "prediction": pred_answer_list[i],
                "context": context_list[i],
            }
            file_path = os.path.join(self.config["save_dir"], "output.jsonl")
            print(f"Saving to {file_path}")
            self.safe_write(file_path, result_data)
        
        dataset.update_output("pred", pred_answer_list)
        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_func=pred_process_func)                 
        return dataset
    
    def uncertainty(self, dataset):
        pass
