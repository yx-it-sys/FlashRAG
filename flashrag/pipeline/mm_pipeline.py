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
    
    def self_run(self, dataset, search_type = "image", do_eval=True, pred_process_func=None):
        questions = dataset.question
        ids = dataset.id
        prediction_list = []

        for i, (question, id) in enumerate(zip(questions, ids)):
            try:
                print(f"[{i}/{len(questions)}] Processing ID: {id}")
                print(f"Question: {question}")

                image_path = f"{self.config['image_path']}/{id}.jpg"
                with Image.open(image_path) as raw_image:
                    raw_image = raw_image.convert("RGB")

                    if max(raw_image.size) > 1024:
                        raw_image.thumbnail((1024, 1024))
                    
                    query_image = raw_image.copy()
                
                if search_type == "image":
                    print("Retrieved by Image...")
                    retrieved_contents = self.retriever.search_by_image(query_image)
                    print(f"Retrieved Contents Count: {len(retrieved_contents)}")
                    retrieved_content = "\n\n".join([f"Doc{i+1}:\n{text}" for i, text in enumerate(retrieved_contents)])
                elif search_type == "text":
                    print("Retrieved by Texts...")
                    retrieved_contents = self.retriever.search_by_text(question)
                    print(f"Retrieved Contents Count: {len(retrieved_contents)}")
                    retrieved_content = "\n\n".join([f"Doc{i+1}:\n{text}" for i, text in enumerate(retrieved_contents)])
                else:
                    print(f"ERROR! search type {search_type} is neither 'image' nor 'text'!")
                    retrieved_content = ""
                
                messages = [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text", "text": "You are an intelligent AI Q&A assistant. Based on Supporting Materials, answer the User's Question from provided image carefully and accurately. Do not use your own knowledge to answer."
                            },
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"User's Question:\n{question}\nSupporting Materials:\n{retrieved_content}"},
                            {"type": "image", "image": query_image}
                        ]
                    }
                ]
                response = self.generator.generate(messages)
                print(f"Response: {response}")
                prediction_list.append(response)
            
            except Exception as e:
                print(f"Error processing ID {id}: {e}")
                prediction_list.append("")
                torch.cuda.empty_cache()
                continue
            
        dataset.update_output("pred", prediction_list)
        dataset = self.evaluate(dataset, do_eval=do_eval)                 
        return dataset

