from flashrag.evaluator import Evaluator
from flashrag.utils import get_retriever, get_generator
import re
import os
import json
import torch
from PIL import Image
from tqdm import tqdm

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

class MMCluePipeline(BasicMultiModalPipeline):
    def __init__(self, config, visual_clue_prompt_template, prompt_template=None, retriever=None, generator=None):
        super().__init__(config, prompt_template)
        self.visual_clue_prompt_template = visual_clue_prompt_template
        self.generator = get_generator(config) if generator is None else generator
        self.retriever = retriever
    def _parse_json_string(self, raw_str):
        if not isinstance(raw_str, str):
            return raw_str
        content = raw_str.strip()
        json_pattern = re.compile(r'(\[.*\]|\{.*\})', re.DOTALL)
        match = json_pattern.search(content)
        if match:
            json_str = match.group(1)
        else:
            json_str = content.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print(f"Warning: Failed to parse JSON string. Raw snippet: {json_str}...")
            return f"Failure JSON:{json_str}"
    def get_clue(self, dataset):
        input_prompts = [
            self.visual_clue_prompt_template.get_string_for_visual_clues(item) for item in dataset
        ]   
        raw_clue_list = self.generator.generate(input_prompts)
        parsed_clue_list = [self._parse_json_string(c) for c in raw_clue_list]
        output_dir = self.config['output_dir'] if 'output_dir' in self.config and self.config['output_dir'] else self.config['save_dir']
        os.makedirs(output_dir, exist_ok=True)
        clue_jsonl_path = os.path.join(output_dir, 'clue.jsonl')
        perf_stats = self.generator.get_performance_stats(reset=True)
        with open(os.path.join(output_dir, 'generate_stats.json'), 'w', encoding='utf-8') as f:
            json.dump(perf_stats, f, ensure_ascii=False, indent=4)
        
        with open(clue_jsonl_path, 'w', encoding='utf-8') as f:
            for item, parsed_clue in tqdm(zip(dataset, parsed_clue_list), desc="Saving clues", total=len(dataset)):
                json.dump({
                    'data_id': item.data_id,
                    'image_id': item.image_id,
                    'question': item.question,
                    'clue': parsed_clue, 
                }, f, ensure_ascii=False)
                f.write('\n')
                
        return parsed_clue_list
    def img_retrieval(self, dataset):
        if self.retriever is None:
            raise ValueError("Retriever is not provided for image retrieval.")
        output_dir = self.config['output_dir'] if 'output_dir' in self.config and self.config['output_dir'] else self.config['save_dir']
        os.makedirs(output_dir, exist_ok=True)
        retrieval_jsonl_path = os.path.join(output_dir, 'retrieval_results.jsonl')
        
        with open(retrieval_jsonl_path, 'w', encoding='utf-8') as f:
            for item in tqdm(dataset, desc="Image Retrieval", total=len(dataset)):
                data_id = item.data_id
                image_id = item.image_id
                image_path = os.path.join(f'{self.config["dataset_image_dir"]}', f'{image_id}.jpg')
                retrieval_result = self.retriever.search(image_path, target_modal="text")
                json.dump({
                    'data_id': data_id,
                    'image_id': image_id,
                    'retrieval_results': retrieval_result,
                }, f, ensure_ascii=False)
                f.write('\n')
    def reranking(self, clue_path, retrieval_results_path, total_budget=100):
        with open(clue_path, 'r', encoding='utf-8') as f:
            clue_data = [json.loads(line) for line in f]
        with open(retrieval_results_path, 'r', encoding='utf-8') as f:
            retrieval_data = {json.loads(line)['data_id']: json.loads(line) for line in f}
        reranked_results = []
        for item in clue_data:
            data_id = item['data_id']
            clues = item['clue']
            retrieval_results = retrieval_data.get(data_id, {}).get('retrieval_results', [])
            if not retrieval_results:
                print(f"Warning: No retrieval results found for data_id {data_id}. Skipping reranking.")
                continue
            for clue in clues:
                budget = total_budget * clue.get('importance', 0.0)
                n_v = 
        #     messages = [
        #         {
        #             "role": "system",
        #             "content": [
        #                 {"type": "text", "text": "You are an intelligent assistant for reranking retrieved documents based on the visual clue. Given the visual clue and retrieved documents, you need to rerank the retrieved documents based on their relevance to the visual clue. Please return the top 3 most relevant documents."}
        #             ]
        #         },
        #         {
        #             "role": "user",
        #             "content": [
        #                 {"type": "text", "text": f"Visual Clue:\n{clue}\nRetrieved Documents:\n" + "\n\n".join([f"Doc{i+1}:\n{text}" for i, text in enumerate(retrieval_results)])}
        #             ]
        #         }
        #     ]
        #     response = self.generator.generate(messages)
        #     reranked_results.append({
        #         'data_id': data_id,
        #         'reranked_results': response
        #     })
        # output_dir = self.config['output_dir'] if 'output_dir' in self.config and self.config['output_dir'] else self.config['save_dir']
        # os.makedirs(output_dir, exist_ok=True)
        # reranked_jsonl_path = os.path.join(output_dir, 'reranked_results.jsonl')
        # with open(reranked_jsonl_path, 'w', encoding='utf-8') as f:
        #     for item in reranked_results:
        #         json.dump(item, f, ensure_ascii=False)
        #         f.write('\n')

    


        
            

            



