from flashrag.evaluator import Evaluator
from flashrag.utils import get_retriever, get_generator
import re
import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import time
from itertools import islice



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
    def __init__(self, config, visual_clue_prompt_template, naive_prompt_template, rag_prompt_template, query_generate_prompt_template,retriever=None, generator=None):
        super().__init__(config, naive_prompt_template)
        self.visual_clue_prompt_template = visual_clue_prompt_template
        self.naive_prompt_template = naive_prompt_template
        self.query_generate_prompt_template = query_generate_prompt_template
        self.rag_prompt_template = rag_prompt_template
        # self.generator = get_generator(config) if generator is None else generator
        self.retriever = get_retriever(config) if retriever is None else retriever
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
                retrieval_result = self.retriever._search(image_path, target_modal="text")
                retrieval_result = [result['text'] for result in retrieval_result]
                json.dump({
                    'data_id': data_id,
                    'image_id': image_id,
                    'retrieval_results': retrieval_result,
                }, f, ensure_ascii=False)
                f.write('\n')
    def reranking(self, clue_path, retrieval_results_path, total_budget=1000):
        stats = {}  # 用于存储性能数据
        overall_start = time.time()

        # 1. 数据读取阶段 (仅读取前10行)
        io_start = time.time()
        with open(clue_path, 'r', encoding='utf-8') as f:
            clue_data = [json.loads(line) for line in f]
        with open(retrieval_results_path, 'r', encoding='utf-8') as f:
            retrieval_data = {json.loads(line)['data_id']: json.loads(line) for line in f}

        # with open(clue_path, 'r', encoding='utf-8') as f:
        #     clue_data = [json.loads(line) for line in islice(f, 1)]
        # with open(retrieval_results_path, 'r', encoding='utf-8') as f:
        #     retrieval_data = {json.loads(line)['data_id']: json.loads(line) for line in islice(f, 1)}
        stats['io_read_time'] = time.time() - io_start

        reranked_results = []
        processing_times = [] # 记录单条数据的处理时间

        # 2. 核心重排序阶段
        for item in clue_data:
            item_start = time.time()
            
            data_id = item['data_id']
            clues = item['clue']
            retrieval_results = retrieval_data.get(data_id, {}).get('retrieval_results', [])
            
            if not retrieval_results:
                print(f"Warning: No retrieval results found for data_id {data_id}. Skipping reranking.")
                continue

            # --- 计算逻辑开始 ---
            # Build Budget Matrix
            budget_matrix = np.zeros((1, len(clues)))
            total_importance = sum(c.get('importance', 0.0) for c in clues) + 1e-8
            for i, clue in enumerate(clues):
                budget_matrix[0, i] = total_budget * clue.get('importance', 0.0) / total_importance

            # Build preference matrix
            preference_matrix = np.zeros((len(clues), len(retrieval_results)))
            for i, clue in enumerate(clues):
                clue_text = clue.get('clue', '')
                for j, doc in enumerate(retrieval_results):
                    preference_matrix[i, j] = self.cosine_similarity(clue_text, doc)

            # Build Voting Matrix
            voting_matrix = np.zeros((len(clues), len(retrieval_results)))
            for i in range(len(clues)):
                row_scores = preference_matrix[i]
                denominator = np.sqrt(np.sum(np.square(row_scores))) + 1e-8
                voting_matrix[i] = (
                    np.square(row_scores) * np.sqrt(max(budget_matrix[0, i], 0.0)) / denominator
                )

            # Reranking process
            doc_scores = np.sum(voting_matrix, axis=0)
            sorted_doc_indices = np.argsort(-doc_scores)
            # --- 计算逻辑结束 ---

            reranked_results.append({
                'data_id': data_id,
                'image_id': item.get('image_id'),
                'reranked_results': [
                    {
                        'text': retrieval_results[idx],
                        'score': float(doc_scores[idx])
                    } for idx in sorted_doc_indices
                ],
            })
            
            processing_times.append(time.time() - item_start)

        # 统计计算阶段数据
        stats['avg_item_processing_time'] = np.mean(processing_times) if processing_times else 0
        stats['total_processing_time'] = sum(processing_times)
        stats['num_items_processed'] = len(reranked_results)

        # 3. 保存结果阶段
        save_start = time.time()
        output_dir = self.config['save_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存重排序结果
        rerank_jsonl_path = os.path.join(output_dir, 'reranked_results.jsonl')
        with open(rerank_jsonl_path, 'w', encoding='utf-8') as f:
            for row in reranked_results:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
                
        stats['io_write_time'] = time.time() - save_start
        stats['overall_total_time'] = time.time() - overall_start

        # 4. 保存性能统计到 JSONL
        stats_path = os.path.join(output_dir, 'performance_stats.jsonl')
        with open(stats_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(stats, ensure_ascii=False) + '\n')

        print(f"Reranking complete. Stats saved to {stats_path}")
        return reranked_results
    def cosine_similarity(self, clue, doc):
        if not clue or not doc:
            return 0.0

        encoder = getattr(self.retriever, 'encoder', None)
        if encoder is None:
            raise ValueError('Retriever does not expose encoder; cannot compute embedding cosine similarity.')

        input_texts = [clue, doc]
        try:
            embeddings = encoder.encode(input_texts, modal='text')
        except TypeError:
            embeddings = encoder.encode(input_texts, is_query=True)

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.shape[0] < 2:
            return 0.0

        clue_emb = embeddings[0]
        doc_emb = embeddings[1]
        denominator = np.linalg.norm(clue_emb) * np.linalg.norm(doc_emb) + 1e-8
        return float(np.dot(clue_emb, doc_emb) / denominator)

    def _compute_normalized_nll(self, token_probs_list):
        normalized_nll = []
        for token_probs in token_probs_list:
            if not token_probs:
                normalized_nll.append(None)
                continue
            probs = np.asarray(token_probs, dtype=np.float64)
            probs = np.clip(probs, 1e-12, 1.0)
            normalized_nll.append(float(-np.mean(np.log(probs))))
        return normalized_nll

    def _extract_reference_text(self, doc):
        if isinstance(doc, str):
            return doc
        if isinstance(doc, dict):
            if "text" in doc and doc["text"] is not None:
                return doc["text"]
            if "contents" in doc and doc["contents"] is not None:
                return doc["contents"]
        return ""

    def run(self, dataset, retrieval_threshold=0.0, reranked_results_path=None, do_eval=True, pred_process_func=None):
        # To Do: 需要统计性能
        reranked_results = {}
        with open(reranked_results_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                reranked_results[item['data_id']] = item['reranked_results']
        pred_answer_list = []
        entity_docs = [reranked_results[item.data_id][0] for item in dataset]
        input_prompts = [
            self.visual_clue_prompt_template.get_string_for_rag_retrieval(item, reference_doc) for item, reference_doc in zip(dataset, entity_docs)
        ]   
        token_probs = None
        pred_answer_list, token_probs = self.generator.generate(input_prompts, return_scores=True)

        normalized_nll = self._compute_normalized_nll(token_probs)

        items_need_retrieval = [
            {"item":item, "reranked_doc": reranked_results[item.data_id][0]} for item, nll in zip(dataset.data, normalized_nll)
            if nll is not None and nll > retrieval_threshold
        ]

        # Text Query Generator
        input_prompts_for_query_gen = [
            self.query_generate_prompt_template.get_string_for_query_generation(item["item"], item["reranked_doc"]) for item in items_need_retrieval
        ]
        generated_queries = self.generator.generate(input_prompts_for_query_gen)
        # To Do: Text Retrieval
        dataset.update_output("pred", pred_answer_list)
        # To Do: RAG Pipeline

        # To Do: Update pred_answer_list with RAG results for items that exceed the NLL threshold









