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
    def __init__(self, config, visual_clue_prompt_template=None, naive_prompt_template=None, rag_prompt_template=None, query_generate_prompt_template=None,retriever=None, generator=None):
        super().__init__(config, naive_prompt_template)
        self.visual_clue_prompt_template = visual_clue_prompt_template
        self.naive_prompt_template = naive_prompt_template
        self.query_generate_prompt_template = query_generate_prompt_template
        self.rag_prompt_template = rag_prompt_template
        self.generator = generator
        self.retriever = retriever

    def _get_output_dir(self):
        output_dir = self.config['output_dir'] if 'output_dir' in self.config and self.config['output_dir'] else self.config['save_dir']
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _to_jsonable(self, value):
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {str(k): self._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._to_jsonable(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if hasattr(value, '__dict__'):
            return self._to_jsonable(vars(value))
        return str(value)

    def _save_phase_jsonl(self, phase_name, records, file_name=None, append=False):
        output_dir = self._get_output_dir()
        target_name = file_name if file_name is not None else f"{phase_name}.jsonl"
        file_path = os.path.join(output_dir, target_name)
        mode = 'a' if append else 'w'

        if isinstance(records, dict):
            records = [records]

        with open(file_path, mode, encoding='utf-8') as f:
            for record in records:
                json.dump(self._to_jsonable(record), f, ensure_ascii=False)
                f.write('\n')
        return file_path

    def _save_phase_json(self, phase_name, payload, file_name=None):
        output_dir = self._get_output_dir()
        target_name = file_name if file_name is not None else f"{phase_name}.json"
        file_path = os.path.join(output_dir, target_name)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self._to_jsonable(payload), f, ensure_ascii=False, indent=4)
        return file_path

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

    def _extract_doc_title(self, doc_text):
        if not isinstance(doc_text, str):
            return ""

        content = doc_text.strip()
        if not content:
            return ""

        match = re.match(r'^\s*(.*?)\s+-\s+.+$', content, flags=re.DOTALL)
        if match:
            return match.group(1).strip()

        return content

    def get_clue(self, dataset):
        input_prompts = [
            self.visual_clue_prompt_template.get_string_for_visual_clues(item) for item in dataset
        ]   
        raw_clue_list = self.generator.generate(input_prompts)
        parsed_clue_list = [self._parse_json_string(c) for c in raw_clue_list]
        clue_jsonl_rows = []
        perf_stats = self.generator.get_performance_stats(reset=True)
        for item, parsed_clue in tqdm(zip(dataset, parsed_clue_list), desc="Saving clues", total=len(dataset)):
            clue_jsonl_rows.append({
                'data_id': item.data_id,
                'image_id': item.image_id,
                'question': item.question,
                'clue': parsed_clue,
            })

        self._save_phase_json('phase1_clue_mining_stats', perf_stats, file_name='generate_stats.json')
        self._save_phase_jsonl('phase1_clue_mining', clue_jsonl_rows, file_name='clue.jsonl')
                
        return parsed_clue_list
    
    def caption_retrieval(self, dataset):
        if self.retriever is None:
            raise ValueError("Retriever is not provided for caption retrieval.")
        if self.generator is None:
            raise ValueError("Generator is not provided for caption retrieval.")

        output_dir = self._get_output_dir()
        image_phase_dir = os.path.join("data/result/infoseek_v6", "phase2_image_retrieval")
        caption_phase_dir = os.path.join("data/result/infoseek_v6", "phase2_image_caption_retrieval")
        os.makedirs(image_phase_dir, exist_ok=True)
        os.makedirs(caption_phase_dir, exist_ok=True)

        image_retrieval_path = os.path.join(image_phase_dir, "image_retrieval_results.jsonl")
        merged_retrieval_path = os.path.join(caption_phase_dir, "final_retrieval_results.jsonl")
        if not os.path.exists(image_retrieval_path):
            raise FileNotFoundError(
                f"Image retrieval file not found: {image_retrieval_path}. Please run img_retrieval first."
            )

        image_retrieval_rows = []
        with open(image_retrieval_path, 'r', encoding='utf-8') as f:
            for line in f:
                image_retrieval_rows.append(json.loads(line))

        row_by_data_id = {row.get('data_id'): row for row in image_retrieval_rows}

        # Step 1: 集中生成全部 item 的 caption（顺序与 dataset 保持一致）
        text_prompt = "Describe the image in detail"
        items = list(dataset)
        caption_inputs = []
        item_metas = []
        for item in tqdm(items, desc="Prepare Caption Inputs", total=len(items)):
            image_path = os.path.join(f'{self.config["dataset_image_dir"]}', f'{item.image_id}.jpg')
            img = Image.open(image_path).convert("RGB")
            caption_inputs.append([
                {"role": "user", "content": [{"type": "text", "text": text_prompt}, {"type": "image", "image": img}]}
            ])
            item_metas.append({
                'data_id': item.data_id,
                'image_id': item.image_id,
            })

        captions = self.generator.generate(caption_inputs) if caption_inputs else []
        if len(captions) != len(item_metas):
            raise ValueError(f"Caption count mismatch: got {len(captions)} captions for {len(item_metas)} items.")

        # Step 2: 集中用 caption 做检索（顺序与 captions 对齐）
        try:
            batch_retrieval_result = self.retriever.batch_search(captions)
        except Exception:
            batch_retrieval_result = [self.retriever.search(caption) for caption in captions]

        if len(batch_retrieval_result) != len(item_metas):
            raise ValueError(
                f"Retrieval result count mismatch: got {len(batch_retrieval_result)} results for {len(item_metas)} items."
            )

        # Step 3: 将检索内容按 data_id 回填到原始结果，保证和原数据匹配
        for meta, retrieval_result in zip(item_metas, batch_retrieval_result):
            retrieval_texts = []
            for result in retrieval_result:
                retrieval_texts.append({'id': result.get('id'), 'text': result.get('text', '')})

            data_id = meta['data_id']
            if data_id not in row_by_data_id:
                new_row = {
                    'data_id': data_id,
                    'image_id': meta['image_id'],
                    'retrieval_results': []
                }
                image_retrieval_rows.append(new_row)
                row_by_data_id[data_id] = new_row

            target_row = row_by_data_id[data_id]
            existing_docs_dict = target_row.get('retrieval_results', [])
            if not isinstance(existing_docs_dict, list):
                existing_docs_dict = []

            # Backward compatibility: normalize mixed legacy formats to {id, text}.
            normalized_docs = []
            existing_set = set()
            for doc in existing_docs_dict:
                doc_id = doc.get('id')
                doc_text = doc.get('text', '')

                dedup_key = str(doc_id)
                if dedup_key in existing_set:
                    continue
                existing_set.add(dedup_key)
                normalized_docs.append({'id': doc_id, 'text': doc_text})

            for doc in retrieval_texts:
                doc_id = doc.get('id')
                doc_text = doc.get('text', '')
                dedup_key =  str(doc_id)
                if dedup_key in existing_set:
                    continue
                existing_set.add(dedup_key)
                normalized_docs.append({'id': doc_id, 'text': doc_text})

            target_row['retrieval_results'] = normalized_docs

        with open(merged_retrieval_path, 'w', encoding='utf-8') as f:
            for row in image_retrieval_rows:
                json.dump(self._to_jsonable(row), f, ensure_ascii=False)
                f.write('\n')
        return image_retrieval_rows
            
    def img_retrieval(self, dataset):
        if self.retriever is None:
            raise ValueError("Retriever is not provided for image retrieval.")

        output_dir = self._get_output_dir()
        image_phase_dir = os.path.join(output_dir, "phase2_image_retrieval")
        os.makedirs(image_phase_dir, exist_ok=True)
        image_retrieval_path = os.path.join(image_phase_dir, "image_retrieval_results.jsonl")

        retrieval_rows = []
        for item in tqdm(dataset, desc="Image Retrieval", total=len(dataset)):
            data_id = item.data_id
            image_id = item.image_id
            image_path = os.path.join(f'{self.config["dataset_image_dir"]}', f'{image_id}.jpg')
            retrieval_result = self.retriever._search(image_path, target_modal="text")
            retrieval_result = [{'id': result['id'], 'text': result['text']} for result in retrieval_result]
            retrieval_rows.append({
                'data_id': data_id,
                'image_id': image_id,
                'retrieval_results': retrieval_result,
            })

        with open(image_retrieval_path, 'w', encoding='utf-8') as f:
            for row in retrieval_rows:
                json.dump(self._to_jsonable(row), f, ensure_ascii=False)
                f.write('\n')
        return retrieval_rows

    def reranking(self, clue_path, retrieval_results_path, total_budget=1000):
        stats = {}  # 用于存储性能数据
        overall_start = time.time()
        cluster_doc_topk = self._get_int_config("phase3_cluster_doc_topk", 2)
        early_stop_topk = self._get_int_config("phase3_early_stop_topk", 5)
        preference_threshold = float(self.config["phase3_preference_threshold"])

        # 1. 数据读取阶段 (仅读取前10行)
        io_start = time.time()
        with open(clue_path, 'r', encoding='utf-8') as f:
            clue_data = [json.loads(line) for line in f]
        with open(retrieval_results_path, 'r', encoding='utf-8') as f:
            retrieval_data = {}
            for line in f:
                row = json.loads(line)
                retrieval_data[row['data_id']] = row

        # with open(clue_path, 'r', encoding='utf-8') as f:
        #     clue_data = [json.loads(line) for line in islice(f, 1)]
        # with open(retrieval_results_path, 'r', encoding='utf-8') as f:
        #     retrieval_data = {json.loads(line)['data_id']: json.loads(line) for line in islice(f, 1)}
        
        stats['io_read_time'] = time.time() - io_start

        reranked_results = []
        processing_times = [] # 记录单条数据的处理时间
        total_full_comparisons = 0
        total_actual_comparisons = 0
        total_early_stopped_docs = 0

        # 2. 核心重排序阶段
        for item in clue_data:
            item_start = time.time()
            
            data_id = item['data_id']
            clues = item['clue']
            if not clues:
                print(f"Warning: No clues found for data_id {data_id}. Skipping reranking.")
                continue
            retrieval_row = retrieval_data.get(data_id, {})
            retrieval_results = retrieval_row.get('retrieval_results')
            if not retrieval_results:
                retrieval_results = retrieval_row.get('caption_retrieval_results', [])
            
            if not retrieval_results:
                print(f"Warning: No retrieval results found for data_id {data_id}. Skipping reranking.")
                continue

            # --- 计算逻辑开始 ---
            # Doc-level early stopping: when a doc cannot reach current top-k even with max remaining gain,
            # stop further clue->doc comparisons for that doc.
            num_docs = len(retrieval_results)
            num_clues = len(clues)
            total_full_comparisons += num_docs * num_clues

            importances = [float(c.get('importance', 1.0)) for c in clues]
            total_importance = sum(importances) + 1e-8
            clue_budget_weights = [
                np.sqrt(max(total_budget * (imp / total_importance), 0.0)) for imp in importances
            ]

            # Remaining theoretical max gain upper bound from clue i+1 ... end
            remaining_gain_ub = np.zeros(num_clues + 1, dtype=np.float32)
            for idx in range(num_clues - 1, -1, -1):
                remaining_gain_ub[idx] = remaining_gain_ub[idx + 1] + clue_budget_weights[idx]

            doc_scores = np.zeros(num_docs, dtype=np.float32)
            active_mask = np.ones(num_docs, dtype=bool)
            item_actual_comparisons = 0
            item_early_stopped_docs = 0

            for clue_idx, clue in enumerate(clues):
                if not np.any(active_mask):
                    break

                clue_text = clue.get('clue', '')
                row_scores = np.zeros(num_docs, dtype=np.float32)
                active_indices = np.where(active_mask)[0]

                for j in active_indices:
                    # score = self.cosine_similarity(clue_text, retrieval_results[j])
                    score = self.llm_nli_judge(clue_text, retrieval_results[j]['text'])
                    item_actual_comparisons += 1
                    if preference_threshold > 0.0 and score < preference_threshold:
                        continue
                    row_scores[j] = score

                denominator = np.sqrt(np.sum(np.square(row_scores))) + 1e-8
                if denominator > 0.0:
                    row_votes = (np.square(row_scores) * clue_budget_weights[clue_idx]) / denominator
                    doc_scores += row_votes

                effective_k = min(early_stop_topk, num_docs)
                if effective_k > 0:
                    kth_score = float(np.partition(doc_scores, -effective_k)[-effective_k])
                    future_ub = float(remaining_gain_ub[clue_idx + 1])
                    cannot_reach_topk = (doc_scores + future_ub) < kth_score
                    newly_stopped = cannot_reach_topk & active_mask
                    if np.any(newly_stopped):
                        active_mask[newly_stopped] = False
                        item_early_stopped_docs += int(np.sum(newly_stopped))

            total_actual_comparisons += item_actual_comparisons
            total_early_stopped_docs += item_early_stopped_docs

            # Title clustering process (regex-based parsing)
            title_cluster_scores = {}
            title_cluster_best_doc_idx = {}
            title_cluster_doc_indices = {}
            for j, doc in enumerate(retrieval_results):
                title = self._extract_doc_title(doc)
                title_cluster_scores[title] = title_cluster_scores.get(title, 0.0) + float(doc_scores[j])
                if title not in title_cluster_doc_indices:
                    title_cluster_doc_indices[title] = []
                title_cluster_doc_indices[title].append(j)
                if title not in title_cluster_best_doc_idx or float(doc_scores[j]) > float(doc_scores[title_cluster_best_doc_idx[title]]):
                    title_cluster_best_doc_idx[title] = j

            sorted_title_clusters = sorted(
                title_cluster_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            best_cluster_title = sorted_title_clusters[0][0] if sorted_title_clusters else ""

            # Keep all clusters; select top-k docs inside each cluster
            cluster_results = []
            reranked_flat_results = []
            for title, score in sorted_title_clusters:
                cluster_doc_indices = title_cluster_doc_indices.get(title, [])
                cluster_doc_indices = sorted(
                    cluster_doc_indices,
                    key=lambda idx: float(doc_scores[idx]),
                    reverse=True,
                )

                if cluster_doc_topk > 0:
                    selected_indices = cluster_doc_indices[:cluster_doc_topk]
                else:
                    selected_indices = cluster_doc_indices

                cluster_docs = [
                    {
                        'text': retrieval_results[idx],
                        'title': self._extract_doc_title(retrieval_results[idx]),
                        'score': float(doc_scores[idx])
                    }
                    for idx in selected_indices
                ]

                cluster_results.append({
                    'title': title,
                    'score': float(score),
                    'docs': cluster_docs,
                })
                reranked_flat_results.extend(cluster_docs)

            best_cluster_doc_idx = title_cluster_best_doc_idx.get(best_cluster_title)
            # --- 计算逻辑结束 ---
            

            reranked_results.append({
                'data_id': data_id,
                'image_id': item.get('image_id'),
                'top_title_cluster': best_cluster_title,
                'top_cluster_best_doc': {
                    'text': retrieval_results[best_cluster_doc_idx],
                    'title': self._extract_doc_title(retrieval_results[best_cluster_doc_idx]),
                    'score': float(doc_scores[best_cluster_doc_idx])
                } if best_cluster_doc_idx is not None else None,
                'title_clusters': [
                    {
                        'title': title,
                        'score': float(score)
                    } for title, score in sorted_title_clusters
                ],
                'cluster_results': cluster_results,
                'reranked_results': reranked_flat_results,
            })
            
            processing_times.append(time.time() - item_start)

        # 统计计算阶段数据
        stats['avg_item_processing_time'] = np.mean(processing_times) if processing_times else 0
        stats['total_processing_time'] = sum(processing_times)
        stats['num_items_processed'] = len(reranked_results)
        stats['full_comparisons'] = int(total_full_comparisons)
        stats['actual_comparisons'] = int(total_actual_comparisons)
        stats['saved_comparisons'] = int(max(total_full_comparisons - total_actual_comparisons, 0))
        stats['comparison_reduction_ratio'] = float(
            (max(total_full_comparisons - total_actual_comparisons, 0) / total_full_comparisons)
            if total_full_comparisons > 0 else 0.0
        )
        stats['early_stopped_docs'] = int(total_early_stopped_docs)

        # 3. 保存结果阶段
        save_start = time.time()
        # 保存重排序结果
        self._save_phase_jsonl('phase3_reranking', reranked_results, file_name='reranked_results.jsonl')
                
        stats['io_write_time'] = time.time() - save_start
        stats['overall_total_time'] = time.time() - overall_start

        # 4. 保存性能统计到 JSONL
        stats_path = self._save_phase_jsonl('phase3_reranking_stats', stats, file_name='performance_stats.jsonl', append=True)

        print(f"Reranking complete. Stats saved to {stats_path}")
        return reranked_results
        
    def naive_reranking(self, clue_path, retrieval_results_path):
        # Naive reranking: for each clue, ask the LLM to vote Yes/No for each retrieved doc,
        # then rank docs by accumulated votes.
        stats = {}

        with open(clue_path, 'r', encoding='utf-8') as f:
            clue_data = [json.loads(line) for line in f]
        with open(retrieval_results_path, 'r', encoding='utf-8') as f:
            retrieval_data = {}
            for line in f:
                row = json.loads(line)
                retrieval_data[row['data_id']] = row

        reranked_results = []
        vote_logs = []
        processing_times = []
        total_binary_judges = 0

        for item in clue_data:
            item_start = time.time()

            data_id = item.get('data_id')
            image_id = item.get('image_id')
            raw_clues = item.get('clue', [])
            if not isinstance(raw_clues, list) or len(raw_clues) == 0:
                print(f"Warning: No clues found for data_id {data_id}. Skipping reranking.")
                continue

            retrieval_row = retrieval_data.get(data_id, {})
            retrieval_results = retrieval_row.get('retrieval_results')
            if not retrieval_results:
                retrieval_results = retrieval_row.get('caption_retrieval_results', [])
            if not isinstance(retrieval_results, list) or len(retrieval_results) == 0:
                print(f"Warning: No retrieval results found for data_id {data_id}. Skipping reranking.")
                continue

            # Normalize clues into plain non-empty strings.
            clue_texts = []
            for clue in raw_clues:
                if isinstance(clue, dict):
                    clue_text = clue.get('clue', '')
                else:
                    clue_text = clue
                clue_text = str(clue_text).strip() if clue_text is not None else ''
                if clue_text:
                    clue_texts.append(clue_text)

            if not clue_texts:
                print(f"Warning: Empty clue texts for data_id {data_id}. Skipping reranking.")
                continue

            doc_votes = np.zeros(len(retrieval_results), dtype=np.float32)
            text_query = item.get('question')
            if text_query is None:
                text_query = retrieval_row.get('text_query')

            # Process one clue at a time; docs are sent to vLLM in mini-batches.
            for clue_text in clue_texts:
                clue_votes = []
                for doc_idx, doc in enumerate(retrieval_results):
                    yes_prob = self.llm_nli_judge(clue_text, doc['text'])
                    vote_yes = yes_prob >= 0.5
                    vote = 1 if vote_yes else 0
                    doc_preview = str(doc).replace("\n", " ")[:120]
                    print(
                        f"Naive LLM Judge - Clue: {clue_text[:100]}, DocIdx: {doc_idx}, Doc: {doc_preview}, YesProb: {yes_prob:.4f}, Vote: {'Yes' if vote_yes else 'No'}"
                    )
                    clue_votes.append(vote)

                if len(clue_votes) != len(retrieval_results):
                    raise RuntimeError(
                        f"Naive clue vote size mismatch: got {len(clue_votes)}, expected {len(retrieval_results)}"
                    )
                doc_votes += np.asarray(clue_votes, dtype=np.float32)
                total_binary_judges += len(clue_votes)

            ranked_indices = sorted(
                range(len(retrieval_results)),
                key=lambda idx: (float(doc_votes[idx]), -idx),
                reverse=True,
            )

            reranked_flat_results = []
            for rank_idx, doc_idx in enumerate(ranked_indices):
                doc_text = retrieval_results[doc_idx]
                vote_count = float(doc_votes[doc_idx])
                reranked_flat_results.append({
                    'text': doc_text,
                    'title': self._extract_doc_title(doc_text),
                    'score': vote_count,
                })
                vote_logs.append({
                    'data_id': data_id,
                    'image_id': image_id,
                    'text_query': text_query,
                    'clue': clue_texts,
                    'doc': doc_text,
                    'vote_count': vote_count,
                    'rank': rank_idx + 1,
                })

            reranked_results.append({
                'data_id': data_id,
                'image_id': image_id,
                'reranked_results': reranked_flat_results,
            })

            processing_times.append(time.time() - item_start)

        stats['total_binary_judges'] = int(total_binary_judges)

        self._save_phase_jsonl('phase3_reranking', reranked_results, file_name='reranked_results.jsonl')
        self._save_phase_jsonl('phase3_reranking_vote_logs', vote_logs, file_name='naive_vote_logs.jsonl')

        stats_path = self._save_phase_jsonl('phase3_reranking_stats', stats, file_name='performance_stats.jsonl', append=True)
        print(f"Naive reranking complete. Stats saved to {stats_path}")
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

    def llm_nli_judge(self, clue, doc):
        prompt = f"Clue: {clue}\nDocument: {doc}\nQuestion: Does the document support the clue? Answer 'Yes' or 'No'."
        if self.generator is None:
            raise ValueError("Generator is required for llm_nli_judge.")

        tokenizer = getattr(self.generator, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("Generator does not expose tokenizer; cannot compute Yes/No logits.")

        yes_candidates = []
        no_candidates = []
        for token_text in ["Yes", " Yes"]:
            token_ids = tokenizer.encode(token_text, add_special_tokens=False)
            if len(token_ids) == 1:
                yes_candidates.append(token_ids[0])
        for token_text in ["No", " No"]:
            token_ids = tokenizer.encode(token_text, add_special_tokens=False)
            if len(token_ids) == 1:
                no_candidates.append(token_ids[0])

        if not yes_candidates:
            yes_candidates = [tokenizer.encode("Yes", add_special_tokens=False)[0]]
        if not no_candidates:
            no_candidates = [tokenizer.encode("No", add_special_tokens=False)[0]]

        # vLLM (decoder-only): use first-step token logprobs and convert to Yes-vs-No score.
        requested_logprobs = 20

        try:
            raw_outputs = self.generator.generate(
                [prompt],
                return_raw_output=True,
                max_tokens=1,
                temperature=0,
                logprobs=requested_logprobs,
            )
        except Exception:
            raw_outputs = self.generator.generate(
                [prompt],
                return_raw_output=True,
                max_tokens=1,
                temperature=0,
                logprobs=20,
            )

        if not raw_outputs:
            return 0.0

        first_candidates = getattr(raw_outputs[0], "outputs", None)
        if not first_candidates:
            return 0.0

        first_output = first_candidates[0]
        token_logprobs = getattr(first_output, "logprobs", None)
        if not token_logprobs:
            return 0.0

        first_step = token_logprobs[0] if isinstance(token_logprobs, list) else token_logprobs

        def _get_logprob(step_logprobs, token_ids):
            if not isinstance(step_logprobs, dict):
                return None
            for token_id in token_ids:
                item = step_logprobs.get(token_id)
                if item is not None:
                    return float(getattr(item, "logprob", item))
                item = step_logprobs.get(str(token_id))
                if item is not None:
                    return float(getattr(item, "logprob", item))
            return None

        yes_logprob = _get_logprob(first_step, yes_candidates)
        no_logprob = _get_logprob(first_step, no_candidates)
        if yes_logprob is None or no_logprob is None:
            return 0.0

        yes_tensor = torch.tensor([[yes_logprob, no_logprob]], dtype=torch.float32)
        yes_prob = torch.nn.functional.softmax(yes_tensor, dim=1)[:, 0]
        print(f"LLM NLI Judge - Clue: {clue}, Doc: {doc[:100]}, Yes Prob: {yes_prob.item()}")
        return float(yes_prob.item())


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

    def _get_int_config(self, key, default):
        if key in self.config and self.config[key] is not None:
            return int(self.config[key])
        return default

    def _truncate_text_for_prompt(self, text, max_chars):
        if text is None:
            return ""
        text = str(text)
        if max_chars <= 0:
            return ""
        return text[:max_chars]

    def query_gen_retrieval(self, dataset, reranked_results_path):
        query_ref_max_chars = self._get_int_config("phase4_query_ref_max_chars", 1200)
        reranked_results = {}
        with open(reranked_results_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                reranked_results[item['data_id']] = item
        data_items = list(dataset.data)

        def _select_clusters_by_max_gap(title_clusters):
            if not isinstance(title_clusters, list) or not title_clusters:
                return []

            sorted_clusters = sorted(
                title_clusters,
                key=lambda c: float(c.get('score', 0.0)),
                reverse=True,
            )

            if len(sorted_clusters) <= 2:
                return sorted_clusters

            scores = [float(c.get('score', 0.0)) for c in sorted_clusters]
            gaps = [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]
            max_gap = max(gaps)

            # No clear cliff: keep all clusters
            if max_gap <= 0:
                return sorted_clusters

            split_idx = int(np.argmax(gaps))
            return sorted_clusters[: split_idx + 1]

        entity_docs = []
        selected_clusters_all = []
        for item in data_items:
            row = reranked_results.get(item.data_id, {})
            title_clusters = row.get('title_clusters', [])
            selected_clusters = _select_clusters_by_max_gap(title_clusters)
            selected_clusters_all.append(selected_clusters)

            # Preferred source: new structure with docs grouped by cluster
            docs_by_title = {}
            cluster_results = row.get('cluster_results', [])
            if isinstance(cluster_results, list):
                for cluster in cluster_results:
                    title = cluster.get('title', '')
                    docs = cluster.get('docs', [])
                    docs_by_title[title] = docs if isinstance(docs, list) else []

            selected_doc_texts = []
            for cluster in selected_clusters:
                title = cluster.get('title', '')
                score = float(cluster.get('score', 0.0))

                cluster_texts = []
                for doc in docs_by_title.get(title, []):
                    doc_text = self._extract_reference_text(doc)
                    if doc_text:
                        cluster_texts.append(doc_text)

                # Backward-compatible fallback: recover docs from flat reranked_results by title
                if not cluster_texts:
                    for doc in row.get('reranked_results', []):
                        doc_text = self._extract_reference_text(doc)
                        doc_title = self._extract_doc_title(doc_text)
                        if doc_title == title and doc_text:
                            cluster_texts.append(doc_text)

                if cluster_texts:
                    selected_doc_texts.append(
                        f"[Cluster: {title} | score={score:.4f}]\n" + "\n\n".join(cluster_texts)
                    )

            # Final fallback for extreme/old cases
            if not selected_doc_texts:
                legacy_docs = row.get('reranked_results', [])
                if isinstance(legacy_docs, list) and legacy_docs:
                    fallback_text = self._extract_reference_text(legacy_docs[0])
                    if fallback_text:
                        selected_doc_texts = [fallback_text]

            entity_docs.append(
                self._truncate_text_for_prompt("\n\n".join(selected_doc_texts), query_ref_max_chars)
            )

        # Generate Queries
        query_prompts = [
            self.query_generate_prompt_template.get_string_sep(item, reference_doc)
            for item, reference_doc in zip(data_items, entity_docs)
        ]
        generated_queries = self.generator.generate(query_prompts) if query_prompts else []
        all_retrieval_results = self.retriever.batch_search(generated_queries) if generated_queries else []
        dataset.update_output("retrieval_results", all_retrieval_results)

        phase4_rows = []
        for item, reference_doc, selected_clusters, query_prompt, generated_query, retrieval_results in zip(
            data_items,
            entity_docs,
            selected_clusters_all,
            query_prompts,
            generated_queries,
            all_retrieval_results,
        ):
            phase4_rows.append({
                'data_id': item.data_id,
                'image_id': getattr(item, 'image_id', None),
                'question': getattr(item, 'question', None),
                'reference_doc': reference_doc,
                'selected_title_clusters': selected_clusters,
                'query_prompt': query_prompt,
                'generated_query': generated_query,
                'retrieval_results': retrieval_results,
            })
        self._save_phase_jsonl('phase4_query_gen_retrieval', phase4_rows, file_name='query_gen_retrieval_results.jsonl')
        return dataset

    def rag_run(self, dataset, reranked_results_path, retrieval_threshold=0.0, do_eval=True, pred_process_func=None):
        # 计算不确定度
        naive_ref_max_chars = self._get_int_config("phase4_naive_ref_max_chars", 1200)
        rag_doc_max_chars = self._get_int_config("phase4_rag_doc_max_chars", 300)
        reranked_results = {}
        with open(reranked_results_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                reranked_results[item['data_id']] = item['reranked_results']

        def _load_retrieval_map(file_path):
            result = {}
            if not file_path or not os.path.exists(file_path):
                return result
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    row = json.loads(line)
                    result[row.get('data_id')] = row.get('retrieval_results', [])
            return result

        def _preview_docs_200(docs):
            if not isinstance(docs, list):
                return []
            previews = []
            for idx, doc in enumerate(docs):
                doc_text = self._extract_reference_text(doc)
                previews.append({
                    'rank': idx + 1,
                    'text_preview_200': self._truncate_text_for_prompt(doc_text, 200),
                })
            return previews

        output_dir = self._get_output_dir()
        image_retrieval_path = self.config.get(
            'phase2_image_retrieval_path',
            os.path.join(output_dir, 'phase2_image_retrieval', 'image_retrieval_results.jsonl')
        )
        caption_retrieval_path = self.config.get(
            'phase2_caption_retrieval_path',
            os.path.join('data/result/infoseek_v4', 'phase2_image_caption_retrieval', 'final_retrieval_results.jsonl')
        )
        image_retrieval_map = _load_retrieval_map(image_retrieval_path)
        caption_retrieval_map = _load_retrieval_map(caption_retrieval_path)

        data_items = list(dataset.data)
        entity_docs = []
        for item in data_items:
            docs = reranked_results.get(item.data_id, [])
            top_docs = docs if isinstance(docs, list) and len(docs) > 0 else ""
            entity_docs.append(
                self._truncate_text_for_prompt(self._extract_reference_text(top_docs), naive_ref_max_chars)
            )

        base_prompts = [
            self.naive_prompt_template.get_string_sep(item, reference_doc)
            for item, reference_doc in zip(data_items, entity_docs)
        ]
        dataset.update_output("prompt", base_prompts)

        pred_answer_list = [None] * len(data_items)
        normalized_nll = [None] * len(data_items)

        # reranked_doc 为空的样本跳过不确定度计算，仅对非空样本计算 normalized_nll
        non_empty_indices = [
            idx for idx, doc in enumerate(entity_docs)
            if isinstance(doc, str) and doc.strip() != ""
        ]
        if non_empty_indices:
            non_empty_prompts = [base_prompts[idx] for idx in non_empty_indices]
            non_empty_preds, token_probs = self.generator.generate(non_empty_prompts, return_scores=True)
            non_empty_nll = self._compute_normalized_nll(token_probs)
            for local_idx, global_idx in enumerate(non_empty_indices):
                pred_answer_list[global_idx] = non_empty_preds[local_idx]
                normalized_nll[global_idx] = non_empty_nll[local_idx]
        
        # RAG生成（门控触发）
        retrieval_results_all = dataset.retrieval_results if hasattr(dataset, "retrieval_results") else [None] * len(data_items)

        items_need_rag = []
        for item, nll, reranked_doc, retrieval_results in zip(data_items, normalized_nll, entity_docs, retrieval_results_all):
            # 如果 reranked_doc 为空：不计算不确定度，直接生成 Query 并检索
            if not (isinstance(reranked_doc, str) and reranked_doc.strip()):
                query_prompt = self.query_generate_prompt_template.get_string_sep(item, "")
                generated_query = self.generator.generate([query_prompt])[0]
                try:
                    direct_retrieval = self.retriever.batch_search([generated_query])[0]
                except Exception:
                    direct_retrieval = self.retriever.search(generated_query)
                items_need_rag.append(
                    {
                        "item": item,
                        "nll": None,
                        "reranked_doc": reranked_doc,
                        "retrieval_results": direct_retrieval if direct_retrieval is not None else [],
                    }
                )
                continue

            if nll is not None and nll > retrieval_threshold:
                items_need_rag.append(
                    {
                        "item": item,
                        "nll": nll,
                        "reranked_doc": reranked_doc,
                        "retrieval_results": retrieval_results if retrieval_results is not None else [],
                    }
                )

        rag_prompts = []
        rag_data_ids = []
        for need_item in items_need_rag:
            top1_results = need_item["retrieval_results"][:1]
            retrieved_docs = "\n\n".join(
                [
                    f"Doc{i+1}:\n{self._truncate_text_for_prompt(self._extract_reference_text(doc), rag_doc_max_chars)}"
                    for i, doc in enumerate(top1_results)
                ]
            )
            rag_prompts.append(
                self.rag_prompt_template.get_string_sep(
                    need_item["item"],
                    f"Description of the Image:\n{need_item['reranked_doc']}\n\nReferences for answering the Question:\n{retrieved_docs}",
                )
            )
            rag_data_ids.append(need_item["item"].data_id)

        rag_generated_answers = self.generator.generate(rag_prompts) if rag_prompts else []
        rag_answer_map = {data_id: answer for data_id, answer in zip(rag_data_ids, rag_generated_answers)}
        rag_prompt_map = {data_id: prompt for data_id, prompt in zip(rag_data_ids, rag_prompts)}

        id_to_index = {item.data_id: idx for idx, item in enumerate(data_items)}
        for need_item, rag_answer in zip(items_need_rag, rag_generated_answers):
            target_idx = id_to_index.get(need_item["item"].data_id)
            if target_idx is not None:
                pred_answer_list[target_idx] = rag_answer

        dataset.update_output("pred", pred_answer_list)

        phase5_rows = []
        for idx, item in enumerate(data_items):
            data_id = item.data_id
            image_docs = image_retrieval_map.get(data_id, [])
            merged_docs = caption_retrieval_map.get(data_id, [])
            image_doc_set = set(image_docs) if isinstance(image_docs, list) else set()
            caption_only_docs = [doc for doc in merged_docs if doc not in image_doc_set] if isinstance(merged_docs, list) else []

            reranked_docs = reranked_results.get(data_id, [])
            reranked_order = []
            if isinstance(reranked_docs, list):
                for rank_i, doc in enumerate(reranked_docs):
                    reranked_order.append({
                        'rank': rank_i + 1,
                        'title': doc.get('title') if isinstance(doc, dict) else self._extract_doc_title(self._extract_reference_text(doc)),
                        'score': doc.get('score') if isinstance(doc, dict) else None,
                        'text_preview_200': self._truncate_text_for_prompt(self._extract_reference_text(doc), 200),
                    })

            query_retrieval_docs = retrieval_results_all[idx] if idx < len(retrieval_results_all) else []
            phase5_rows.append({
                'data_id': data_id,
                'image_id': getattr(item, 'image_id', None),
                'question': getattr(item, 'question', None),
                'normalized_nll': normalized_nll[idx],
                'trigger_rag': data_id in rag_answer_map,
                'rag_answer': rag_answer_map.get(data_id),
                'final_pred': pred_answer_list[idx],
                'image_retrieval_docs_preview_200': _preview_docs_200(image_docs),
                'caption_retrieval_docs_preview_200': _preview_docs_200(caption_only_docs),
                'reranked_doc_order': reranked_order,
                'query_retrieval_docs_preview_200': _preview_docs_200(query_retrieval_docs),
            })
        self._save_phase_jsonl('phase5_rag_generation', phase5_rows, file_name='rag_generation_case_study.jsonl')

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_func=pred_process_func)
        return dataset