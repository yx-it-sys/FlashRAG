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
import inspect


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

    def _normalize_generation_stats(self, stats, sample_count):
        if not stats or sample_count <= 0:
            return None

        if isinstance(stats, dict):
            summary = dict(stats)
        elif isinstance(stats, list):
            total_samples = 0
            total_input_tokens = 0.0
            total_output_tokens = 0.0
            total_tokens = 0.0
            total_latency = 0.0

            for item in stats:
                if not isinstance(item, dict):
                    continue
                batch_size = int(item.get("batch_size", item.get("total_samples", 1)) or 1)
                total_samples += batch_size
                total_input_tokens += float(item.get("input_tokens", 0.0))
                total_output_tokens += float(item.get("output_tokens", 0.0))
                total_tokens += float(item.get("total_tokens", total_input_tokens + total_output_tokens))
                total_latency += float(item.get("latency_seconds", item.get("total_latency_seconds", 0.0)))

            if total_samples <= 0:
                total_samples = len(stats)
            summary = {
                "total_samples": int(total_samples),
                "total_input_tokens": int(total_input_tokens),
                "total_output_tokens": int(total_output_tokens),
                "total_tokens": int(total_tokens),
                "total_latency_seconds": float(total_latency),
            }
        else:
            return None

        total_samples = int(summary.get("total_samples", sample_count) or sample_count)
        total_input_tokens = float(summary.get("total_input_tokens", summary.get("input_tokens", 0.0)))
        total_output_tokens = float(summary.get("total_output_tokens", summary.get("output_tokens", 0.0)))
        total_tokens = float(summary.get("total_tokens", total_input_tokens + total_output_tokens))
        total_latency = float(summary.get("total_latency_seconds", summary.get("latency_seconds", 0.0)))

        compact_summary = {
            "total_samples": total_samples,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            "latency_seconds": total_latency,
            "avg_input_tokens_per_sample": (total_input_tokens / total_samples) if total_samples > 0 else 0.0,
            "avg_output_tokens_per_sample": (total_output_tokens / total_samples) if total_samples > 0 else 0.0,
            "avg_total_tokens_per_sample": (total_tokens / total_samples) if total_samples > 0 else 0.0,
            "avg_latency_seconds_per_sample": (total_latency / total_samples) if total_samples > 0 else 0.0,
            "input_tokens_per_second": (total_input_tokens / total_latency) if total_latency > 0 else 0.0,
            "output_tokens_per_second": (total_output_tokens / total_latency) if total_latency > 0 else 0.0,
            "total_tokens_per_second": (total_tokens / total_latency) if total_latency > 0 else 0.0,
        }
        return [compact_summary for _ in range(sample_count)]

    def _attach_generation_stats(self, dataset):
        generator = getattr(self, "generator", None)
        if generator is None:
            return

        stats = None
        for method_name in ("get_generation_stats", "get_performance_stats", "cost_stats"):
            if not hasattr(generator, method_name):
                continue
            method = getattr(generator, method_name)
            try:
                stats = method(reset=True)
            except TypeError:
                stats = method()
            except Exception:
                stats = None
            if stats:
                break

        normalized_stats = self._normalize_generation_stats(stats, len(dataset))
        if normalized_stats is not None:
            dataset.update_output("generation_stats", normalized_stats)

    def parse_response(self, response):
        return response
    
    def _truncate_after_search(self, response):
        if not response or "<Search>" not in response:
            return response

        search_start = response.find("<Search>")
        search_end = response.find("</Search>", search_start)
        if search_end != -1:
            search_end += len("</Search>")
            tail = response[search_end:]
            next_markers = [
                "<Thought>",
                "</Thought>",
                "<Sub-Question>",
                "</Sub-Question>",
                "<Search>",
                "<End>",
                "<Final Answer>",
                "Final Answer:",
            ]

            next_positions = []
            for marker in next_markers:
                pos = tail.find(marker)
                if pos != -1:
                    next_positions.append(pos)

            if not next_positions:
                return response[:search_end].rstrip()

            cut_pos = search_end + min(next_positions)
            return response[:cut_pos].rstrip()

        search_body_start = search_start + len("<Search>")
        tail = response[search_body_start:]
        next_markers = [
            "<Thought>",
            "</Thought>",
            "<Sub-Question>",
            "</Sub-Question>",
            "<Search>",
            "<End>",
            "<Final Answer>",
            "Final Answer:",
        ]

        next_positions = []
        for marker in next_markers:
            pos = tail.find(marker)
            if pos != -1:
                next_positions.append(pos)

        if not next_positions:
            return response

        cut_pos = search_body_start + min(next_positions)
        return response[:cut_pos].rstrip()
    
    def _source_matches_target(self, item_source):
        if self.target_source is None:
            return True
        if item_source == self.target_source:
            return True
        if self.target_source == "infoseek" and item_source == "oven":
            return True
        return False
    
    def _clip_text(self, text, limit):
        if text is None:
            return None
        text = str(text)
        if len(text) <= limit:
            return text
        clipped = text[:limit]
        return clipped + "\n\n[Truncated due to prompt length limit]"

    def _format_retrieval_content(self, retrieved_docs, preferred_field=None):
        if not retrieved_docs:
            return ""
        if preferred_field is None:
            return self._format_image_retrieval_content(retrieved_docs)
        if not isinstance(retrieved_docs, list):
            retrieved_docs = [retrieved_docs]
        per_doc_limit = max(0, self.retrieved_doc_char_limit)
        return "\n\n".join(
            [
                f"Doc{i+1}:\n{self._clip_text(self._format_retrieved_doc(doc, preferred_field=preferred_field), per_doc_limit)}"
                for i, doc in enumerate(retrieved_docs)
            ]
        )

    def _extract_retrieval_query(self, response, retrieval_label):
        pattern = rf'{re.escape(retrieval_label)}[:\s"]*(.*?)(?=<|$)'
        match = re.search(pattern, response, re.DOTALL)
        if not match:
            return ""
        return match.group(1).strip()

    def _extract_search_body(self, response):
        if not response:
            return ""

        match = re.search(r"<Search>\s*(.*?)\s*</Search>", response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        search_start = response.find("<Search>")
        if search_start == -1:
            return ""

        tail = response[search_start + len("<Search>") :]
        next_markers = [
            "<Thought>",
            "</Thought>",
            "<Sub-Question>",
            "</Sub-Question>",
            "<Search>",
            "<End>",
            "<Final Answer>",
            "Final Answer:",
        ]
        next_positions = [pos for marker in next_markers if (pos := tail.find(marker)) != -1]
        if not next_positions:
            return tail.strip()
        return tail[: min(next_positions)].strip()
    
    def _extract_action_nodes(self, response):
        if not response:
            return []

        nodes = []
        tag_pattern = re.compile(r"<Thought>|<Sub-Question>|<Search>|<End>")
        matches = list(tag_pattern.finditer(response))
        action_name_map = {
            "<Thought>": "thought",
            "<Sub-Question>": "sub-question",
            "<Search>": "search",
        }

        for idx, match in enumerate(matches):
            tag = match.group(0)
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(response)
            content = response[start:end].strip()

            if tag in action_name_map and content:
                nodes.append({
                    "action": action_name_map[tag],
                    "content": content,
                })
            elif tag == "<End>":
                final_answer_match = re.search(r"Final Answer:\s*(.*?)(?=$)", content, re.DOTALL)
                if final_answer_match:
                    final_answer = final_answer_match.group(1).strip().replace("\n", "")
                    if final_answer:
                        nodes.append({
                            "action": "final_answer",
                            "content": final_answer,
                        })

        final_answer_match = re.search(r"(?:<Final Answer>|Final Answer:)\s*(.*?)(?=<|$)", response, re.DOTALL)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip().replace("\n", "")
            if final_answer and not any(node.get("action") == "final_answer" for node in nodes):
                nodes.append({
                    "action": "final_answer",
                    "content": final_answer,
                })
        return nodes
    
    def _search_with_main_retriever(self, query, query_kind):
            search_callable = getattr(self.retriever, "_search", self.retriever.search)
            search_signature = inspect.signature(search_callable)
            search_params = search_signature.parameters
            supports_kwargs = any(
                param.kind == inspect.Parameter.VAR_KEYWORD for param in search_params.values()
            )
            kwargs = {}

            if query_kind == "image":
                topk = self.config["image_retrieval_topk"]
            else:
                topk = self.config["text_retrieval_topk"]

            if topk is not None and ("num" in search_params or supports_kwargs):
                kwargs["num"] = topk
            if "query_type" in search_params or supports_kwargs:
                kwargs["query_type"] = query_kind
            if query_kind == "image" and ("target_modal" in search_params or supports_kwargs):
                kwargs["target_modal"] = self.config["image_retrieval_target_modal"]

            return self.retriever.search(query, **kwargs)    
        
    def _generate_text(self, messages):
        delay = 2
        max_retries = 5
        for attempt in range(max_retries):
            try:
                raw_response = self.generator.generate([messages])
                return self._normalize_generation_output(raw_response)
            except Exception as e:
                if not self._is_retryable_api_error(e):
                    raise
                if attempt == max_retries - 1:
                    raise
                print(
                    f"Transient API error ({e.__class__.__name__}), retrying in {delay}s ..."
                )
                time.sleep(delay)
                delay *= 2
        raw_response = self.generator.generate([messages])
        return self._normalize_generation_output(raw_response)

    def _log_retrieval_preview(self, retrieval_content):
        if retrieval_content is None:
            print("Retrieval result: None")
            return
        preview_limit = min(self.retrieval_char_limit, 500)
        preview = str(retrieval_content)
        if len(preview) > preview_limit:
            preview = preview[:preview_limit] + "\n\n[Preview truncated in log]"
        print(f"Retrieval result:\n{preview}")

    def _record_response_actions(self, trajectory, response):
        trajectory.extend(self._extract_action_nodes(response))

    def _search_text_docs(self, query_txt):
        return self._search_with_main_retriever(query_txt, query_kind="text")

    def _record_retrieval_result(self, trajectory, retrieval_mode, query_txt, retrieval_content):
        mode_name_map = {
            "text_retrieval": "text_retrieval_result",
            "image_retrieval": "image_retrieval_result",
            "no_retrieval": "no_retrieval_result",
        }
        trajectory.append({
            "action": mode_name_map.get(retrieval_mode, "retrieval_result"),
            "mode": retrieval_mode,
            "query": query_txt if query_txt else None,
            "content": retrieval_content,
        })

    def _write_trajectory(self, record):
        self.safe_write(self.trajectory_path, self._serialize_for_log(record))

    def evaluate(self, dataset, do_eval=True, pred_process_func=None):
        """The evaluation process after finishing overall generation"""

        if pred_process_func is not None:
            dataset = pred_process_func(dataset)

        self._attach_generation_stats(dataset)

        if do_eval:
            # evaluate & save result
            eval_result = self.evaluator.evaluate(dataset)
            print(eval_result)

        return dataset


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
            self.visual_clue_prompt_template.get_string_no_retrieval(item) for item in dataset
        ]
        print("Generating clues for dataset...")   
        raw_clue_list = self.generator.generate(input_prompts)
        print("Raw clues generated. Parsing clues...")
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
    
    def caption_retrieval(self, dataset, image_phase_dir, caption_phase_dir):
        if self.retriever is None:
            raise ValueError("Retriever is not provided for caption retrieval.")
        if self.generator is None:
            raise ValueError("Generator is not provided for caption retrieval.")

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
            def _text_dedup_key(text):
                # Dedup by actual document text instead of corpus id.
                return str(text or '').strip()

            for doc in existing_docs_dict:
                doc_id = doc.get('id')
                doc_text = doc.get('text', '')

                dedup_key = _text_dedup_key(doc_text)
                if dedup_key in existing_set:
                    continue
                existing_set.add(dedup_key)
                normalized_docs.append({'id': doc_id, 'text': doc_text})

            for doc in retrieval_texts:
                doc_id = doc.get('id')
                doc_text = doc.get('text', '')
                dedup_key = _text_dedup_key(doc_text)
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

    def quadric_reranking(self, clue_path, retrieval_results_path, total_budget=1000):
        stats = {}  # 用于存储性能数据
        overall_start = time.time()

        # 1. 数据读取阶段
        io_start = time.time()
        with open(clue_path, 'r', encoding='utf-8') as f:
            clue_data = [json.loads(line) for line in f]
        with open(retrieval_results_path, 'r', encoding='utf-8') as f:
            retrieval_data = {}
            for line in f:
                row = json.loads(line)
                retrieval_data[row['data_id']] = row

        stats['io_read_time'] = time.time() - io_start

        reranked_results = []
        vote_logs = []
        processing_times = []
        total_comparisons = 0

        # 2. 核心重排序阶段 (无聚类、无剪枝)
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
            num_docs = len(retrieval_results)
            num_clues = len(clues)
            total_comparisons += num_docs * num_clues

            # 初始化评分矩阵: num_docs x num_clues
            score_matrix = np.zeros((num_clues, num_docs), dtype=np.float32)

            # 矩阵计算: 批量计算所有线索对文档的NLI评分
            for clue_idx, clue in enumerate(clues):
                clue_text = clue.get('clue', '')
                for doc_idx, doc in enumerate(retrieval_results):
                    logit = self.llm_nli_judge(clue_text, doc['text'])
                    score = 1 / (1 + np.exp(-logit))
                    score_matrix[clue_idx, doc_idx] = score

            # NEW: Entropy-QV approach for budget weight allocation
            # ORIGINAL (annotated):
            # importances = [float(c.get('importance', 1.0)) for c in clues]
            # total_importance = sum(importances) + 1e-8
            # clue_budget_weights = np.array([
            #     np.sqrt(max(total_budget * (imp / total_importance), 0.0)) for imp in importances
            # ], dtype=np.float32)
            importances = [float(c.get('importance', 1.0)) for c in clues]

            # 1. Normalize each clue's scores into probability distribution
            # P_{i,j} = S_{i,j} / sum(S_{i,m})
            row_sums = np.sum(score_matrix, axis=1, keepdims=True) + 1e-8
            probability_distributions = score_matrix / row_sums  # (num_clues, num_docs)

            # 2. Calculate Shannon entropy for each clue
            # H(c_i) = -sum(P_{i,j} * log(P_{i,j}))
            entropy_values = -np.sum(probability_distributions * np.log(probability_distributions + 1e-8), axis=1)

            # 3. Calculate discriminative weight
            # w_i^{dist} = 1 - H(c_i) / log(k)
            max_entropy = np.log(num_docs)  # log(k) where k = num_docs
            discriminative_weights = 1 - (entropy_values / max_entropy)

            # 4. Calculate revised budget allocation
            # \hat{I}_i = (r_i * w_i^{dist}) / sum(r_m * w_m^{dist}) * I_{total}
            # weighted_importances = np.array(importances) * discriminative_weights
            weighted_importances = discriminative_weights
            total_weighted_importance = np.sum(weighted_importances) + 1e-8
            clue_budget_weights = np.array([
                np.sqrt(max(total_budget * (wi / total_weighted_importance), 0.0))
                for wi in weighted_importances
            ], dtype=np.float32)

            # 矩阵计算: 标准化并加权
            # 对每个线索进行标准化: (score * weight) / sqrt(sum(score^2))
            row_sq_scores = np.square(score_matrix)  # (num_clues, num_docs)
            row_norms = np.sqrt(np.sum(row_sq_scores, axis=1, keepdims=True)) + 1e-8  # (num_clues, 1)
            weighted_scores = (score_matrix * clue_budget_weights[:, np.newaxis]) / row_norms  # (num_clues, num_docs)

            # 累加所有线索的得分
            doc_scores = np.sum(weighted_scores, axis=0)  # (num_docs,)

            # 获取线索文本列表
            clue_texts = [c.get('clue', '') for c in clues]

            # 按得分排序文档
            ranked_indices = sorted(
                range(num_docs),
                key=lambda idx: (float(doc_scores[idx]), -idx),
                reverse=True,
            )

            # 创建重排序结果
            reranked_flat_results = []
            for rank_idx, doc_idx in enumerate(ranked_indices, 1):
                doc = retrieval_results[doc_idx]
                score = float(doc_scores[doc_idx])
                reranked_flat_results.append({
                    'text': doc,
                    'title': self._extract_doc_title(doc),
                    'score': score,
                })

                # 创建投票日志 (与 naive_reranking 格式一致)
                vote_logs.append({
                    'data_id': data_id,
                    'image_id': item.get('image_id'),
                    'text_query': item.get('question'),
                    'clue': clue_texts,
                    'doc': doc,
                    'vote_count': score,  # 使用得分作为投票数
                    'rank': rank_idx,
                    'clue_scores': [float(s) for s in score_matrix[:, doc_idx]],  # 每个线索的原始评分
                })

            reranked_results.append({
                'data_id': data_id,
                'image_id': item.get('image_id'),
                'reranked_results': reranked_flat_results,
            })

            processing_times.append(time.time() - item_start)
            # --- 计算逻辑结束 ---

        # 3. 统计数据
        stats['avg_item_processing_time'] = np.mean(processing_times) if processing_times else 0
        stats['total_processing_time'] = sum(processing_times)
        stats['num_items_processed'] = len(reranked_results)
        stats['total_comparisons'] = int(total_comparisons)

        # 4. 保存结果
        save_start = time.time()
        self._save_phase_jsonl('phase3_quadric_reranking', reranked_results, file_name='reranked_results.jsonl')
        self._save_phase_jsonl('phase3_quadric_reranking_vote_logs', vote_logs, file_name='naive_vote_logs.jsonl')

        stats['io_write_time'] = time.time() - save_start
        stats['overall_total_time'] = time.time() - overall_start

        # 5. 保存性能统计
        stats_path = self._save_phase_jsonl('phase3_quadric_reranking_stats', stats, file_name='performance_stats.jsonl', append=True)

        print(f"Quadric reranking complete. Stats saved to {stats_path}")
        return reranked_results
        
    def my_reranking_v1(self, clue_path, retrieval_results_path):
        # Naive reranking: for each clue, ask the LLM to vote Yes/No for each retrieved doc,
        # then rank docs by accumulated votes, and cluster by title.
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

        for item in tqdm(clue_data, desc="Processing clues"):
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
                    output = self.llm_nli_judge_binary(clue_text, doc['text'])
                    vote = 1 if output == 'Yes' else -0.1
                    clue_votes.append(vote)

                if len(clue_votes) != len(retrieval_results):
                    raise RuntimeError(
                        f"Naive clue vote size mismatch: got {len(clue_votes)}, expected {len(retrieval_results)}"
                    )
                doc_votes += np.asarray(clue_votes, dtype=np.float32)
                total_binary_judges += len(clue_votes)

            # Cluster documents by title and rank clusters by accumulated votes.
            title_clusters = self._cluster_documents_by_title(retrieval_results, doc_votes)
            reranked_flat_results = self._create_reranked_results_from_clusters(title_clusters)
            reranked_flat_results = self._apply_listwise_tie_breaks(text_query, image_id, reranked_flat_results)

            for rank_idx, doc_info in enumerate(reranked_flat_results):
                doc_text = doc_info['text']
                vote_count = float(doc_info['score'])
                reranked_flat_results[rank_idx] = {
                    'text': doc_text,
                    'title': str(doc_info.get('title', '')),
                    'score': vote_count,
                }
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

    def my_reranking_v0(self, clue_path, retrieval_results_path):
        # Naive reranking: for each clue, ask the LLM to vote Yes/No for each retrieved doc,
        # then rank docs by accumulated votes, and cluster by title.
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

        for item in tqdm(clue_data, desc="Processing clues"):
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
                    output = self.llm_nli_judge_binary(clue_text, doc['text'])
                    vote = 1 if output == 'Yes' else 0.0
                    clue_votes.append(vote)

                if len(clue_votes) != len(retrieval_results):
                    raise RuntimeError(
                        f"Naive clue vote size mismatch: got {len(clue_votes)}, expected {len(retrieval_results)}"
                    )
                doc_votes += np.asarray(clue_votes, dtype=np.float32)
                total_binary_judges += len(clue_votes)

            # Cluster documents by title and rank clusters by accumulated votes.
            title_clusters = self._cluster_documents_by_title(retrieval_results, doc_votes)
            reranked_flat_results = self._create_reranked_results_from_clusters(title_clusters)

            for rank_idx, doc_info in enumerate(reranked_flat_results):
                doc_text = doc_info['text']
                vote_count = float(doc_info['score'])
                reranked_flat_results[rank_idx] = {
                    'text': doc_text,
                    'title': str(doc_info.get('title', '')),
                    'score': vote_count,
                }
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

    def naive_reranking_wo_cluster(self, clue_path, retrieval_results_path):
        # Naive reranking without clustering: score each document independently.
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

        for item in tqdm(clue_data, desc="Processing clues"):
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

            for clue_text in clue_texts:
                clue_votes = []
                for doc in retrieval_results:
                    output = self.llm_nli_judge_binary(clue_text, doc['text'])
                    vote = 1 if output == 'Yes' else 0
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
                doc_text = retrieval_results[doc_idx]['text']
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
        print(f"Naive reranking (wo cluster) complete. Stats saved to {stats_path}")
        return reranked_results

    def _cluster_documents_by_title(self, retrieval_results, doc_votes):
        """Cluster documents by title and sum voting counts for documents with the same title."""
        title_clusters = {}

        for doc_idx, doc in enumerate(retrieval_results):
            title = self._extract_doc_title(doc['text'])
            if title not in title_clusters:
                title_clusters[title] = {
                    'title': title,
                    'docs': [],
                    'total_votes': 0.0
                }

            title_clusters[title]['docs'].append({
                'text': doc['text'],
                'vote': doc_votes[doc_idx]
            })
            title_clusters[title]['total_votes'] += doc_votes[doc_idx]

        return list(title_clusters.values())

    def _create_reranked_results_from_clusters(self, title_clusters):
        """Create final reranked results by sorting clusters and splicing their documents."""
        # Sort clusters by total votes (descending)
        sorted_clusters = sorted(
            title_clusters,
            key=lambda cluster: cluster['total_votes'],
            reverse=True
        )

        reranked_results = []
        for cluster in sorted_clusters:
            # Sort documents within each cluster by their individual votes (descending)
            sorted_docs = sorted(
                cluster['docs'],
                key=lambda doc_info: doc_info['vote'],
                reverse=True
            )

            title = str(cluster.get('title', '')).strip()

            def _strip_same_title_prefix(doc_text, cluster_title):
                text = str(doc_text or '').strip()
                if not text:
                    return ''
                if not cluster_title:
                    return text

                escaped_title = re.escape(cluster_title)
                # Remove leading "<title> - " (allow flexible spaces around '-').
                pattern = rf'^\s*{escaped_title}\s*-\s*'
                return re.sub(pattern, '', text, count=1, flags=re.IGNORECASE).strip()

            cleaned_parts = []
            for doc_info in sorted_docs:
                body = _strip_same_title_prefix(doc_info.get('text', ''), title)
                if body:
                    cleaned_parts.append(body)

            merged_body = "\n\n".join(cleaned_parts)
            merged_text = f"{title} - {merged_body}" if merged_body else f"{title}"

            # Keep one row per cluster; score is the cluster total vote sum.
            reranked_results.append({
                'text': merged_text,
                'title': title,
                'score': float(cluster.get('total_votes', 0.0))
            })

        return reranked_results

    def _get_listwise_cfg(self):
        cfg = self.config if isinstance(self.config, dict) else {}
        window_size = int(cfg.get("listwise_window_size", 4))
        step_size = int(cfg.get("listwise_step_size", 2))
        num_repeat = int(cfg.get("listwise_num_repeat", 5))
        max_doc_words = int(cfg.get("listwise_doc_max_words", 200))

        if window_size <= 1:
            window_size = 2
        if step_size <= 0:
            step_size = 1
        if num_repeat <= 0:
            num_repeat = 1

        return {
            'window_size': window_size,
            'step_size': step_size,
            'num_repeat': num_repeat,
            'max_doc_words': max_doc_words,
        }

    def _clean_listwise_response_to_indices(self, response_text):
        cleaned = []
        for ch in str(response_text or ''):
            cleaned.append(ch if ch.isdigit() else ' ')
        tokens = "".join(cleaned).split()
        seen = set()
        indices = []
        for tok in tokens:
            try:
                idx = int(tok) - 1
            except Exception:
                continue
            if idx not in seen:
                seen.add(idx)
                indices.append(idx)
        return indices

    def _receive_listwise_permutation(self, local_ranking, permutation_text, rank_start, rank_end):
        indices = self._clean_listwise_response_to_indices(permutation_text)
        cut_range = list(local_ranking[rank_start:rank_end])
        original = list(range(len(cut_range)))
        indices = [i for i in indices if i in original]
        indices.extend([i for i in original if i not in indices])
        for pos, src_i in enumerate(indices):
            local_ranking[rank_start + pos] = cut_range[src_i]
        return local_ranking

    def _build_listwise_prompt(self, query, image_id, docs, max_doc_words):
        image_query_path = os.path.join(self.config['dataset_path'], 'images', f'{image_id}.jpg')
        question_image = Image.open(image_query_path).convert('RGB')

        lines = [
            "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query.",
            "",
            f"I will provide {len(docs)} passages with identifiers [1] ...[{len(docs)}].",
            f"Query: {query}",
            "",
        ]
        for i, doc in enumerate(docs, 1):
            doc_text = str(doc.get('text', '') if isinstance(doc, dict) else doc)
            if max_doc_words > 0:
                doc_text = ' '.join(doc_text.split()[:max_doc_words])
            lines.append(f"[{i}] {doc_text}")
        lines.extend([
            "",
            "Rank the passages by relevance to the query in descending order.",
            "Output format strictly: [x] > [y] > ... > [N]",
            "Only output identifiers. Do not explain.",
        ])

        return [{
            "role": "user",
            "content": [
                {'type': 'image', 'image': question_image},
                {'type': 'text', 'text': "\n".join(lines)},
            ],
        }]

    def _listwise_compare(self, query, image_id, docs, max_doc_words):
        if self.generator is None:
            raise ValueError("Generator is required for listwise reranking.")

        self.total_compare = getattr(self, 'total_compare', 0) + 1
        prompt = self._build_listwise_prompt(query, image_id, docs, max_doc_words)
        tokenizer = getattr(self, 'tokenizer', None)
        if tokenizer is None:
            tokenizer = getattr(self.generator, 'tokenizer', None)

        if tokenizer is not None:
            try:
                self.total_prompt_tokens = getattr(self, 'total_prompt_tokens', 0) + len(
                    tokenizer.encode(str(prompt), add_special_tokens=False)
                )
            except Exception:
                pass

        outputs = self.generator.generate([prompt])
        if not outputs:
            return ""

        output_text = outputs[0] if isinstance(outputs[0], str) else str(outputs[0])
        if tokenizer is not None:
            try:
                self.total_completion_tokens = getattr(self, 'total_completion_tokens', 0) + len(
                    tokenizer.encode(output_text, add_special_tokens=False)
                )
            except Exception:
                pass
        return output_text

    def _apply_listwise_reranking(self, query, image_id, ranking):
        ranking = list(ranking)
        if len(ranking) <= 1:
            return ranking

        cfg = self._get_listwise_cfg()
        for _ in range(cfg['num_repeat']):
            end_pos = len(ranking)
            while end_pos > 0:
                start_pos = max(end_pos - cfg['window_size'], 0)
                if end_pos - start_pos <= 1:
                    break
                window_docs = ranking[start_pos:end_pos]
                permutation = self._listwise_compare(query, image_id, window_docs, cfg['max_doc_words'])
                ranking = self._receive_listwise_permutation(ranking, permutation, start_pos, end_pos)
                if start_pos == 0:
                    break
                end_pos = max(0, end_pos - cfg['step_size'])
        return ranking

    def _apply_listwise_tie_breaks(self, query, image_id, ranking):
        ranking = list(ranking)
        if len(ranking) <= 1:
            return ranking

        idx = 0
        while idx < len(ranking):
            current_score = float(ranking[idx].get('score', 0.0))
            end_idx = idx + 1
            while end_idx < len(ranking) and float(ranking[end_idx].get('score', 0.0)) == current_score:
                end_idx += 1

            if end_idx - idx > 1:
                tie_block = [{
                    'text': doc.get('text', ''),
                    'title': doc.get('title', self._extract_doc_title(doc.get('text', ''))),
                    'score': current_score,
                } for doc in ranking[idx:end_idx]]
                reranked_block = self._apply_listwise_reranking(query, image_id, tie_block)
                for offset, doc in enumerate(reranked_block):
                    doc['score'] = current_score
                    ranking[idx + offset] = doc

            idx = end_idx

        return ranking

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

    def llm_nli_judge_binary(self, clue, doc):
        """Return 'Yes' or 'No' based on the final LLM output instead of logits."""
        prompt = f"Clue: {clue}\nDocument: {doc}\nQuestion: Does the document support the clue? Answer 'Yes' or 'No'."
#         prompt = """
# Role: You are a rigorous fact-checker for aircraft identification.

# Task: Evaluate if the provided Document contains explicit evidence that directly supports the Clue.

# Instructions:

# First, search the Document for any specific identifiers, such as registration codes (e.g., D-XXXX), model names, or serial numbers.

# Compare the identified information in the Document with the Clue.

# If the Clue contains a specific code like a registration (e.g., D-BFDO), you must output 'Yes' ONLY if the Document mentions that exact code or its verified synonymous entity.

# If there is a character mismatch in identifiers or the Document only describes a similar type of aircraft without the specific clue's detail, you MUST output 'No'.

# Do not infer or be "helpful." Be strict.

# Input Data:
# Clue: {clue}
# Document: {doc}

# Response Format:
# Directly output 'Yes' or 'No' as the final judgment.
# """.format(clue=clue, doc=doc)
        if self.generator is None:
            raise ValueError("Generator is required for llm_nli_judge_binary.")

        try:
            raw_outputs = self.generator.generate(
                [prompt],
                return_raw_output=False,
                max_tokens=5,
                temperature=0,
            )
        except Exception as e:
            print(f"Error in llm_nli_judge_binary: {e}")
            return "No"

        if not raw_outputs or len(raw_outputs) == 0:
            return "No"

        # Get the generated text output
        output_text = raw_outputs[0].strip() if isinstance(raw_outputs[0], str) else str(raw_outputs[0]).strip()

        # Check if output contains 'Yes' or 'No'
        if "yes" in output_text.lower():
            final_output = "Yes"
        elif "no" in output_text.lower():
            final_output = "No"
        else:
            # Default to 'No' if unclear response
            final_output = "No"

        # print(f"LLM NLI Judge Binary - Clue: {clue}, Doc: {doc[:100]}, Output: {final_output}")
        return final_output


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
            os.path.join('data/result/infoseek', 'phase2_image_caption_retrieval', 'final_retrieval_results.jsonl')
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

class BaselineMMPipeline(BasicMultiModalPipeline):
    def __init__(self, config, generator, prompt_template):
        super().__init__(config)
        self.prompt_template = prompt_template
        self.generator = generator
        self.tokenizer = getattr(self.generator, "tokenizer", None)
        self.batch_size = int(config.get("batch_size", 8)) if isinstance(config, dict) else 8
        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0

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

    def _llm_yes_prob(self, item, passage_text):
        if self.generator is None:
            raise ValueError("Generator is required for baseline reranking.")
        if self.tokenizer is None:
            raise ValueError("Generator does not expose tokenizer; cannot compute Yes/No logits.")
        
        pointwise_bin_prompt_template = self.prompt_template

        prompt = pointwise_bin_prompt_template.get_string_sep(item=item, reference=passage_text)

        # Guard against vLLM decoder prompt overflow.
        max_model_len = int(self.config.get("max_model_len", 0)) if isinstance(self.config, dict) else 0
        if max_model_len > 0 and isinstance(prompt, str):
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            if len(prompt_ids) > max_model_len:
                # Keep the tail so query/instruction part is preserved.
                prompt = self.tokenizer.decode(
                    prompt_ids[-max_model_len:],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )

        self.total_compare += 1

        yes_candidates = []
        no_candidates = []
        for token_text in ["Yes", " Yes"]:
            encoded = self.tokenizer.encode(token_text, add_special_tokens=False)
            if len(encoded) == 1:
                yes_candidates.append(encoded[0])
        for token_text in ["No", " No"]:
            encoded = self.tokenizer.encode(token_text, add_special_tokens=False)
            if len(encoded) == 1:
                no_candidates.append(encoded[0])

        if not yes_candidates:
            encoded_yes = self.tokenizer.encode("Yes", add_special_tokens=False)
            if encoded_yes:
                yes_candidates = [encoded_yes[0]]
        if not no_candidates:
            encoded_no = self.tokenizer.encode("No", add_special_tokens=False)
            if encoded_no:
                no_candidates = [encoded_no[0]]

        max_retries = int(self.config.get("pointwise_logprob_retries", 3)) if isinstance(self.config, dict) else 3
        data_id = getattr(item, "data_id", None) if not isinstance(item, dict) else item.get("data_id")
        token_logprobs = None

        try:
            raw_outputs = self.generator.generate(
                [prompt],
                return_raw_output=True,
                max_tokens=1,
                temperature=0,
                logprobs=20,
            )
        except Exception as e:
            raise RuntimeError(
                f"Pointwise reranking generate() failed (data_id={data_id}): {e}"
            ) from e

        if not raw_outputs:
            raise RuntimeError(f"generator returned empty raw outputs (data_id={data_id}).")

        raw_output = raw_outputs[0]
        first_candidates = getattr(raw_output, "outputs", None)
        if not first_candidates:
            raise RuntimeError(f"missing candidate outputs in raw result (data_id={data_id}).")

        first_output = first_candidates[0]
        token_logprobs = getattr(first_output, "logprobs", None)
        if not token_logprobs:
            raise RuntimeError(f"missing token logprobs in first candidate (data_id={data_id}).")

        if token_logprobs is None:
            raise RuntimeError(
                f"Pointwise reranking failed to obtain token logprobs after {max_retries} tries"
                f" (data_id={data_id})."
            )

        first_step = token_logprobs[0] if isinstance(token_logprobs, list) else token_logprobs

        def _get_logprob(step_logprobs, candidate_ids):
            if not isinstance(step_logprobs, dict):
                return None
            for candidate_id in candidate_ids:
                item_obj = step_logprobs.get(candidate_id)
                if item_obj is not None:
                    return float(getattr(item_obj, "logprob", item_obj))
                item_obj = step_logprobs.get(str(candidate_id))
                if item_obj is not None:
                    return float(getattr(item_obj, "logprob", item_obj))
            return None

        yes_logprob = _get_logprob(first_step, yes_candidates)
        no_logprob = _get_logprob(first_step, no_candidates)
        if yes_logprob is None or no_logprob is None:
            raise RuntimeError("Failed to extract Yes/No token logprobs from first decoding step.")

        logits = torch.tensor([[yes_logprob, no_logprob]], dtype=torch.float32)
        yes_prob = torch.nn.functional.softmax(logits, dim=1)[:, 0]
        # print(f"LLM Yes Prob - Prompt: {prompt_preview[:200]}, Yes Prob: {yes_prob.item()}")
        return float(yes_prob.item())

    def pointwise_bin_reranking(self, dataset, retrieval_results_path):
        if self.tokenizer is None:
            raise ValueError("BaselineMMPipeline requires generator tokenizer for reranking.")

        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0
        stats = {}

        with open(retrieval_results_path, 'r', encoding='utf-8') as f:
            retrieval_data = {}
            for line in f:
                row = json.loads(line)
                retrieval_data[row.get('data_id')] = row

        if hasattr(dataset, 'data'):
            data_items = list(dataset.data)
        else:
            data_items = list(dataset)

        def _get_field(item, key, default=None):
            if isinstance(item, dict):
                return item.get(key, default)
            try:
                return getattr(item, key)
            except Exception:
                return default

        def _extract_title(doc_text):
            text = str(doc_text or '').strip()
            if not text:
                return ''
            parts = text.split(' - ', 1)
            return parts[0].strip()

        reranked_results = []
        vote_logs = []
        for item in tqdm(data_items, desc="Reranking"):
            data_id = _get_field(item, 'data_id')
            image_id = _get_field(item, 'image_id')
            query = _get_field(item, 'question', '')

            retrieval_row = retrieval_data.get(data_id, {})
            retrieval_results = retrieval_row.get('retrieval_results')
            if not retrieval_results:
                retrieval_results = retrieval_row.get('caption_retrieval_results', [])
            if not isinstance(retrieval_results, list) or len(retrieval_results) == 0:
                continue

            scored_docs = []
            for doc in retrieval_results:
                doc_text = doc.get('text', '') if isinstance(doc, dict) else str(doc)

                score = self._llm_yes_prob(item, doc_text)
                scored_docs.append({
                    'text': doc_text,
                    'title': _extract_title(doc_text),
                    'score': float(score),
                })

            scored_docs = sorted(scored_docs, key=lambda x: x['score'], reverse=True)
            for rank_idx, doc_info in enumerate(scored_docs, 1):
                vote_logs.append({
                    'data_id': data_id,
                    'image_id': image_id,
                    'text_query': query,
                    'doc': doc_info['text'],
                    'vote_count': float(doc_info['score']),
                    'rank': rank_idx,
                })

            reranked_results.append({
                'data_id': data_id,
                'image_id': image_id,
                'reranked_results': scored_docs,
            })

        stats['total_binary_judges'] = int(self.total_compare)
        self._save_phase_jsonl('phase3_reranking', reranked_results, file_name='reranked_results.jsonl')
        self._save_phase_jsonl('phase3_reranking_vote_logs', vote_logs, file_name='naive_vote_logs.jsonl')
        self._save_phase_jsonl('phase3_reranking_stats', stats, file_name='performance_stats.jsonl', append=True)

        return reranked_results

    def _llm_qlm_score(self, item, passage_text):
        if self.generator is None:
            raise ValueError("Generator is required for QLM reranking.")
        if self.tokenizer is None:
            raise ValueError("Generator does not expose tokenizer; cannot compute QLM score.")

        if isinstance(item, dict):
            query = item.get('question', '')
            data_id = item.get('data_id')
        else:
            query = getattr(item, 'question', '')
            data_id = getattr(item, 'data_id', None)

        query = '' if query is None else str(query).strip()
        if not query:
            raise RuntimeError(f"Empty query for QLM scoring (data_id={data_id}).")

        context_prompt = f"Passage: {str(passage_text or '')}\nPlease write a question based on this passage."
        target_suffix = f" {query}"
        target_ids = self.tokenizer.encode(target_suffix, add_special_tokens=False)
        if not target_ids:
            raise RuntimeError(f"Empty target token ids for QLM scoring (data_id={data_id}).")

        cfg = self.config if isinstance(self.config, dict) else {}
        qlm_target_max_tokens = int(cfg.get("pointwise_qlm_max_target_tokens", 128))
        if qlm_target_max_tokens > 0 and len(target_ids) > qlm_target_max_tokens:
            # Keep the tail of query tokens so question intent is preserved.
            target_ids = target_ids[-qlm_target_max_tokens:]

        context_ids = self.tokenizer.encode(context_prompt, add_special_tokens=False)
        model_max_len_cfg = int(cfg.get("max_model_len", 0))
        qlm_max_model_len = int(cfg.get("pointwise_qlm_max_model_len", 4096))
        effective_model_len = qlm_max_model_len
        if model_max_len_cfg > 0:
            effective_model_len = min(model_max_len_cfg, qlm_max_model_len)

        if effective_model_len <= len(target_ids):
            raise RuntimeError(
                f"QLM target query is too long for effective max len={effective_model_len} (data_id={data_id})."
            )

        qlm_max_context_tokens = int(cfg.get("pointwise_qlm_max_context_tokens", 2048))
        initial_context_cap = min(qlm_max_context_tokens, effective_model_len - len(target_ids))
        if initial_context_cap <= 0:
            raise RuntimeError(
                f"Invalid QLM context cap={initial_context_cap} (data_id={data_id})."
            )

        configured_k = cfg.get("pointwise_qlm_prompt_logprobs")
        if configured_k is not None:
            requested_prompt_logprobs = max(1, int(configured_k))
        else:
            # Default to small top-k to minimize memory pressure.
            requested_prompt_logprobs = 5

        def _build_prompt_with_context_cap(context_cap):
            capped_context_ids = context_ids[-context_cap:] if len(context_ids) > context_cap else context_ids
            prompt_ids = capped_context_ids + target_ids
            return self.tokenizer.decode(
                prompt_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

        def _extract_score_from_prompt_logprobs(raw_output_obj):
            prompt_token_ids = getattr(raw_output_obj, "prompt_token_ids", None)
            prompt_logprobs = getattr(raw_output_obj, "prompt_logprobs", None)
            if not prompt_token_ids or prompt_logprobs is None:
                raise RuntimeError(f"missing prompt token logprobs in QLM scoring (data_id={data_id}).")

            if len(target_ids) > len(prompt_token_ids):
                raise RuntimeError(
                    f"target ids longer than prompt ids in QLM scoring (data_id={data_id})."
                )

            start_idx = len(prompt_token_ids) - len(target_ids)
            qlm_score_val = 0.0
            missing_count = 0

            for offset, token_id in enumerate(target_ids):
                idx = start_idx + offset
                if idx >= len(prompt_logprobs):
                    raise RuntimeError(
                        f"prompt logprob index out of range in QLM scoring (data_id={data_id}, idx={idx})."
                    )

                step_logprobs = prompt_logprobs[idx]
                if not isinstance(step_logprobs, dict):
                    raise RuntimeError(
                        f"invalid prompt logprob step type in QLM scoring (data_id={data_id}, idx={idx})."
                    )

                item_obj = step_logprobs.get(token_id)
                if item_obj is None:
                    item_obj = step_logprobs.get(str(token_id))

                if item_obj is None:
                    print(f"Warning! item_obj missing for token_id={token_id} at idx={idx} in QLM scoring (data_id={data_id}).")
                    # vLLM only returns top-k prompt logprobs. If target token is missing,
                    # approximate with a floor below the worst observed candidate to keep ranking stable.
                    floor_logprob = -50.0
                    if len(step_logprobs) > 0:
                        vals = [float(getattr(v, "logprob", v)) for v in step_logprobs.values()]
                        floor_logprob = min(vals) - 5.0
                    qlm_score_val += floor_logprob
                    missing_count += 1
                    continue

                qlm_score_val += float(getattr(item_obj, "logprob", item_obj))

            return float(qlm_score_val), int(missing_count), int(len(prompt_token_ids))

        prompt = _build_prompt_with_context_cap(initial_context_cap)
        try:
            raw_outputs = self.generator.generate(
                [prompt],
                return_raw_output=True,
                max_tokens=1,
                temperature=0,
                prompt_logprobs=requested_prompt_logprobs,
            )
        except Exception as e:
            raise RuntimeError(f"QLM generate() failed (data_id={data_id}): {e}") from e

        if not raw_outputs:
            raise RuntimeError(f"generator returned empty raw outputs in QLM scoring (data_id={data_id}).")

        raw_output = raw_outputs[0]
        qlm_score, missing_count, prompt_token_count = _extract_score_from_prompt_logprobs(raw_output)

        self.total_compare += 1
        self.total_prompt_tokens += int(prompt_token_count)

        if isinstance(self.config, dict) and bool(self.config.get("debug_pointwise_qlm", False)) and missing_count > 0:
            print(
                f"[QLM][WARN] data_id={data_id}: {missing_count}/{len(target_ids)} target tokens missing in top-k prompt_logprobs; used floor approximation."
            )

        return float(qlm_score)

    def _llm_qlm_score_batch(self, item, passage_texts):
        """Compute QLM scores for multiple passages of one item in a single generate() call."""
        if self.generator is None:
            raise ValueError("Generator is required for QLM reranking.")
        if self.tokenizer is None:
            raise ValueError("Generator does not expose tokenizer; cannot compute QLM score.")

        if not isinstance(passage_texts, list) or len(passage_texts) == 0:
            return []

        if isinstance(item, dict):
            query = item.get('question', '')
            data_id = item.get('data_id')
        else:
            query = getattr(item, 'question', '')
            data_id = getattr(item, 'data_id', None)

        query = '' if query is None else str(query).strip()
        if not query:
            raise RuntimeError(f"Empty query for QLM scoring (data_id={data_id}).")

        target_suffix = f" {query}"
        target_ids = self.tokenizer.encode(target_suffix, add_special_tokens=False)
        if not target_ids:
            raise RuntimeError(f"Empty target token ids for QLM scoring (data_id={data_id}).")

        cfg = self.config if isinstance(self.config, dict) else {}
        qlm_target_max_tokens = int(cfg.get("pointwise_qlm_max_target_tokens", 128))
        if qlm_target_max_tokens > 0 and len(target_ids) > qlm_target_max_tokens:
            target_ids = target_ids[-qlm_target_max_tokens:]

        model_max_len_cfg = int(cfg.get("max_model_len", 0))
        qlm_max_model_len = int(cfg.get("pointwise_qlm_max_model_len", 4096))
        effective_model_len = qlm_max_model_len
        if model_max_len_cfg > 0:
            effective_model_len = min(model_max_len_cfg, qlm_max_model_len)

        if effective_model_len <= len(target_ids):
            raise RuntimeError(
                f"QLM target query is too long for effective max len={effective_model_len} (data_id={data_id})."
            )

        qlm_max_context_tokens = int(cfg.get("pointwise_qlm_max_context_tokens", 2048))
        context_cap = min(qlm_max_context_tokens, effective_model_len - len(target_ids))
        if context_cap <= 0:
            raise RuntimeError(
                f"Invalid QLM context cap={context_cap} (data_id={data_id})."
            )

        configured_k = cfg.get("pointwise_qlm_prompt_logprobs")
        if configured_k is not None:
            requested_prompt_logprobs = max(1, int(configured_k))
        else:
            requested_prompt_logprobs = 5

        prompts = []
        for passage_text in passage_texts:
            context_prompt = f"Passage: {str(passage_text or '')}\nPlease write a question based on this passage."
            context_ids = self.tokenizer.encode(context_prompt, add_special_tokens=False)
            capped_context_ids = context_ids[-context_cap:] if len(context_ids) > context_cap else context_ids
            prompt_ids = capped_context_ids + target_ids
            prompts.append(
                self.tokenizer.decode(
                    prompt_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )

        try:
            raw_outputs = self.generator.generate(
                prompts,
                return_raw_output=True,
                max_tokens=1,
                temperature=0,
                prompt_logprobs=requested_prompt_logprobs,
            )
        except Exception as e:
            raise RuntimeError(f"QLM batch generate() failed (data_id={data_id}): {e}") from e

        if not raw_outputs or len(raw_outputs) != len(prompts):
            got = 0 if not raw_outputs else len(raw_outputs)
            raise RuntimeError(
                f"generator returned invalid batch raw outputs in QLM scoring (data_id={data_id}, got={got}, expected={len(prompts)})."
            )

        def _extract_score_from_prompt_logprobs(raw_output_obj):
            prompt_token_ids = getattr(raw_output_obj, "prompt_token_ids", None)
            prompt_logprobs = getattr(raw_output_obj, "prompt_logprobs", None)
            if not prompt_token_ids or prompt_logprobs is None:
                raise RuntimeError(f"missing prompt token logprobs in QLM scoring (data_id={data_id}).")

            if len(target_ids) > len(prompt_token_ids):
                raise RuntimeError(
                    f"target ids longer than prompt ids in QLM scoring (data_id={data_id})."
                )

            start_idx = len(prompt_token_ids) - len(target_ids)
            qlm_score_val = 0.0
            missing_count = 0

            for offset, token_id in enumerate(target_ids):
                idx = start_idx + offset
                if idx >= len(prompt_logprobs):
                    raise RuntimeError(
                        f"prompt logprob index out of range in QLM scoring (data_id={data_id}, idx={idx})."
                    )

                step_logprobs = prompt_logprobs[idx]
                if not isinstance(step_logprobs, dict):
                    raise RuntimeError(
                        f"invalid prompt logprob step type in QLM scoring (data_id={data_id}, idx={idx})."
                    )

                item_obj = step_logprobs.get(token_id)
                if item_obj is None:
                    item_obj = step_logprobs.get(str(token_id))

                if item_obj is None:
                    floor_logprob = -50.0
                    if len(step_logprobs) > 0:
                        vals = [float(getattr(v, "logprob", v)) for v in step_logprobs.values()]
                        floor_logprob = min(vals) - 5.0
                    qlm_score_val += floor_logprob
                    missing_count += 1
                    continue

                qlm_score_val += float(getattr(item_obj, "logprob", item_obj))

            return float(qlm_score_val), int(missing_count), int(len(prompt_token_ids))

        scores = []
        total_prompt_tokens = 0
        total_missing_count = 0
        for raw_output in raw_outputs:
            qlm_score, missing_count, prompt_token_count = _extract_score_from_prompt_logprobs(raw_output)
            scores.append(float(qlm_score))
            total_prompt_tokens += int(prompt_token_count)
            total_missing_count += int(missing_count)

        self.total_compare += len(scores)
        self.total_prompt_tokens += int(total_prompt_tokens)

        if isinstance(self.config, dict) and bool(self.config.get("debug_pointwise_qlm", False)) and total_missing_count > 0:
            print(
                f"[QLM][WARN] data_id={data_id}: total missing target tokens={total_missing_count} across {len(scores)} docs; used floor approximation."
            )

        return scores

    def pointwise_qlm_reranking(self, dataset, retrieval_results_path):
        if self.tokenizer is None:
            raise ValueError("BaselineMMPipeline requires generator tokenizer for reranking.")

        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0
        stats = {}

        with open(retrieval_results_path, 'r', encoding='utf-8') as f:
            retrieval_data = {}
            for line in f:
                row = json.loads(line)
                retrieval_data[row.get('data_id')] = row

        if hasattr(dataset, 'data'):
            data_items = list(dataset.data)
        else:
            data_items = list(dataset)

        def _get_field(item, key, default=None):
            if isinstance(item, dict):
                return item.get(key, default)
            try:
                return getattr(item, key)
            except Exception:
                return default

        def _extract_title(doc_text):
            text = str(doc_text or '').strip()
            if not text:
                return ''
            parts = text.split(' - ', 1)
            return parts[0].strip()

        reranked_results = []
        vote_logs = []
        for item in tqdm(data_items, desc="Reranking"):
            data_id = _get_field(item, 'data_id')
            image_id = _get_field(item, 'image_id')
            query = _get_field(item, 'question', '')

            retrieval_row = retrieval_data.get(data_id, {})
            retrieval_results = retrieval_row.get('retrieval_results')
            if not retrieval_results:
                retrieval_results = retrieval_row.get('caption_retrieval_results', [])
            if not isinstance(retrieval_results, list) or len(retrieval_results) == 0:
                continue

            doc_texts = [doc.get('text', '') if isinstance(doc, dict) else str(doc) for doc in retrieval_results]
            scores = self._llm_qlm_score_batch(item, doc_texts)
            if len(scores) != len(doc_texts):
                raise RuntimeError(
                    f"QLM score size mismatch (data_id={data_id}): got {len(scores)}, expected {len(doc_texts)}."
                )

            scored_docs = []
            for doc_text, score in zip(doc_texts, scores):
                scored_docs.append({
                    'text': doc_text,
                    'title': _extract_title(doc_text),
                    'score': float(score),
                })

            scored_docs = sorted(scored_docs, key=lambda x: x['score'], reverse=True)
            for rank_idx, doc_info in enumerate(scored_docs, 1):
                vote_logs.append({
                    'data_id': data_id,
                    'image_id': image_id,
                    'text_query': query,
                    'doc': doc_info['text'],
                    'vote_count': float(doc_info['score']),
                    'rank': rank_idx,
                })

            reranked_results.append({
                'data_id': data_id,
                'image_id': image_id,
                'reranked_results': scored_docs,
            })

        stats['total_qlm_judges'] = int(self.total_compare)
        stats['total_prompt_tokens'] = int(self.total_prompt_tokens)
        stats['total_completion_tokens'] = int(self.total_completion_tokens)
        self._save_phase_jsonl('phase3_reranking', reranked_results, file_name='reranked_results.jsonl')
        self._save_phase_jsonl('phase3_reranking_vote_logs', vote_logs, file_name='naive_vote_logs.jsonl')
        self._save_phase_jsonl('phase3_reranking_stats', stats, file_name='performance_stats.jsonl', append=True)

        return reranked_results
    
    def listwise_reranking(self, dataset, retrieval_results_path):
        if self.generator is None:
            raise ValueError("Generator is required for listwise reranking.")

        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0
        stats = {}

        cfg = self._get_listwise_cfg()
        window_size = cfg['window_size']
        step_size = cfg['step_size']
        num_repeat = cfg['num_repeat']

        with open(retrieval_results_path, 'r', encoding='utf-8') as f:
            retrieval_data = {}
            for line in f:
                row = json.loads(line)
                retrieval_data[row.get('data_id')] = row

        if hasattr(dataset, 'data'):
            data_items = list(dataset.data)
        else:
            data_items = list(dataset)

        def _get_field(item, key, default=None):
            if isinstance(item, dict):
                return item.get(key, default)
            try:
                return getattr(item, key)
            except Exception:
                return default

        reranked_results = []
        vote_logs = []

        for item in tqdm(data_items, desc="Reranking"):
            data_id = _get_field(item, 'data_id')
            image_id = _get_field(item, 'image_id')
            query = _get_field(item, 'question', '')

            retrieval_row = retrieval_data.get(data_id, {})
            retrieval_results = retrieval_row.get('retrieval_results')
            if not retrieval_results:
                retrieval_results = retrieval_row.get('caption_retrieval_results', [])
            if not isinstance(retrieval_results, list) or len(retrieval_results) == 0:
                continue

            ranking = [{
                'text': doc.get('text', '') if isinstance(doc, dict) else str(doc),
                'title': self._extract_doc_title(doc.get('text', '') if isinstance(doc, dict) else str(doc)),
            } for doc in retrieval_results]

            if len(ranking) > 1:
                ranking = self._apply_listwise_reranking(query, image_id, ranking)
            else:
                print(f"ERROR!")
            scored_docs = []
            for rank_idx, doc_info in enumerate(ranking, 1):
                score = float(-(rank_idx - 1))
                scored_doc = {
                    'text': doc_info['text'],
                    'title': doc_info['title'],
                    'score': score,
                }
                scored_docs.append(scored_doc)
                vote_logs.append({
                    'data_id': data_id,
                    'image_id': image_id,
                    'text_query': query,
                    'doc': doc_info['text'],
                    'vote_count': score,
                    'rank': rank_idx,
                })

            reranked_results.append({
                'data_id': data_id,
                'image_id': image_id,
                'reranked_results': scored_docs,
            })

        stats['total_listwise_compares'] = int(self.total_compare)
        stats['total_prompt_tokens'] = int(self.total_prompt_tokens)
        stats['total_completion_tokens'] = int(self.total_completion_tokens)
        stats['listwise_window_size'] = int(window_size)
        stats['listwise_step_size'] = int(step_size)
        stats['listwise_num_repeat'] = int(num_repeat)

        self._save_phase_jsonl('phase3_reranking', reranked_results, file_name='reranked_results.jsonl')
        self._save_phase_jsonl('phase3_reranking_vote_logs', vote_logs, file_name='naive_vote_logs.jsonl')
        self._save_phase_jsonl('phase3_reranking_stats', stats, file_name='performance_stats.jsonl', append=True)

        return reranked_results
    
    def setwise_reranking(self, dataset, retrieval_results_path):
        if self.generator is None:
            raise ValueError("Generator is required for setwise reranking.")

        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0
        stats = {}

        cfg = self.config if isinstance(self.config, dict) else {}
        setwise_num_child = int(cfg.get("setwise_num_child", 3))
        setwise_k = int(cfg.get("setwise_topk", 1))
        setwise_max_doc_words = int(cfg.get("setwise_doc_max_words", 200))
        setwise_max_tokens = int(cfg.get("setwise_max_tokens", 4))
        setwise_chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        if setwise_num_child <= 0:
            setwise_num_child = 3
        if setwise_max_tokens <= 0:
            setwise_max_tokens = 4

        with open(retrieval_results_path, 'r', encoding='utf-8') as f:
            retrieval_data = {}
            for line in f:
                row = json.loads(line)
                retrieval_data[row.get('data_id')] = row

        if hasattr(dataset, 'data'):
            data_items = list(dataset.data)
        else:
            data_items = list(dataset)

        def _get_field(item, key, default=None):
            if isinstance(item, dict):
                return item.get(key, default)
            try:
                return getattr(item, key)
            except Exception:
                return default

        def _extract_title(doc_text):
            text = str(doc_text or '').strip()
            if not text:
                return ''
            parts = text.split(' - ', 1)
            return parts[0].strip()

        def _normalize_doc_text(doc_text):
            text = str(doc_text or '')
            if setwise_max_doc_words > 0:
                text = ' '.join(text.split()[:setwise_max_doc_words])
            return text

        def _build_setwise_prompt(item, docs):
            query = item.question
            image_query_id = item.image_id
            image_query_path = os.path.join(self.config['dataset_path'], 'images', f'{image_query_id}.jpg')
            question_image = Image.open(image_query_path).convert('RGB')

            lines = [
                f"Given a query \"{query}\", which of the following passages is the most relevant one to the query?",
                "",
            ]
            for i, doc in enumerate(docs):
                label = setwise_chars[i]
                doc_text = _normalize_doc_text(doc.get('text', '') if isinstance(doc, dict) else doc)
                lines.extend([
                    f"Passage {label}: \"{doc_text}\"",
                    "",
                ])
            lines.append("Output only the passage label(A, B, C...) of the most relevant passage:")

            content_list = []
            content_list.append({'type': 'image', 'image': question_image})
            content_list.append({'type': 'text', 'text': "\n".join(lines)})
            messages = []
            messages.append({"role": "user", "content": content_list})
            return messages

        def _parse_setwise_output(output_text, num_docs):
            text = str(output_text or '').strip().upper()
            valid_chars = set(setwise_chars[:num_docs])
            if text in valid_chars:
                return text

            match = re.search(r'PASSAGE\s+([A-Z])', text)
            if match:
                candidate = match.group(1)
                if candidate in valid_chars:
                    return candidate

            match = re.search(r'\b([A-Z])\b', text)
            if match:
                candidate = match.group(1)
                if candidate in valid_chars:
                    return candidate

            return setwise_chars[0]

        def _compare(item, docs):
            self.total_compare += 1
            prompt = _build_setwise_prompt(item, docs)

            if self.tokenizer is not None:
                try:
                    self.total_prompt_tokens += len(self.tokenizer.encode(prompt, add_special_tokens=False))
                except Exception:
                    pass

            outputs = self.generator.generate(
                [prompt],
                max_tokens=setwise_max_tokens,
                temperature=0,
            )
            if not outputs:
                return setwise_chars[0]

            output_text = outputs[0] if isinstance(outputs[0], str) else str(outputs[0])
            if self.tokenizer is not None:
                try:
                    self.total_completion_tokens += len(self.tokenizer.encode(output_text, add_special_tokens=False))
                except Exception:
                    pass

            return _parse_setwise_output(output_text, len(docs))

        def _heapify(arr, n, i, item):
            if setwise_num_child * i + 1 < n:
                end_idx = min((setwise_num_child * (i + 1) + 1), n)
                docs = [arr[i]] + arr[setwise_num_child * i + 1:end_idx]
                inds = [i] + list(range(setwise_num_child * i + 1, end_idx))
                output = _compare(item, docs)
                try:
                    best_ind = setwise_chars.index(output)
                except ValueError:
                    best_ind = 0
                try:
                    largest = inds[best_ind]
                except IndexError:
                    largest = i
                if largest != i:
                    arr[i], arr[largest] = arr[largest], arr[i]
                    _heapify(arr, n, largest, item)

        def _heap_sort(arr, item, k):
            n = len(arr)
            ranked = 0
            for i in range(n // setwise_num_child, -1, -1):
                _heapify(arr, n, i, item)
            for i in range(n - 1, 0, -1):
                arr[i], arr[0] = arr[0], arr[i]
                ranked += 1
                if ranked == k:
                    break
                _heapify(arr, i, 0, item)

        reranked_results = []
        vote_logs = []

        for item in tqdm(data_items, desc="Reranking"):
            data_id = _get_field(item, 'data_id')
            image_id = _get_field(item, 'image_id')
            query = _get_field(item, 'question', '')

            retrieval_row = retrieval_data.get(data_id, {})
            retrieval_results = retrieval_row.get('retrieval_results')
            if not retrieval_results:
                retrieval_results = retrieval_row.get('caption_retrieval_results', [])
            if not isinstance(retrieval_results, list) or len(retrieval_results) == 0:
                continue

            ranking = [{
                'text': doc.get('text', '') if isinstance(doc, dict) else str(doc),
                'title': _extract_title(doc.get('text', '') if isinstance(doc, dict) else str(doc)),
            } for doc in retrieval_results]

            if len(ranking) > 1:
                original_ranking = list(ranking)
                k = setwise_k if setwise_k > 0 else len(ranking)
                k = min(k, len(ranking))
                _heap_sort(ranking, item, k)
                ranking = list(reversed(ranking))

                top_reranked = ranking[:k]
                top_ids = {id(doc_info) for doc_info in top_reranked}
                remaining = [doc_info for doc_info in original_ranking if id(doc_info) not in top_ids]
                ranking = top_reranked + remaining

            scored_docs = []
            for rank_idx, doc_info in enumerate(ranking, 1):
                score = float(-(rank_idx - 1))
                scored_doc = {
                    'text': doc_info['text'],
                    'title': doc_info['title'],
                    'score': score,
                }
                scored_docs.append(scored_doc)
                vote_logs.append({
                    'data_id': data_id,
                    'image_id': image_id,
                    'text_query': query,
                    'doc': doc_info['text'],
                    'vote_count': score,
                    'rank': rank_idx,
                })

            reranked_results.append({
                'data_id': data_id,
                'image_id': image_id,
                'reranked_results': scored_docs,
            })

        stats['total_setwise_compares'] = int(self.total_compare)
        stats['total_prompt_tokens'] = int(self.total_prompt_tokens)
        stats['total_completion_tokens'] = int(self.total_completion_tokens)
        stats['setwise_num_child'] = int(setwise_num_child)
        stats['setwise_topk'] = int(setwise_k)

        self._save_phase_jsonl('phase3_reranking', reranked_results, file_name='reranked_results.jsonl')
        self._save_phase_jsonl('phase3_reranking_vote_logs', vote_logs, file_name='naive_vote_logs.jsonl')
        self._save_phase_jsonl('phase3_reranking_stats', stats, file_name='performance_stats.jsonl', append=True)

        return reranked_results



    def pairwise_reranking(self, dataset, retrieval_results_path):
        if self.generator is None:
            raise ValueError("Generator is required for pairwise reranking.")

        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0
        stats = {}

        cfg = self.config if isinstance(self.config, dict) else {}
        pairwise_k = int(cfg.get("pairwise_topk", 1))
        pairwise_max_doc_words = int(cfg.get("pairwise_doc_max_words", 200))
        pairwise_max_tokens = int(cfg.get("pairwise_max_tokens", 4))

        if pairwise_max_tokens <= 0:
            pairwise_max_tokens = 4

        with open(retrieval_results_path, 'r', encoding='utf-8') as f:
            retrieval_data = {}
            for line in f:
                row = json.loads(line)
                retrieval_data[row.get('data_id')] = row

        if hasattr(dataset, 'data'):
            data_items = list(dataset.data)
        else:
            data_items = list(dataset)

        def _get_field(item, key, default=None):
            if isinstance(item, dict):
                return item.get(key, default)
            try:
                return getattr(item, key)
            except Exception:
                return default

        def _extract_title(doc_text):
            text = str(doc_text or '').strip()
            if not text:
                return ''
            parts = text.split(' - ', 1)
            return parts[0].strip()
        
        def _build_pairwise_prompt(item, doc_a, doc_b):
            query = item.question
            image_query_id = item.image_id
            image_query_path = os.path.join(self.config['dataset_path'], 'images', f'{image_query_id}.jpg')
            question_image = Image.open(image_query_path).convert('RGB')

            lines = [
                f"Given a query \"{query}\", which of the following two passages is more relevant to the query?",
                "",
                f"Passage A: \"{doc_a}\"",
                "",
                f"Passage B: \"{doc_b}\"",
                "",
                "Output Passage A or Passage B:",
            ]
            content_list = []
            content_list.append({'type': 'image', 'image': question_image})
            content_list.append({'type': 'text', 'text': "\n".join(lines)})
            messages = []
            messages.append({"role": "user", "content": content_list})
            return messages

        def _normalize_doc_text(doc_text):
            text = str(doc_text or '')
            if pairwise_max_doc_words > 0:
                text = ' '.join(text.split()[:pairwise_max_doc_words])
            return text

        def _parse_pairwise_output(output_text):
            text = str(output_text or '').strip().upper()
            if "PASSAGE A" in text or text == "A":
                return "A"
            if "PASSAGE B" in text or text == "B":
                return "B"
            return None

        def _single_compare(item, doc_a, doc_b):
            self.total_compare += 1
            prompt = _build_pairwise_prompt(item, doc_a, doc_b)

            if self.tokenizer is not None:
                try:
                    self.total_prompt_tokens += len(self.tokenizer.encode(prompt, add_special_tokens=False))
                except Exception:
                    pass

            outputs = self.generator.generate(
                [prompt],
                max_tokens=pairwise_max_tokens,
                temperature=0,
            )
            if not outputs:
                return None

            output_text = outputs[0] if isinstance(outputs[0], str) else str(outputs[0])
            if self.tokenizer is not None:
                try:
                    self.total_completion_tokens += len(self.tokenizer.encode(output_text, add_special_tokens=False))
                except Exception:
                    pass

            return _parse_pairwise_output(output_text)

        def _compare_pair(item, doc_left, doc_right):
            out1 = _single_compare(item, doc_left, doc_right)
            out2 = _single_compare(item, doc_right, doc_left)

            # Follow pairwise_ranker logic: two-direction consistency check.
            if out1 == "A" and out2 == "B":
                return "left"
            if out1 == "B" and out2 == "A":
                return "right"
            return "tie"

        reranked_results = []
        vote_logs = []

        for item in tqdm(data_items, desc="Reranking"):
            data_id = _get_field(item, 'data_id')
            image_id = _get_field(item, 'image_id')
            query = _get_field(item, 'question', '')

            retrieval_row = retrieval_data.get(data_id, {})
            retrieval_results = retrieval_row.get('retrieval_results')
            if not retrieval_results:
                retrieval_results = retrieval_row.get('caption_retrieval_results', [])
            if not isinstance(retrieval_results, list) or len(retrieval_results) == 0:
                continue

            ranking = [{
                'text': doc.get('text', '') if isinstance(doc, dict) else str(doc),
                'title': _extract_title(doc.get('text', '') if isinstance(doc, dict) else str(doc)),
            } for doc in retrieval_results]

            if len(ranking) > 1:
                k = pairwise_k if pairwise_k > 0 else len(ranking)
                k = min(k, len(ranking))

                # Bubble top-k using pairwise comparator
                last_end = len(ranking) - 1
                for i in range(k):
                    current_ind = last_end
                    is_change = False
                    while True:
                        if current_ind <= i:
                            break
                        doc_right = ranking[current_ind].get('text', '')
                        doc_left = ranking[current_ind - 1].get('text', '')
                        left_text = _normalize_doc_text(doc_left)
                        right_text = _normalize_doc_text(doc_right)
                        winner = _compare_pair(item, left_text, right_text)
                        if winner == "right":
                            ranking[current_ind - 1], ranking[current_ind] = ranking[current_ind], ranking[current_ind - 1]
                            
                            if not is_change:
                                is_change = True
                                # print(f"New ranking for item {data_id} after comparing doc {current_ind-1} and doc {current_ind}: {[doc.get('text', '')[:100] if isinstance(doc, dict) else str(doc) for doc in ranking]}")
                                if last_end != len(ranking) - 1:
                                    last_end += 1
                        if not is_change:
                            last_end -= 1
                        current_ind -= 1
            scored_docs = []
            for rank_idx, doc_info in enumerate(ranking, 1):
                score = float(-(rank_idx - 1))
                scored_doc = {
                    'text': doc_info['text'],
                    'title': doc_info['title'],
                    'score': score,
                }
                scored_docs.append(scored_doc)
                vote_logs.append({
                    'data_id': data_id,
                    'image_id': image_id,
                    'text_query': query,
                    'doc': doc_info['text'],
                    'vote_count': score,
                    'rank': rank_idx,
                })

            reranked_results.append({
                'data_id': data_id,
                'image_id': image_id,
                'reranked_results': scored_docs,
            })

        stats['total_pairwise_compares'] = int(self.total_compare)
        stats['total_prompt_tokens'] = int(self.total_prompt_tokens)
        stats['total_completion_tokens'] = int(self.total_completion_tokens)
        stats['pairwise_topk'] = int(pairwise_k)

        self._save_phase_jsonl('phase3_reranking', reranked_results, file_name='reranked_results.jsonl')
        self._save_phase_jsonl('phase3_reranking_vote_logs', vote_logs, file_name='naive_vote_logs.jsonl')
        self._save_phase_jsonl('phase3_reranking_stats', stats, file_name='performance_stats.jsonl', append=True)

        return reranked_results
    
