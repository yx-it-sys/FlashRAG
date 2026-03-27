from flashrag.evaluator import Evaluator
from flashrag.utils import get_retriever, get_generator
from flashrag.pipeline import BasicMultiModalPipeline
import re
import os
import json
from PIL import Image
import tomllib
from tqdm import tqdm
import time
import openai

class OmniSearchPipeline(BasicMultiModalPipeline):
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        super().__init__(config, prompt_template)
        self.config = config
        self.data_dir = self.config['data_dir']
        self.dataset_name = self.config['dataset_name']
        self.generator = get_generator(config) if generator is None else generator
        if retriever is not None:
            self.retriever = retriever
        elif "text_retriever_config" in self.config or "image_retriever_config" in self.config:
            self.retriever = None
        else:
            self.retriever = get_retriever(config)
        prompt_path = self.config['omni_prompt_path']
        with open(prompt_path, 'rb') as f:
            self.prompt = tomllib.load(f)['system_prompt']
        self.output_dir = self.config['output_dir'] if 'output_dir' in self.config and self.config['output_dir'] else self.config['save_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        self.trajectory_path = os.path.join(self.output_dir, "omnisearch_trajectories.jsonl")
        self.retrieval_char_limit = int(self.config["omni_retrieval_char_limit"])

    def _retriever_device_context(self, device):
        if device != "cpu":
            from contextlib import nullcontext

            return nullcontext()

        from contextlib import contextmanager
        import torch
        import flashrag.retriever.encoder as retriever_encoder_module
        import flashrag.retriever.utils as retriever_utils_module

        @contextmanager
        def cpu_context():
            original_encoder_get_device = retriever_encoder_module.get_device
            original_utils_get_device = retriever_utils_module.get_device
            original_module_cuda = torch.nn.Module.cuda
            original_tensor_cuda = torch.Tensor.cuda
            retriever_encoder_module.get_device = lambda: "cpu"
            retriever_utils_module.get_device = lambda: "cpu"
            torch.nn.Module.cuda = lambda self, device=None, *args, **kwargs: self.to("cpu")
            torch.Tensor.cuda = lambda self, device=None, non_blocking=False, memory_format=None: self.to("cpu")
            try:
                yield
            finally:
                retriever_encoder_module.get_device = original_encoder_get_device
                retriever_utils_module.get_device = original_utils_get_device
                torch.nn.Module.cuda = original_module_cuda
                torch.Tensor.cuda = original_tensor_cuda

        return cpu_context()

    def _normalize_generation_output(self, response):
        if isinstance(response, str):
            return response
        if isinstance(response, list):
            flattened = []
            for item in response:
                normalized = self._normalize_generation_output(item)
                if normalized:
                    flattened.append(normalized)
            return "\n".join(flattened).strip()
        if response is None:
            return ""
        return str(response)

    def _generate_text(self, messages):
        delay = 2
        max_retries = 5
        for attempt in range(max_retries):
            try:
                raw_response = self.generator.generate([messages])
                return self._normalize_generation_output(raw_response)
            except openai.RateLimitError as e:
                if attempt == max_retries - 1:
                    raise
                print(f"Rate limit hit, retrying in {delay}s ...")
                time.sleep(delay)
                delay *= 2
        raw_response = self.generator.generate([messages])
        return self._normalize_generation_output(raw_response)

    def _clip_text(self, text, limit):
        if text is None:
            return None
        text = str(text)
        if len(text) <= limit:
            return text
        clipped = text[:limit]
        return clipped + "\n\n[Truncated due to prompt length limit]"

    def _build_followup_message(self, retrieval_mode, retrieval_content):
        if retrieval_mode == "no_retrieval":
            return (
                "You chose `No Retrieval`, which means external retrieval is unnecessary for this sub-question. "
                "Do not say that retrieval failed or that no relevant information was found. "
                "Continue reasoning from the image and prior conversation, then either output the next step or the final answer."
            )

        if retrieval_content:
            return f"Contents of retrieved documents:\n{retrieval_content}"

        return (
            "Retrieval was attempted but returned no useful results. "
            "You may refine the query, switch retrieval mode, or conclude if the answer cannot be grounded."
        )

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

    def _format_retrieved_doc(self, doc, preferred_field=None):
        if isinstance(doc, str):
            return doc
        if isinstance(doc, dict):
            if preferred_field:
                value = doc.get(preferred_field)
                if value:
                    return str(value)
            for key in ("text", "contents", "content", "body", "passage"):
                value = doc.get(key)
                if value:
                    return str(value)
            return json.dumps(doc, ensure_ascii=False)
        return str(doc)

    def _format_retrieval_content(self, retrieved_docs, preferred_field=None):
        if not retrieved_docs:
            return ""
        if not isinstance(retrieved_docs, list):
            retrieved_docs = [retrieved_docs]
        return "\n\n".join(
            [f"Doc{i+1}:\n{self._format_retrieved_doc(doc, preferred_field=preferred_field)}" for i, doc in enumerate(retrieved_docs)]
        )

    def _log_retrieval_preview(self, retrieval_content):
        if retrieval_content is None:
            print("Retrieval result: None")
            return
        preview_limit = min(self.retrieval_char_limit, 500)
        preview = str(retrieval_content)
        if len(preview) > preview_limit:
            preview = preview[:preview_limit] + "\n\n[Preview truncated in log]"
        print(f"Retrieval result:\n{preview}")

    def _search_text_docs(self, query_txt):
        from copy import deepcopy

        text_cfg = deepcopy(self.config.final_config)
        text_cfg.update(deepcopy(self.config["text_retriever_config"]))
        text_device = text_cfg.get("retrieval_device", "cuda")
        with self._retriever_device_context(text_device):
            if not hasattr(self, "_text_retriever"):
                self._text_retriever = get_retriever(text_cfg)
            return self._text_retriever.search(query_txt)

    def _search_image_docs(self, img):
        from copy import deepcopy
        from flashrag.retriever.retriever import MultiModalRetriever

        image_cfg = deepcopy(self.config.final_config)
        image_cfg.update(deepcopy(self.config["image_retriever_config"]))
        target_modal = image_cfg.get("image_retrieval_target_modal", "image")
        image_device = image_cfg.get("retrieval_device", "cuda")
        with self._retriever_device_context(image_device):
            if not hasattr(self, "_image_retriever"):
                self._image_retriever = MultiModalRetriever(image_cfg)
            return self._image_retriever.search(img, target_modal=target_modal)

    def _serialize_for_log(self, value):
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {str(k): self._serialize_for_log(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._serialize_for_log(v) for v in value]
        if isinstance(value, Image.Image):
            return {"type": "pil_image", "size": list(value.size), "mode": value.mode}
        return str(value)

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

        if not nodes:
            final_answer_match = re.search(r"Final Answer:\s*(.*?)(?=$)", response, re.DOTALL)
            if final_answer_match:
                final_answer = final_answer_match.group(1).strip().replace("\n", "")
                if final_answer:
                    nodes.append({
                        "action": "final_answer",
                        "content": final_answer,
                    })
        return nodes

    def _record_response_actions(self, trajectory, response):
        trajectory.extend(self._extract_action_nodes(response))

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
    
    def iterative_infer(self, question, id, image_id):
        img_path = os.path.join(self.data_dir, self.dataset_name, "images", f"{image_id}.jpg")
        
        if not os.path.exists(img_path):
            return None
        
        img = Image.open(img_path).convert("RGB")
        messages = [
            {"role": "system", "content": [
                {"type": "text", "text": self.prompt},
            ]},
            {"role": "user", "content":[
                {"type": "text", "text": f"Input Question: {question}"},
                {"type": "image", "image": img}
            ]}

        ]
        trajectory = []
        start_time = time.time()
        response = self._generate_text(messages)
        response = self._truncate_after_search(response)
        print(f"First Response: {response}")
        self._record_response_actions(trajectory, response)
        messages.append({'role': 'assistant', 'content': response})
        
        conversation_num, max_turns = 0, 5
        while conversation_num < max_turns:
            if "Final Answer" in response or "<Final Answer>" in response:
                break
            need_txt_ret = "Text Retrieval" in response
            need_img_ret = "Image Retrieval" in response
            need_no_ret = "No Retrieval" in response
            if need_txt_ret or need_img_ret or need_no_ret:
                retrieval_content = ""
                retrieval_mode = None
                query_txt = ""
                retrieved_docs = []
                if need_txt_ret:
                    retrieval_mode = "text_retrieval"
                    print("Start Text Retrieval...")
                    pattern = r'Text Retrieval[:\s"]*(.*?)(?=<|$)'
                    match = re.search(pattern, response, re.DOTALL)
                    if match:
                        query_txt = match.group(1).strip()
                        print(f"Query Text: {query_txt}")
                    if query_txt == "":
                        print(f"ERROR!!Query_txt is None")
                        retrieved_docs = self._search_text_docs(query_txt)
                        retrieval_content = self._format_retrieval_content(retrieved_docs)
                        retrieval_content = self._clip_text(retrieval_content, self.retrieval_char_limit)
                        self._log_retrieval_preview(retrieval_content)
                    else:
                        retrieved_docs = self._search_text_docs(query_txt)
                        retrieval_content = self._format_retrieval_content(retrieved_docs)
                        retrieval_content = self._clip_text(retrieval_content, self.retrieval_char_limit)
                        self._log_retrieval_preview(retrieval_content)
                elif need_img_ret:
                    retrieval_mode = "image_retrieval"
                    print("Start Image Retrieval...")
                    retrieved_docs = self._search_image_docs(img)
                    image_return_field = self.config["image_retriever_config"].get("image_retrieval_return_field", "title")
                    retrieval_content = self._format_retrieval_content(
                        retrieved_docs,
                        preferred_field=image_return_field,
                    )
                    retrieval_content = self._clip_text(retrieval_content, self.retrieval_char_limit)
                    self._log_retrieval_preview(retrieval_content)
                elif need_no_ret:
                    retrieval_mode = "no_retrieval"
                    retrieval_content = None

                self._record_retrieval_result(
                    trajectory=trajectory,
                    retrieval_mode=retrieval_mode,
                    query_txt=query_txt,
                    retrieval_content=retrieval_content,
                )

                contents = []
                contents.append({
                    'type': 'text',
                    'text': self._build_followup_message(retrieval_mode, retrieval_content),
                })

                messages.append({'role': 'user', 'content': contents})

                try:
                    response = self._generate_text(messages)
                    response = self._truncate_after_search(response)
                    print(f"Response: {response}")
                    self._record_response_actions(trajectory, response)
                    messages.append({"role":"assistant", "content": response})
                except Exception as e:
                    print("Inference error, hidden states ignored:", e)
                    self._write_trajectory({
                        "question": question,
                        "id": id,
                        "final_answer": response,
                        "status": "generation_error",
                        "duration_seconds": time.time() - start_time,
                        "trajectory": trajectory,
                    })
                    return response, messages
                       
            conversation_num += 1
        
        pattern = r'(?:<Final Answer>|Final Answer:)\s*(.*?)(?=<|$)'
        final_answer_match = re.search(pattern, response, re.DOTALL)

        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            final_answer = final_answer.replace('\n', '')
            print(f"Final Answer: {final_answer}")
            self._write_trajectory({
                "question": question,
                "id": id,
                "final_answer": final_answer,
                "status": "ok",
                "duration_seconds": time.time() - start_time,
                "trajectory": trajectory,
            })
            return final_answer, messages
        else:
            print(f"Warning: reached end of agent loop for item {conversation_num} without a 'Final Answer'. returning last response")
            self._write_trajectory({
                "question": question,
                "id": id,
                "final_answer": response,
                "status": "missing_final_answer",
                "duration_seconds": time.time() - start_time,
                "trajectory": trajectory,
            })
            return response, messages
      
    def safe_write(self, file_path: str, data: dict):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        questions = dataset.question
        image_ids = dataset.image_id
        ids = dataset.id
        prediction_list = []

        # 2. 记录开始时间
        start_time = time.time()

        for question, id , image_id in tqdm(zip(questions, ids, image_ids), total=len(questions)):
            answer, context = self.iterative_infer(question, id, image_id)
            prediction_list.append(answer)

        dataset.update_output("pred", prediction_list)
        dataset = self.evaluate(dataset, do_eval=do_eval)
        end_time = time.time()
        total_duration = end_time - start_time
        count = len(questions)
        
        avg_time = total_duration / count if count > 0 else 0

        print(f"\n[Timing] Total: {total_duration:.2f}s | Count: {count} | Avg per item: {avg_time:.4f}s")
        with open(os.path.join(self.output_dir, "records.txt"), "a", encoding="utf-8") as f:
            f.write(f"\n[Timing] Total: {total_duration:.2f}s | Count: {count} | Avg per item: {avg_time:.4f}s")
        return dataset
