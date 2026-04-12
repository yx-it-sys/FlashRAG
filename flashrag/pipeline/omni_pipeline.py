from flashrag.evaluator import Evaluator
from flashrag.utils import get_retriever, get_generator
from flashrag.pipeline import BasicMultiModalPipeline
import re
import os
import json
import subprocess
import tempfile
from PIL import Image
import tomllib
from tqdm import tqdm
import time
import openai
import inspect

class OmniSearchPipeline(BasicMultiModalPipeline):
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        super().__init__(config, prompt_template)
        self.config = config
        self.data_dir = self.config['data_dir']
        self.dataset_name = self.config['dataset_name']
        self.generator = get_generator(config) if generator is None else generator
        self.retriever = retriever if retriever is not None else get_retriever(config)
        prompt_path = self.config['omni_prompt_path']
        with open(prompt_path, 'rb') as f:
            self.prompt = tomllib.load(f)['system_prompt']
        self.output_dir = self.config['output_dir'] if 'output_dir' in self.config and self.config['output_dir'] else self.config['save_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        self.trajectory_path = os.path.join(self.output_dir, "omnisearch_trajectories.jsonl")
        self.retrieval_char_limit = int(self.config["omni_retrieval_char_limit"])
        self.roi_preprocess_config = self.config["roi_preprocess_config"]

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

    def  _build_followup_message(self, retrieval_mode, retrieval_content):
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
        return self._search_with_main_retriever(query_txt, query_kind="text")

    def _search_with_main_retriever(self, query, query_kind):
        search_signature = inspect.signature(self.retriever.search)
        search_params = search_signature.parameters
        kwargs = {}

        if query_kind == "image":
            topk = self.config["image_retrieval_topk"]
        else:
            topk = self.config["text_retrieval_topk"]

        if "num" in search_params and topk is not None:
            kwargs["num"] = topk
        if "query_type" in search_params:
            kwargs["query_type"] = query_kind
        if query_kind == "image" and "target_modal" in search_params:
            kwargs["target_modal"] = self.config["image_retrieval_target_modal"]

        return self.retriever.search(query, **kwargs)

    def _extract_retrieval_query(self, response, retrieval_label):
        pattern = rf'{re.escape(retrieval_label)}[:\s"]*(.*?)(?=<|$)'
        match = re.search(pattern, response, re.DOTALL)
        if not match:
            return ""
        return match.group(1).strip()

    def _run_roi_preprocess(self, img, image_query):
        roi_cfg = self.roi_preprocess_config
        if not roi_cfg or not roi_cfg.get("enabled", False):
            return img
        if not image_query:
            return img

        python_bin = roi_cfg.get("python_bin")
        script_path = roi_cfg.get("script_path")
        if not python_bin or not script_path:
            print("ROI preprocessing skipped: missing python_bin or script_path.")
            return img

        output_dir = roi_cfg.get("output_dir", os.path.join(self.output_dir, "roi_cache"))
        os.makedirs(output_dir, exist_ok=True)

        with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".png", delete=False) as src_file:
            src_path = src_file.name
        with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".png", delete=False) as crop_file:
            crop_path = crop_file.name
        with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".json", delete=False) as meta_file:
            meta_path = meta_file.name

        img.save(src_path)

        command = [
            python_bin,
            script_path,
            "--image-path", src_path,
            "--phrase", image_query,
            "--output-path", crop_path,
            "--json-output", meta_path,
        ]

        optional_args = {
            "--groundingdino-root": roi_cfg.get("groundingdino_root"),
            "--config-file": roi_cfg.get("config_file"),
            "--checkpoint-path": roi_cfg.get("checkpoint_path"),
            "--device": roi_cfg.get("device"),
            "--box-threshold": roi_cfg.get("box_threshold"),
            "--text-threshold": roi_cfg.get("text_threshold"),
            "--expand-ratio": roi_cfg.get("expand_ratio"),
        }
        for flag, value in optional_args.items():
            if value is not None:
                command.extend([flag, str(value)])

        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("VIRTUAL_ENV", None)
        env.pop("ALL_PROXY", None)
        env.pop("all_proxy", None)

        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            if completed.stdout.strip():
                print(f"ROI preprocess stdout:\n{completed.stdout.strip()}")
            if completed.stderr.strip():
                print(f"ROI preprocess stderr:\n{completed.stderr.strip()}")
            with Image.open(crop_path) as cropped:
                return cropped.convert("RGB")
        except subprocess.CalledProcessError as exc:
            stdout = exc.stdout.strip() if exc.stdout else ""
            stderr = exc.stderr.strip() if exc.stderr else ""
            if stdout:
                print(f"ROI preprocess stdout before failure:\n{stdout}")
            if stderr:
                print(f"ROI preprocess stderr before failure:\n{stderr}")
            print(f"ROI preprocessing failed, fallback to original image: {exc}")
            return img
        except Exception as exc:
            print(f"ROI preprocessing failed, fallback to original image: {exc}")
            return img

    def _search_image_docs(self, img, image_query=""):
        retrieval_img = self._run_roi_preprocess(img, image_query)
        return self._search_with_main_retriever(retrieval_img, query_kind="image")

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
                    query_txt = self._extract_retrieval_query(response, "Text Retrieval")
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
                    query_txt = self._extract_retrieval_query(response, "Image Retrieval")
                    print(f"Image Query: {query_txt}")
                    retrieved_docs = self._search_image_docs(img, query_txt)
                    retrieval_content = self._format_retrieval_content(retrieved_docs)
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
