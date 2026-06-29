from flashrag.evaluator import Evaluator
from flashrag.utils import get_retriever, get_generator
from flashrag.pipeline import BasicMultiModalPipeline
import atexit
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
    RETRYABLE_API_ERROR_NAMES = {
        "RateLimitError",
        "APIConnectionError",
        "APITimeoutError",
        "APIStatusError",
        "InternalServerError",
    }
    RETRYABLE_API_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}

    @staticmethod
    def _config_value(config, key, default=None):
        if hasattr(config, "final_config"):
            return config.final_config.get(key, default)
        if isinstance(config, dict):
            return config.get(key, default)
        value = config[key]
        return default if value is None else value

    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        super().__init__(config, prompt_template)
        self.config = config
        self.data_dir = self.config['data_dir']
        self.dataset_name = self.config['dataset_name']
        prompt_version = self.config['omni_prompt_version']
        self.generator = get_generator(config) if generator is None else generator
        self.retriever = retriever if retriever is not None else get_retriever(config)
        prompt_path = self.config['omni_prompt_path']
        with open(prompt_path, 'rb') as f:
            self.prompt = tomllib.load(f)[prompt_version]
        self.output_dir = self.config['output_dir'] if 'output_dir' in self.config and self.config['output_dir'] else self.config['save_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        self.trajectory_path = os.path.join(self.output_dir, "omnisearch_trajectories.jsonl")
        self.retrieval_char_limit = int(self.config["omni_retrieval_char_limit"])
        self.retrieved_doc_char_limit = int(
            self._config_value(self.config, "omni_retrieved_doc_char_limit", self.retrieval_char_limit)
        )
        self.roi_preprocess_config = self.config["roi_preprocess_config"]
        self.target_source = self._normalize_source_name(self._config_value(self.config, "source"))
        self._roi_worker_process = None
        self._roi_worker_exit_registered = False

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

    @staticmethod
    def _format_exception(exc):
        return f"{exc.__class__.__name__}: {exc}"

    def _is_retryable_api_error(self, exc):
        if not isinstance(exc, Exception):
            return False
        if exc.__class__.__name__ in self.RETRYABLE_API_ERROR_NAMES:
            return True
        status_code = getattr(exc, "status_code", None)
        if status_code in self.RETRYABLE_API_STATUS_CODES:
            return True
        cause = getattr(exc, "__cause__", None)
        if cause is not None and self._is_retryable_api_error(cause):
            return True
        return False

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

    def _write_failed_trajectory(self, *, question, id, response, trajectory, start_time, status, error_text):
        trajectory.append({
            "action": "error",
            "content": error_text,
        })
        self._write_trajectory({
            "question": question,
            "id": id,
            "final_answer": response,
            "status": status,
            "error": error_text,
            "duration_seconds": time.time() - start_time,
            "trajectory": trajectory,
        })

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
            return (
                "Based on the retrieval you requested, the retrieved evidence is:\n"
                f"{retrieval_content}"
            )

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

    @staticmethod
    def _truncate_crag_image_text(value, max_chars):
        text = str(value)
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + " ..."

    def _is_crag_image_result(self, doc):
        return isinstance(doc, dict) and "entities" in doc

    def _format_crag_image_retrieval_content(self, retrieved_docs):
        max_attr_chars = int(self._config_value(self.config, "crag_image_attr_max_chars", 400))
        max_entity_chars = int(self._config_value(self.config, "crag_image_entity_max_chars", 4000))
        chunks = []
        for i, item in enumerate(retrieved_docs, 1):
            lines = [f"{i}."]
            entities = item.get("entities", [])
            if not entities:
                lines.append("- entities: []")
            else:
                for entity_idx, ent in enumerate(entities, 1):
                    lines.append(f"- entity_{entity_idx}_name: {ent.get('entity_name', 'Unknown')}")
                    attrs = ent.get("entity_attributes", {}) or {}
                    for key in sorted(attrs.keys()):
                        value = attrs[key]
                        if value is None or value == "":
                            continue
                        lines.append(f"  {key}: {self._truncate_crag_image_text(value, max_attr_chars)}")
            entity_text = "\n".join(lines)
            chunks.append(self._truncate_crag_image_text(entity_text, max_entity_chars))
        return "\n\n".join(chunks)

    def _format_image_retrieval_content(self, retrieved_docs):
        if not retrieved_docs:
            return ""
        if isinstance(retrieved_docs, str):
            return retrieved_docs
        if not isinstance(retrieved_docs, list):
            retrieved_docs = [retrieved_docs]
        if retrieved_docs and all(self._is_crag_image_result(doc) for doc in retrieved_docs):
            return self._format_crag_image_retrieval_content(retrieved_docs)

        per_doc_limit = max(0, self.retrieved_doc_char_limit)
        return "\n\n".join(
            [
                f"Doc{i+1}:\n{self._clip_text(self._format_retrieved_doc(doc), per_doc_limit)}"
                for i, doc in enumerate(retrieved_docs)
            ]
        )

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

    def _extract_retrieval_query(self, response, retrieval_label):
        action_mode, query_txt = self._parse_search_action(response)
        expected_mode = {
            "Text Retrieval": "text_retrieval",
            "Image Retrieval": "image_retrieval",
        }.get(retrieval_label)
        if action_mode != expected_mode:
            return ""
        return query_txt

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

    def _parse_search_action(self, response):
        search_body = self._extract_search_body(response)
        if not search_body:
            return None, ""

        normalized_body = search_body.strip()
        lowered = normalized_body.lower()

        if lowered.startswith("no retrieval"):
            return "no_retrieval", ""

        if lowered.startswith("image retrieval"):
            query_txt = re.sub(r"^image retrieval[:.\s-]*", "", normalized_body, flags=re.IGNORECASE).strip()
            return "image_retrieval", query_txt.strip(' "\'.')

        if lowered.startswith("text retrieval"):
            query_txt = re.sub(r"^text retrieval[:.\s-]*", "", normalized_body, flags=re.IGNORECASE).strip()
            return "text_retrieval", query_txt.strip(' "\'.')

        # Fallback: treat any other non-empty <Search> content as a bare text-retrieval query.
        bare_query = normalized_body.strip(' "\'.')
        if bare_query:
            return "text_retrieval", bare_query

        return None, ""

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
        env.pop("HTTP_PROXY", None)
        env.pop("HTTPS_PROXY", None)
        env.pop("http_proxy", None)
        env.pop("https_proxy", None)

        try:
            completed = self._run_roi_preprocess_command(
                command=command,
                env=env,
                src_path=src_path,
                crop_path=crop_path,
                meta_path=meta_path,
                image_query=image_query,
            )
            if completed and completed.stdout.strip():
                print(f"ROI preprocess stdout:\n{completed.stdout.strip()}")
            if completed and completed.stderr.strip():
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

    def warmup_roi_preprocess(self):
        roi_cfg = self.roi_preprocess_config
        if not roi_cfg or not roi_cfg.get("enabled", False):
            return False
        python_bin = roi_cfg.get("python_bin")
        script_path = roi_cfg.get("script_path")
        if not python_bin or not script_path:
            return False
        self._ensure_roi_worker()
        return self._roi_worker_process is not None

    def _build_roi_worker_env(self):
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("VIRTUAL_ENV", None)
        env.pop("ALL_PROXY", None)
        env.pop("all_proxy", None)
        env.pop("HTTP_PROXY", None)
        env.pop("HTTPS_PROXY", None)
        env.pop("http_proxy", None)
        env.pop("https_proxy", None)
        return env

    def _build_roi_worker_command(self, serve=False):
        roi_cfg = self.roi_preprocess_config or {}
        python_bin = roi_cfg.get("python_bin")
        script_path = roi_cfg.get("script_path")
        if not python_bin or not script_path:
            return None

        command = [python_bin, script_path]
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
        if serve:
            command.append("--serve")
        return command

    def _ensure_roi_worker(self):
        process = self._roi_worker_process
        if process is not None and process.poll() is None:
            return process

        command = self._build_roi_worker_command(serve=True)
        if command is None:
            return None

        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=self._build_roi_worker_env(),
        )
        ready_payload = None
        startup_lines = []
        startup_deadline = time.time() + 300

        while time.time() < startup_deadline:
            if process.poll() is not None:
                stderr = process.stderr.read().strip() if process.stderr else ""
                stdout_preview = "\n".join(startup_lines).strip()
                raise RuntimeError(
                    f"Persistent ROI worker exited during startup. stdout={stdout_preview} stderr={stderr}"
                )

            ready_line = process.stdout.readline().strip() if process.stdout else ""
            if not ready_line:
                continue

            try:
                payload = json.loads(ready_line)
            except json.JSONDecodeError:
                startup_lines.append(ready_line)
                continue

            if payload.get("status") == "ready":
                ready_payload = payload
                break

            startup_lines.append(ready_line)

        if ready_payload is None:
            stderr = process.stderr.read().strip() if process.stderr else ""
            stdout_preview = "\n".join(startup_lines).strip()
            raise RuntimeError(
                f"Persistent ROI worker failed to report ready status. stdout={stdout_preview} stderr={stderr}"
            )

        self._roi_worker_process = process
        if not self._roi_worker_exit_registered:
            atexit.register(self._shutdown_roi_worker)
            self._roi_worker_exit_registered = True
        print("Persistent GroundingDINO ROI worker is ready.")
        return process

    def _shutdown_roi_worker(self):
        process = self._roi_worker_process
        if process is None:
            return
        try:
            if process.poll() is None and process.stdin is not None:
                process.stdin.write(json.dumps({"command": "shutdown"}, ensure_ascii=False) + "\n")
                process.stdin.flush()
        except Exception:
            pass
        try:
            process.terminate()
        except Exception:
            pass
        self._roi_worker_process = None

    def _run_roi_preprocess_command(self, command, env, src_path, crop_path, meta_path, image_query):
        try:
            process = self._ensure_roi_worker()
        except Exception as exc:
            print(f"Persistent ROI worker unavailable, fallback to one-shot mode: {exc}")
            process = None

        if process is None:
            return subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )

        request = {
            "image_path": src_path,
            "phrase": image_query,
            "output_path": crop_path,
            "json_output": meta_path,
        }
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("Persistent ROI worker lost its stdio streams.")

        process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        process.stdin.flush()
        response_line = process.stdout.readline().strip()
        if not response_line:
            stderr = process.stderr.read().strip() if process.stderr else ""
            raise RuntimeError(f"Persistent ROI worker returned no response. stderr={stderr}")

        payload = json.loads(response_line)
        if payload.get("status") != "ok":
            error = payload.get("error", "unknown worker error")
            raise RuntimeError(f"Persistent ROI worker failed: {error}")

        return None

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

        final_answer_match = re.search(r"(?:<Final Answer>|Final Answer:)\s*(.*?)(?=<|$)", response, re.DOTALL)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip().replace("\n", "")
            if final_answer and not any(node.get("action") == "final_answer" for node in nodes):
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

    @staticmethod
    def _normalize_source_name(source):
        if not source:
            return None
        source = str(source).strip().lower()
        if source in {"mc-search", "mcsearch"}:
            return "mcsearch"
        if source in {"info-seek", "infoseek"}:
            return "infoseek"
        return source

    def _get_item_source(self, item):
        metadata = item.data.get("metadata", {}) or {}
        source = (
            item.data.get("source")
            or item.data.get("source_dataset")
            or metadata.get("source")
            or metadata.get("source_dataset")
            or self.dataset_name
        )
        return self._normalize_source_name(source)

    def _source_matches_target(self, item_source):
        if self.target_source is None:
            return True
        if item_source == self.target_source:
            return True
        if self.target_source == "infoseek" and item_source == "oven":
            return True
        return False
    
    def iterative_infer(self, question, id, image_id):
        img_path = os.path.join(self.data_dir, self.dataset_name, "images", f"{image_id}.jpg")
        
        if not os.path.exists(img_path):
            message = f"Image file not found: {img_path}"
            print(message)
            self._write_trajectory({
                "question": question,
                "id": id,
                "final_answer": "",
                "status": "missing_image",
                "duration_seconds": 0.0,
                "trajectory": [
                    {
                        "action": "error",
                        "content": message,
                    }
                ],
            })
            return "", []
        
        img = Image.open(img_path).convert("RGB")
        system_prompt = self.prompt.format(question=question) if "{question}" in self.prompt else self.prompt
        messages = [
            {"role": "system", "content": [
                {"type": "text", "text": system_prompt},
            ]},
            {"role": "user", "content":[
                {"type": "text", "text": f"Input Question: {question}"},
                {"type": "image", "image": img}
            ]}

        ]
        trajectory = []
        start_time = time.time()
        response = ""
        try:
            response = self._generate_text(messages)
            response = self._truncate_after_search(response)
            print(f"First Response: {response}")
            self._record_response_actions(trajectory, response)
            messages.append({'role': 'assistant', 'content': response})
        except Exception as e:
            error_text = self._format_exception(e)
            status = "api_error" if self._is_retryable_api_error(e) else "generation_error"
            print("Inference error, hidden states ignored:", error_text)
            self._write_failed_trajectory(
                question=question,
                id=id,
                response=response,
                trajectory=trajectory,
                start_time=start_time,
                status=status,
                error_text=error_text,
            )
            return response, messages
        
        conversation_num, max_turns = 0, 5
        while conversation_num < max_turns:
            if "Final Answer" in response or "<Final Answer>" in response:
                break
            retrieval_mode, query_txt = self._parse_search_action(response)
            if retrieval_mode is not None:
                retrieval_content = ""
                retrieved_docs = []
                if retrieval_mode == "text_retrieval":
                    print("Start Text Retrieval...")
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
                elif retrieval_mode == "image_retrieval":
                    print("Start Image Retrieval...")
                    print(f"Image Query: {query_txt}")
                    retrieved_docs = self._search_image_docs(img, query_txt)
                    retrieval_content = self._format_retrieval_content(retrieved_docs)
                    retrieval_content = self._clip_text(retrieval_content, self.retrieval_char_limit)
                    self._log_retrieval_preview(retrieval_content)
                elif retrieval_mode == "no_retrieval":
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
                    error_text = self._format_exception(e)
                    status = "api_error" if self._is_retryable_api_error(e) else "generation_error"
                    print("Inference error, hidden states ignored:", error_text)
                    self._write_failed_trajectory(
                        question=question,
                        id=id,
                        response=response,
                        trajectory=trajectory,
                        start_time=start_time,
                        status=status,
                        error_text=error_text,
                    )
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
        prediction_list = []
        original_count = len(dataset.data)
        selected_items = [item for item in dataset.data if self._source_matches_target(self._get_item_source(item))]
        dataset.data = selected_items

        if self.target_source is not None:
            print(
                f"Stage source filter `{self.target_source}` selected {len(selected_items)} / {original_count} items."
            )
        if not dataset.data:
            print("No items matched the configured source filter. Skip inference and evaluation.")
            return dataset

        # 2. 记录开始时间
        start_time = time.time()

        for item in tqdm(dataset.data, total=len(dataset.data)):
            answer, context = self.iterative_infer(item.question, item.id, item.image_id)
            prediction_list.append(answer)

        dataset.update_output("pred", prediction_list)
        dataset = self.evaluate(dataset, do_eval=do_eval)
        end_time = time.time()
        total_duration = end_time - start_time
        count = len(dataset.data)
        
        avg_time = total_duration / count if count > 0 else 0

        print(f"\n[Timing] Total: {total_duration:.2f}s | Count: {count} | Avg per item: {avg_time:.4f}s")
        with open(os.path.join(self.output_dir, "records.txt"), "a", encoding="utf-8") as f:
            f.write(f"\n[Timing] Total: {total_duration:.2f}s | Count: {count} | Avg per item: {avg_time:.4f}s")
        return dataset
