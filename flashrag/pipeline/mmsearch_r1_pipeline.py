import base64
import inspect
import json
import os
import pickle
import re
import time
from io import BytesIO
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from flashrag.pipeline.mm_pipeline import BasicMultiModalPipeline
from flashrag.utils import get_retriever


MMSEARCH_PROMPT_DIR = Path("/home/you/multimodal-search-r1/mmsearch_r1/prompts")


class MMSearchR1Pipeline(BasicMultiModalPipeline):
    """Dedicated MMSearch-R1 pipeline following the official inference flow."""

    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        super().__init__(config, prompt_template)
        self.config = config
        self.data_dir = self.config["data_dir"]
        self.dataset_name = self.config["dataset_name"]
        self.output_dir = self.config["output_dir"] if "output_dir" in self.config and self.config["output_dir"] else self.config["save_dir"]
        os.makedirs(self.output_dir, exist_ok=True)
        self.trajectory_path = os.path.join(self.output_dir, "mmsearch_r1_trajectories.jsonl")
        self.retrieval_char_limit = int(self.config["omni_retrieval_char_limit"])
        self.retrieved_doc_char_limit = int(self.config.get("omni_retrieved_doc_char_limit", self.retrieval_char_limit))
        self.roi_preprocess_config = self.config.get("roi_preprocess_config", {})
        self.target_source = self._normalize_source_name(self.config.get("source"))
        self.retriever = retriever if retriever is not None else get_retriever(config)
        self.generator = None
        self.round_1_prompt, self.after_image_search_prompt, self.after_text_search_prompt = self._load_official_prompts()
        self.mm_model, self.mm_processor = self._load_official_model()

    @staticmethod
    def _config_value(config, key, default=None):
        if hasattr(config, "final_config"):
            return config.final_config.get(key, default)
        if isinstance(config, dict):
            return config.get(key, default)
        value = config[key]
        return default if value is None else value

    def _load_official_prompts(self):
        with open(MMSEARCH_PROMPT_DIR / "round_1_user_prompt_qwenvl.pkl", "rb") as f:
            round_1_prompt = pickle.load(f).replace("<image>", "").strip()
        with open(MMSEARCH_PROMPT_DIR / "after_image_search_prompt_qwenvl.pkl", "rb") as f:
            after_image_search_prompt = pickle.load(f).strip()
        with open(MMSEARCH_PROMPT_DIR / "after_text_search_prompt_qwenvl.pkl", "rb") as f:
            after_text_search_prompt = pickle.load(f).strip()
        return round_1_prompt, after_image_search_prompt, after_text_search_prompt

    def _load_official_model(self):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        model_path = self.config["generator_model_path"]
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_path)
        return model, processor

    @staticmethod
    def _normalize_generation_output(response):
        if isinstance(response, str):
            return response
        if isinstance(response, list):
            flattened = []
            for item in response:
                text = MMSearchR1Pipeline._normalize_generation_output(item)
                if text:
                    flattened.append(text)
            return "\n".join(flattened).strip()
        if response is None:
            return ""
        return str(response)

    @staticmethod
    def _format_exception(exc):
        return f"{exc.__class__.__name__}: {exc}"

    def _is_retryable_api_error(self, exc):
        return False

    def _build_image_uri(self, img):
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _build_first_turn_messages(self, question, img):
        return [{
            "role": "user",
            "content": [
                {"type": "text", "text": f"{self.round_1_prompt}\nQuestion: {question}\nImage: "},
                {"type": "image", "image": self._build_image_uri(img), "max_pixels": 672 * 672},
            ],
        }]

    def _build_image_search_followup(self, question, retrieval_content):
        return (
            "Search Results: <information> [Image search results]\n"
            f"{retrieval_content}\n"
            f"</information> Original user's question: {question}\n{self.after_image_search_prompt}"
        )

    def _build_text_search_followup(self, question, retrieval_content):
        return (
            f"Search Results: <information>{retrieval_content}</information> Original question: {question}\n"
            f"{self.after_text_search_prompt}"
        )

    def _build_user_message(self, text):
        return {"role": "user", "content": [{"type": "text", "text": text}]}

    def _build_assistant_message(self, text):
        return {"role": "assistant", "content": [{"type": "text", "text": text}]}

    def _generate_text(self, messages, max_new_tokens=None):
        from qwen_vl_utils import process_vision_info

        prompt = self.mm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.mm_processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.mm_model.device)

        if max_new_tokens is None:
            max_new_tokens = int(self._config_value(self.config, "mmsearch_r1_max_new_tokens", 512))

        generated_ids = self.mm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.mm_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return self._normalize_generation_output(output_text)

    def _clip_text(self, text, limit):
        if text is None:
            return None
        text = str(text)
        if len(text) <= limit:
            return text
        return text[:limit] + "\n\n[Truncated due to prompt length limit]"

    def _format_retrieved_doc(self, doc, preferred_field=None):
        if isinstance(doc, str):
            return doc
        if isinstance(doc, dict):
            if preferred_field:
                value = doc.get(preferred_field)
                if value:
                    return str(value)
            for key in ("text", "contents", "content", "body", "passage", "title", "url"):
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

    def _search_with_main_retriever(self, query, query_kind):
        search_callable = getattr(self.retriever, "_search", self.retriever.search)
        search_signature = inspect.signature(search_callable)
        search_params = search_signature.parameters
        supports_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in search_params.values())
        kwargs = {}

        topk = self.config["image_retrieval_topk"] if query_kind == "image" else self.config["text_retrieval_topk"]
        if topk is not None and ("num" in search_params or supports_kwargs):
            kwargs["num"] = topk
        if "query_type" in search_params or supports_kwargs:
            kwargs["query_type"] = query_kind
        if query_kind == "image" and ("target_modal" in search_params or supports_kwargs):
            kwargs["target_modal"] = self.config["image_retrieval_target_modal"]

        return self.retriever.search(query, **kwargs)

    def _search_text_docs(self, query_txt):
        return self._search_with_main_retriever(query_txt, query_kind="text")

    def _run_roi_preprocess(self, img, image_query):
        roi_cfg = self.roi_preprocess_config
        if not roi_cfg or not roi_cfg.get("enabled", False):
            return img
        return img

    def _search_image_docs(self, img, image_query=""):
        retrieval_img = self._run_roi_preprocess(img, image_query)
        return self._search_with_main_retriever(retrieval_img, query_kind="image")

    def _extract_response_text(self, response):
        if not response:
            return ""
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.IGNORECASE | re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()
        answer_match = re.search(r"(?:<Final Answer>|Final Answer:|<answer>)\s*(.*?)(?=<|$)", response, re.IGNORECASE | re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip().replace("\n", "")
        return response.strip()

    def _extract_text_search_query(self, response):
        match = re.search(r"<text_search>(.*?)</text_search>", response, re.IGNORECASE | re.DOTALL)
        if not match:
            return ""
        return match.group(1).strip()

    def _extract_image_search_flag(self, response):
        return bool(re.search(r"<search><img></search>\s*$", response.strip(), re.IGNORECASE))

    def _record_response_actions(self, trajectory, response):
        trajectory.append({"action": "assistant_response", "content": response})

    def _record_retrieval_result(self, trajectory, retrieval_mode, query_txt, retrieval_content):
        trajectory.append({
            "action": "retrieval_result",
            "mode": retrieval_mode,
            "query": query_txt if query_txt else None,
            "content": retrieval_content,
        })

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

    def safe_write(self, file_path, data):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def _write_trajectory(self, record):
        self.safe_write(self.trajectory_path, self._serialize_for_log(record))

    def _write_failed_trajectory(self, *, question, id, response, trajectory, start_time, status, error_text):
        trajectory.append({"action": "error", "content": error_text})
        self._write_trajectory({
            "question": question,
            "id": id,
            "final_answer": response,
            "status": status,
            "error": error_text,
            "duration_seconds": time.time() - start_time,
            "trajectory": trajectory,
        })

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
        return item_source == self.target_source or (self.target_source == "infoseek" and item_source == "oven")

    def iterative_infer(self, question, id, image_id):
        img_path = Path(self.data_dir) / self.dataset_name / "images" / f"{image_id}.jpg"
        if not img_path.exists():
            message = f"Image file not found: {img_path}"
            print(message)
            self._write_trajectory({
                "question": question,
                "id": id,
                "final_answer": "",
                "status": "missing_image",
                "duration_seconds": 0.0,
                "trajectory": [{"action": "error", "content": message}],
            })
            return "", []

        img = Image.open(img_path).convert("RGB")
        messages = self._build_first_turn_messages(question, img)
        trajectory = []
        start_time = time.time()

        try:
            response = self._generate_text(messages)
            print(f"First Response: {response}")
            self._record_response_actions(trajectory, response)
            messages.append(self._build_assistant_message(response))
        except Exception as e:
            error_text = self._format_exception(e)
            print("Inference error, hidden states ignored:", error_text)
            self._write_failed_trajectory(
                question=question,
                id=id,
                response="",
                trajectory=trajectory,
                start_time=start_time,
                status="generation_error",
                error_text=error_text,
            )
            return "", messages

        if self._extract_image_search_flag(response):
            try:
                retrieved_docs = self._search_image_docs(img, "")
                retrieval_content = self._format_retrieval_content(retrieved_docs)
                retrieval_content = self._clip_text(retrieval_content, self.retrieval_char_limit)
                self._log_retrieval_preview(retrieval_content)
                self._record_retrieval_result(trajectory, "image_retrieval", "", retrieval_content)
                messages.append(self._build_user_message(self._build_image_search_followup(question, retrieval_content)))
                response = self._generate_text(messages)
                print(f"Second Response: {response}")
                self._record_response_actions(trajectory, response)
                messages.append(self._build_assistant_message(response))
            except Exception as e:
                error_text = self._format_exception(e)
                print("Inference error, hidden states ignored:", error_text)
                self._write_failed_trajectory(
                    question=question,
                    id=id,
                    response=response,
                    trajectory=trajectory,
                    start_time=start_time,
                    status="generation_error",
                    error_text=error_text,
                )
                return response, messages

            text_search_query = self._extract_text_search_query(response)
            if text_search_query:
                try:
                    retrieved_docs = self._search_text_docs(text_search_query)
                    retrieval_content = self._format_retrieval_content(retrieved_docs)
                    retrieval_content = self._clip_text(retrieval_content, self.retrieval_char_limit)
                    self._log_retrieval_preview(retrieval_content)
                    self._record_retrieval_result(trajectory, "text_retrieval", text_search_query, retrieval_content)
                    messages.append(self._build_user_message(self._build_text_search_followup(question, retrieval_content)))
                    response = self._generate_text(messages)
                    print(f"Third Response: {response}")
                    self._record_response_actions(trajectory, response)
                    messages.append(self._build_assistant_message(response))
                except Exception as e:
                    error_text = self._format_exception(e)
                    print("Inference error, hidden states ignored:", error_text)
                    self._write_failed_trajectory(
                        question=question,
                        id=id,
                        response=response,
                        trajectory=trajectory,
                        start_time=start_time,
                        status="generation_error",
                        error_text=error_text,
                    )
                    return response, messages

        final_answer = self._extract_response_text(response)
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

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        prediction_list = []
        original_count = len(dataset.data)
        selected_items = [item for item in dataset.data if self._source_matches_target(self._get_item_source(item))]
        dataset.data = selected_items

        if self.target_source is not None:
            print(f"Stage source filter `{self.target_source}` selected {len(selected_items)} / {original_count} items.")
        if not dataset.data:
            print("No items matched the configured source filter. Skip inference and evaluation.")
            return dataset

        start_time = time.time()
        for item in tqdm(dataset.data, total=len(dataset.data)):
            answer, _ = self.iterative_infer(item.question, item.id, item.image_id)
            prediction_list.append(answer)

        dataset.update_output("pred", prediction_list)
        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

        total_duration = time.time() - start_time
        count = len(dataset.data)
        avg_time = total_duration / count if count > 0 else 0
        print(f"\n[Timing] Total: {total_duration:.2f}s | Count: {count} | Avg per item: {avg_time:.4f}s")
        with open(os.path.join(self.output_dir, "records.txt"), "a", encoding="utf-8") as f:
            f.write(f"\n[Timing] Total: {total_duration:.2f}s | Count: {count} | Avg per item: {avg_time:.4f}s")
        return dataset
