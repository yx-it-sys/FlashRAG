import json
import os
import time
from typing import Any
import inspect

from PIL import Image
from tqdm import tqdm

from flashrag.pipeline.mm_pipeline import MMSequentialPipeline


class MMSequentialStatsPipeline(MMSequentialPipeline):
    """MMSequentialPipeline with sample-level token and latency accounting."""

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
        super().__init__(config, prompt_template=prompt_template, retriever=retriever, generator=generator)
        self.config = config
        self.data_dir = self.config["data_dir"]
        self.dataset_name = self.config["dataset_name"]
        self.output_dir = (
            self.config["output_dir"]
            if "output_dir" in self.config and self.config["output_dir"]
            else self.config["save_dir"]
        )
        os.makedirs(self.output_dir, exist_ok=True)
        self.retrieval_char_limit = int(self._config_value(self.config, "omni_retrieval_char_limit", 4000))
        self.retrieved_doc_char_limit = int(
            self._config_value(self.config, "omni_retrieved_doc_char_limit", self.retrieval_char_limit)
        )
        self.target_source = self._normalize_source_name(self._config_value(self.config, "source"))
        self._current_sample_generation_stats = None
        self._last_sample_generation_stats = None
        self._sample_generation_stats = []

    def _empty_sample_generation_summary(self, latency_seconds=0.0):
        latency_seconds = float(latency_seconds or 0.0)
        return {
            "total_samples": 1,
            "input_tokens": 0.0,
            "output_tokens": 0.0,
            "total_tokens": 0.0,
            "latency_seconds": latency_seconds,
            "avg_input_tokens_per_sample": 0.0,
            "avg_output_tokens_per_sample": 0.0,
            "avg_total_tokens_per_sample": 0.0,
            "avg_latency_seconds_per_sample": latency_seconds,
            "input_tokens_per_second": 0.0,
            "output_tokens_per_second": 0.0,
            "total_tokens_per_second": 0.0,
        }

    def _coerce_generation_totals(self, stats):
        totals = {
            "input_tokens": 0.0,
            "output_tokens": 0.0,
            "total_tokens": 0.0,
            "latency_seconds": 0.0,
        }
        if not stats:
            return totals
        if isinstance(stats, dict):
            totals["input_tokens"] = float(
                stats.get("total_input_tokens", stats.get("input_tokens", 0.0))
            )
            totals["output_tokens"] = float(
                stats.get("total_output_tokens", stats.get("output_tokens", 0.0))
            )
            totals["total_tokens"] = float(
                stats.get("total_tokens", totals["input_tokens"] + totals["output_tokens"])
            )
            totals["latency_seconds"] = float(
                stats.get("total_latency_seconds", stats.get("latency_seconds", stats.get("latency(s)", 0.0)))
            )
            return totals
        if isinstance(stats, list):
            for item in stats:
                if not isinstance(item, dict):
                    continue
                item_input_tokens = float(
                    item.get("total_input_tokens", item.get("input_tokens", 0.0))
                )
                item_output_tokens = float(
                    item.get("total_output_tokens", item.get("output_tokens", 0.0))
                )
                totals["input_tokens"] += item_input_tokens
                totals["output_tokens"] += item_output_tokens
                totals["total_tokens"] += float(
                    item.get("total_tokens", item_input_tokens + item_output_tokens)
                )
                totals["latency_seconds"] += float(
                    item.get("total_latency_seconds", item.get("latency_seconds", item.get("latency(s)", 0.0)))
                )
        return totals

    def _pop_generator_generation_totals(self):
        generator = getattr(self, "generator", None)
        if generator is None:
            return self._coerce_generation_totals(None)
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
        return self._coerce_generation_totals(stats)

    def _accumulate_current_sample_generation_totals(self, totals):
        if self._current_sample_generation_stats is None:
            return
        for key in ("input_tokens", "output_tokens", "total_tokens", "latency_seconds"):
            self._current_sample_generation_stats[key] += float(totals.get(key, 0.0))

    def _start_sample_generation_accounting(self):
        self._pop_generator_generation_totals()
        self._last_sample_generation_stats = None
        self._current_sample_generation_stats = {
            "start_time": time.time(),
            "input_tokens": 0.0,
            "output_tokens": 0.0,
            "total_tokens": 0.0,
            "latency_seconds": 0.0,
        }

    def _finish_sample_generation_accounting(self):
        if self._current_sample_generation_stats is None:
            summary = self._empty_sample_generation_summary()
            self._last_sample_generation_stats = summary
            return summary
        self._accumulate_current_sample_generation_totals(self._pop_generator_generation_totals())
        current = self._current_sample_generation_stats
        wall_clock_latency = max(0.0, time.time() - float(current.get("start_time", time.time())))
        input_tokens = float(current.get("input_tokens", 0.0))
        output_tokens = float(current.get("output_tokens", 0.0))
        total_tokens = float(current.get("total_tokens", input_tokens + output_tokens))
        summary = {
            "total_samples": 1,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "latency_seconds": wall_clock_latency,
            "avg_input_tokens_per_sample": input_tokens,
            "avg_output_tokens_per_sample": output_tokens,
            "avg_total_tokens_per_sample": total_tokens,
            "avg_latency_seconds_per_sample": wall_clock_latency,
            "input_tokens_per_second": (input_tokens / wall_clock_latency) if wall_clock_latency > 0 else 0.0,
            "output_tokens_per_second": (output_tokens / wall_clock_latency) if wall_clock_latency > 0 else 0.0,
            "total_tokens_per_second": (total_tokens / wall_clock_latency) if wall_clock_latency > 0 else 0.0,
        }
        self._current_sample_generation_stats = None
        self._last_sample_generation_stats = summary
        return summary

    def _attach_generation_stats(self, dataset):
        if self._sample_generation_stats and len(self._sample_generation_stats) == len(dataset):
            dataset.update_output("generation_stats", self._sample_generation_stats)
            return
        super()._attach_generation_stats(dataset)

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

    def _generate_text(self, prompt):
        delay = 2
        max_retries = 5
        for attempt in range(max_retries):
            try:
                raw_response = self.generator.generate([prompt])
                return self._normalize_generation_output(raw_response)
            except Exception as exc:
                if not self._is_retryable_api_error(exc):
                    raise
                if attempt == max_retries - 1:
                    raise
                print(f"Transient API error ({exc.__class__.__name__}), retrying in {delay}s ...")
                time.sleep(delay)
                delay *= 2
            finally:
                self._accumulate_current_sample_generation_totals(self._pop_generator_generation_totals())
        raw_response = self.generator.generate([prompt])
        return self._normalize_generation_output(raw_response)

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
            for key in ("text", "contents", "content", "body", "passage"):
                value = doc.get(key)
                if value:
                    return str(value)
            return json.dumps(doc, ensure_ascii=False)
        return str(doc)

    def _normalize_retrieval_doc_for_prompt(self, doc):
        """Coerce heterogeneous retriever outputs into mm_prompt-compatible dicts."""
        if isinstance(doc, str):
            text = self._clip_text(doc.strip(), self.retrieved_doc_char_limit)
            return {"contents": text, "text": text}

        if isinstance(doc, dict):
            normalized = dict(doc)
            if "image" in normalized:
                if "text" not in normalized or not str(normalized.get("text", "")).strip():
                    normalized["text"] = self._clip_text(
                        self._format_retrieved_doc(normalized),
                        self.retrieved_doc_char_limit,
                    )
                else:
                    normalized["text"] = self._clip_text(
                        str(normalized.get("text", "")).strip(),
                        self.retrieved_doc_char_limit,
                    )
                return normalized

            contents = ""
            for key in ("contents", "text", "content", "body", "passage", "title"):
                value = normalized.get(key)
                if value is not None and str(value).strip():
                    contents = self._clip_text(str(value).strip(), self.retrieved_doc_char_limit)
                    break
            normalized["contents"] = contents
            if "text" not in normalized or not str(normalized.get("text", "")).strip():
                normalized["text"] = contents
            else:
                normalized["text"] = self._clip_text(
                    str(normalized.get("text", "")).strip(),
                    self.retrieved_doc_char_limit,
                )
            return normalized

        text = self._clip_text(str(doc).strip(), self.retrieved_doc_char_limit)
        return {"contents": text, "text": text}

    def _truncate_retrieval_results_to_budget(self, docs):
        """Apply the config-level total retrieval char budget across all prompt docs."""
        remaining = max(0, int(self.retrieval_char_limit))
        if remaining <= 0:
            return []

        truncated_docs = []
        for doc in docs:
            if remaining <= 0:
                break

            normalized = dict(doc)
            if "image" in normalized:
                text_value = str(normalized.get("text", "") or "").strip()
                if text_value:
                    clipped_text = self._clip_text(text_value, remaining)
                    normalized["text"] = clipped_text
                    remaining -= len(clipped_text)
                truncated_docs.append(normalized)
                continue

            contents_value = str(normalized.get("contents", "") or "").strip()
            if not contents_value:
                continue
            clipped_contents = self._clip_text(contents_value, remaining)
            normalized["contents"] = clipped_contents
            if "text" in normalized and str(normalized.get("text", "") or "").strip():
                normalized["text"] = clipped_contents
            remaining -= len(clipped_contents)
            truncated_docs.append(normalized)

        return truncated_docs

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

    def _resolve_item_image_path(self, item):
        image_id = item.data.get("image_id")
        if not image_id:
            raise KeyError(f"Item `{item.id}` is missing `image_id`.")
        candidate_paths = []
        dataset_image_dir = self.config.get("dataset_image_dir")
        if dataset_image_dir:
            candidate_paths.append(os.path.join(str(dataset_image_dir), f"{image_id}.jpg"))
        dataset_path = self.config.get("dataset_path")
        if dataset_path:
            candidate_paths.append(os.path.join(str(dataset_path), "images", f"{image_id}.jpg"))
        candidate_paths.append(os.path.join(self.data_dir, self.dataset_name, "images", f"{image_id}.jpg"))
        candidate_paths.append(os.path.join(self.data_dir, "RefAmb", "images", f"{image_id}.jpg"))
        for image_path in candidate_paths:
            if image_path and os.path.exists(image_path):
                return image_path
        raise FileNotFoundError(
            f"Image file not found for item `{item.id}` with image_id `{image_id}`. Tried: {candidate_paths}"
        )

    def _prepare_item_image(self, item):
        image_path = self._resolve_item_image_path(item)
        with Image.open(image_path) as img:
            question_image = img.convert("RGB")
        item.data["image"] = question_image
        return question_image

    def _batch_search_single(self, query, target_modal):
        search_callable = getattr(self.retriever, "_search", self.retriever.search)
        search_signature = inspect.signature(search_callable)
        search_params = search_signature.parameters
        supports_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in search_params.values()
        )

        kwargs = {}
        effective_target_modal = target_modal
        if isinstance(query, Image.Image):
            effective_target_modal = self._config_value(
                self.config, "image_retrieval_target_modal", target_modal
            )
        topk = None
        if effective_target_modal == "image":
            topk = self._config_value(self.config, "image_retrieval_topk")
        elif effective_target_modal == "text":
            topk = self._config_value(self.config, "text_retrieval_topk")

        if topk is not None and ("num" in search_params or supports_kwargs):
            kwargs["num"] = topk
        if "target_modal" in search_params or supports_kwargs:
            kwargs["target_modal"] = effective_target_modal
        if "query_type" in search_params or supports_kwargs:
            kwargs["query_type"] = "image" if isinstance(query, Image.Image) else "text"

        return self.retriever.search(query, **kwargs)

    def infer_sample(self, item, perform_modality_dict=None):
        if perform_modality_dict is None:
            perform_modality_dict = self.PERFORM_MODALITY_DICT
        self._start_sample_generation_accounting()
        try:
            question = item.question if item.question is not None else item.data.get("text", "")
            query_image = self._prepare_item_image(item)

            retrieval_groups = []
            for modal in perform_modality_dict.get("text", []):
                retrieved_docs = self._batch_search_single(question, target_modal=modal)
                retrieval_groups.append(retrieved_docs)

            for modal in perform_modality_dict.get("image", []):
                retrieved_docs = self._batch_search_single(query_image, target_modal=modal)
                retrieval_groups.append(retrieved_docs)

            if retrieval_groups:
                merged_retrieval_result = []
                for group in retrieval_groups:
                    merged_retrieval_result.extend(group)
            else:
                merged_retrieval_result = []
            merged_retrieval_result = [
                self._normalize_retrieval_doc_for_prompt(doc) for doc in merged_retrieval_result
            ]
            merged_retrieval_result = self._truncate_retrieval_results_to_budget(merged_retrieval_result)
            item.update_output("retrieval_result", merged_retrieval_result)

            prompt = self.prompt_template.get_string(item)
            answer = self._generate_text(prompt)
            return answer, prompt, merged_retrieval_result
        finally:
            self._finish_sample_generation_accounting()

    def run(self, dataset, do_eval=True, perform_modality_dict=MMSequentialPipeline.PERFORM_MODALITY_DICT, pred_process_func=None):
        prediction_list = []
        prompt_list = []
        retrieval_result_list = []
        self._sample_generation_stats = []
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

        start_time = time.time()
        for item in tqdm(dataset.data, total=len(dataset.data)):
            answer, prompt, retrieval_result = self.infer_sample(item, perform_modality_dict=perform_modality_dict)
            prediction_list.append(answer)
            prompt_list.append(prompt)
            retrieval_result_list.append(retrieval_result)
            self._sample_generation_stats.append(
                self._last_sample_generation_stats or self._empty_sample_generation_summary()
            )

        dataset.update_output("retrieval_result", retrieval_result_list)
        dataset.update_output("prompt", prompt_list)
        dataset.update_output("pred", prediction_list)
        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_func=pred_process_func)

        total_duration = time.time() - start_time
        count = len(dataset.data)
        avg_time = total_duration / count if count > 0 else 0
        print(f"\n[Timing] Total: {total_duration:.2f}s | Count: {count} | Avg per item: {avg_time:.4f}s")
        with open(os.path.join(self.output_dir, "records.txt"), "a", encoding="utf-8") as f:
            f.write(f"\n[Timing] Total: {total_duration:.2f}s | Count: {count} | Avg per item: {avg_time:.4f}s")
        return dataset
