"""IIGCR_Pipeline pipeline for RefAmb-style multimodal retrieval-control experiments.

This module implements a controller-augmented OmniSearch pipeline that studies how
text retrieval trajectories behave under repeated-query repair, low-information-gain
repair, reflection-based recovery, ROI-assisted image retrieval, and trajectory
pruning. The main runtime entrypoint is ``IIGCR_Pipeline.iterative_infer()``, which
runs one sample through a bounded multi-turn search loop and records detailed
controller actions and retrieval traces for later analysis.

Primary outputs produced by this module:
- final answer strings returned to the caller
- per-sample trajectory records written by the parent OmniSearch pipeline
- controller-action traces written to ``controller_actions.jsonl``
- optional ROI debug images saved under the run output directory
"""

from difflib import SequenceMatcher
import json
import os
import shutil
import re
import subprocess
import tempfile
import time
import tomllib

import numpy as np
from PIL import Image

from flashrag.pipeline.omni_pipeline import OmniSearchPipeline


class TerminalPrinter:
    """Small stdout logger used to keep controller traces human-readable during runs."""
    def info(self, message):
        """Print a plain text log line.

        Args:
            message: Text payload that should be emitted immediately to stdout.
        """
        print(message)

    def module(self, module_name, payload):
        """Print a structured module log entry.

        Args:
            module_name: Logical stage name such as ``information_increment``.
            payload: JSON-serializable diagnostics attached to that stage.
        """
        self.info(f"[{module_name}] {json.dumps(payload, ensure_ascii=False)}")


class IIGCR_Pipeline(OmniSearchPipeline):
    """Controller-heavy OmniSearch variant used for trajectory ablation studies.

    The class extends ``OmniSearchPipeline`` with explicit repair / recovery logic,
    detailed trajectory logging, retrieval sanitization, and helper utilities for
    rebuilding or pruning message histories when the current search path stalls.
    """
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        """Initialize prompts, thresholds, controller switches, and logging state.

        Args:
            config: Fully materialized FlashRAG runtime config.
            prompt_template: Optional prompt template override.
            retriever: Optional prebuilt retriever instance.
            generator: Optional prebuilt generator instance.
        """
        super().__init__(config, prompt_template=prompt_template, retriever=retriever, generator=generator)

        prompt_path = self.config.get("all_prompts_path")
        if not prompt_path:
            raise ValueError(
                "IIGCR_Pipeline requires `all_prompts_path` or `omni_prompt_path` in config."
            )
        with open(prompt_path, "rb") as f:
            self.prompt_book = tomllib.load(f)

        self.rewrite_prompt = self.prompt_book["system_rewrite_similar_text_query"]
        self.ambiguity_check_prompt = self.prompt_book["system_ambiguity_check"]

        self.query_similarity_threshold = float(self.config.get("omni_query_similarity_threshold"))
        self.failure_threshold = float(self.config.get("omni_failure_threshold"))
        self.max_turns = int(self.config.get("omni_max_turns"))
        self.min_effective_retrieval_chars = int(self.config.get("omni_min_effective_retrieval_chars"))
        self.info_gain_similarity_threshold = float(
            self.config.get("omni_info_gain_similarity_threshold")
        )
        self.rebuild_module = self._as_bool(self.config.get("rebuild_module", True), default=True)
        self.missing_info_repair_module = self._as_bool(
            self.config.get("missing_info_repair_module", self.rebuild_module),
            default=self.rebuild_module,
        )
        self.reflection_recovery_module = self._as_bool(
            self.config.get("reflection_recovery_module", self.rebuild_module),
            default=self.rebuild_module,
        )
        self.pruning_module = self._as_bool(self.config.get("pruning_module", True), default=True)
        self.terminal = TerminalPrinter()
        self._controller_trace_filename = str(
            self.config.get("omni_controller_trace_filename", "controller_actions.jsonl")
        ).strip() or "controller_actions.jsonl"
        self._controller_item_stats_filename = str(
            self.config.get("omni_controller_item_stats_filename", "controller_item_stats.jsonl")
        ).strip() or "controller_item_stats.jsonl"
        self._current_item_id = None
        self._current_question = None
        self._current_item_controller_stats = None
        self.terminal.info(
            f"[AblationConfig] rebuild_module={self.rebuild_module}, "
            f"missing_info_repair_module={self.missing_info_repair_module}, "
            f"reflection_recovery_module={self.reflection_recovery_module}, "
            f"pruning_module={self.pruning_module}"
        )

    def _as_bool(self, value, default=True):
        """Normalize loosely-typed config values into booleans.

        Returns:
            Parsed boolean value, or ``default`` when the input cannot be interpreted.
        """
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return default

    def _append_controller_action(self, trajectory, controller_name, trigger_condition, **payload):
        """Append one controller decision to the in-memory trajectory and trace file.

        Args:
            trajectory: Mutable trajectory list for the current sample.
            controller_name: Logical controller identifier such as ``repair``.
            trigger_condition: Human-readable condition that fired the controller.
            **payload: Extra structured context to persist for debugging and analysis.
        """
        action = {
            "action": "controller_action",
            "content": {
                "controller": controller_name,
                "trigger_condition": trigger_condition,
                "item_id": self._current_item_id,
                "question": self._current_question,
            },
        }
        action["content"].update(self._serialize_for_log(payload))
        if trajectory is not None:
            trajectory.append(action)
        self._write_controller_action_trace(action["content"])

    def _write_controller_action_trace(self, record):
        """Persist one controller-action record to disk when an output directory exists."""
        if not getattr(self, "output_dir", None):
            return
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            log_path = os.path.join(self.output_dir, self._controller_trace_filename)
            payload = {"timestamp_ms": int(time.time() * 1000)}
            payload.update(self._serialize_for_log(record))
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as exc:
            self.terminal.info(f"Failed to write controller action trace: {exc}")

    def _init_item_controller_stats(self, item_id, question):
        """Initialize per-item controller counters for the current sample."""
        self._current_item_controller_stats = {
            "item_id": item_id,
            "question": question,
            "rebuild_messages_total_count": 0,
            "rebuild_messages_missing_info_count": 0,
            "missing_info_validation_count": 0,
            "rebuild_messages_reflection_count": 0,
        }

    def _increment_item_controller_stat(self, field, amount=1):
        """Increment one per-item controller counter when stats are active."""
        if self._current_item_controller_stats is None:
            return
        self._current_item_controller_stats[field] = (
            int(self._current_item_controller_stats.get(field, 0)) + amount
        )

    def _write_item_controller_stats(self, *, status, duration_seconds, final_answer=None):
        """Persist per-item controller counters to a JSONL file."""
        if not getattr(self, "output_dir", None):
            return
        if self._current_item_controller_stats is None:
            return
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            log_path = os.path.join(self.output_dir, self._controller_item_stats_filename)
            payload = dict(self._current_item_controller_stats)
            payload.update(
                {
                    "status": status,
                    "duration_seconds": duration_seconds,
                    "final_answer": final_answer,
                    "timestamp_ms": int(time.time() * 1000),
                }
            )
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(self._serialize_for_log(payload), ensure_ascii=False) + "\n")
        except Exception as exc:
            self.terminal.info(f"Failed to write controller item stats: {exc}")

    def _log_module_output(self, trajectory, module_name, **payload):
        """Emit a size-limited structured debug record for one logical module.

        Large text fields are truncated before logging so trajectory files remain
        readable and bounded.
        """
        truncate_limit = int(self.config.get("omni_module_log_char_limit", 1200))
        truncation_keys = {
            "retrieval_content",
            "enhanced_information",
            "flattened_text",
            "vlm_feedback",
            "removed_messages",
            "rebuilt_response",
            "previous_retrieval_content",
            "fallback_response",
        }

        def truncate_value(key, value):
            """Bound oversized values for trajectory-oriented debug logging."""
            if isinstance(value, str) and key in truncation_keys and len(value) > truncate_limit:
                return value[:truncate_limit] + "\n\n[Truncated for trajectory log]"
            if isinstance(value, list) and key in truncation_keys:
                text = json.dumps(value, ensure_ascii=False)
                if len(text) > truncate_limit:
                    return text[:truncate_limit] + "\n\n[Truncated for trajectory log]"
            return value

        payload = {key: truncate_value(key, value) for key, value in payload.items()}
        serialized_payload = self._serialize_for_log(payload)
        self.terminal.module(module_name, serialized_payload)

    def _normalize_query_text(self, query):
        """Lowercase and whitespace-normalize a query for robust comparisons."""
        if not query:
            return ""
        return re.sub(r"\s+", " ", str(query).strip().lower())

    def _describe_image_for_log(self, image_obj):
        """Convert an image-like object into a compact log-friendly descriptor."""
        if image_obj is None:
            return {"type": "none"}
        if isinstance(image_obj, Image.Image):
            return {
                "type": "pil_image",
                "size": list(image_obj.size),
                "mode": image_obj.mode,
            }
        return {"type": type(image_obj).__name__}

    def _save_roi_debug_image(self, crop_path, image_query, trajectory=None):
        """Copy a produced ROI crop into the run debug directory and log its path.

        Returns:
            Absolute path to the saved debug image.
        """
        roi_debug_dir = os.path.join(self.output_dir, "roi_debug")
        os.makedirs(roi_debug_dir, exist_ok=True)
        query_key = re.sub(r"[^a-z0-9]+", "_", self._normalize_query_text(image_query))[:80].strip("_")
        if not query_key:
            query_key = "roi"
        timestamp = int(time.time() * 1000)
        saved_path = os.path.join(roi_debug_dir, f"{timestamp}_{query_key}.png")
        shutil.copyfile(crop_path, saved_path)
        self._log_module_output(
            trajectory,
            "roi_saved_image",
            image_query=image_query,
            saved_path=saved_path,
        )
        return saved_path

    def _is_invalid_retrieved_doc(self, doc):
        """Decide whether a retrieved document is empty, null-like, or unusable."""
        if doc is None:
            return True
        if isinstance(doc, str):
            normalized = re.sub(r"\s+", " ", doc).strip().lower()
            return normalized in {"", "none", "none none"}
        if isinstance(doc, dict):
            entities = doc.get("entities")
            if isinstance(entities, list):
                for entity in entities:
                    if not isinstance(entity, dict):
                        continue
                    if str(entity.get("entity_name", "")).strip():
                        return False
                    entity_attributes = entity.get("entity_attributes", {}) or {}
                    if isinstance(entity_attributes, dict):
                        for value in entity_attributes.values():
                            if value not in (None, "", [], {}):
                                return False
            text_fields = [
                str(doc.get("title", "")).strip(),
                str(doc.get("text", "")).strip(),
                str(doc.get("contents", "")).strip(),
                str(doc.get("content", "")).strip(),
                str(doc.get("body", "")).strip(),
                str(doc.get("passage", "")).strip(),
            ]
            meaningful = [field for field in text_fields if field and field.lower() != "none"]
            return len(meaningful) == 0
        return False

    def _sanitize_retrieved_docs(self, retrieved_docs, retrieval_mode, trajectory=None):
        """Filter invalid retrieval results before later formatting or alignment logic.

        Returns:
            Sanitized document list, or a single sanitized document when the input was scalar.
        """
        if retrieved_docs is None:
            self._log_module_output(
                trajectory,
                "retrieval_sanitize",
                retrieval_mode=retrieval_mode,
                status="none_input",
            )
            return retrieved_docs

        if not isinstance(retrieved_docs, list):
            docs = [retrieved_docs]
            unwrap_single = True
        else:
            docs = list(retrieved_docs)
            unwrap_single = False

        filtered_docs = [doc for doc in docs if not self._is_invalid_retrieved_doc(doc)]
        removed_count = len(docs) - len(filtered_docs)
        if removed_count > 0:
            self._log_module_output(
                trajectory,
                "retrieval_sanitize",
                retrieval_mode=retrieval_mode,
                status="filtered_invalid_docs",
                input_count=len(docs),
                output_count=len(filtered_docs),
                removed_count=removed_count,
            )

        if unwrap_single:
            return filtered_docs[0] if filtered_docs else None
        return filtered_docs

    def _search_image_docs(self, img, image_query="", trajectory=None):
        """Run ROI preprocessing on an image query, then search with the main retriever."""
        roi_img = self._roi(image_query, trajectory=trajectory)
        query_image = roi_img or img
        self._log_image_retrieval_pipeline(
            trajectory,
            stage="search_image_docs_after_roi",
            image_query=image_query,
            query_image=query_image,
            note="main image retrieval will use this image after ROI preprocessing",
        )
        return self._search_with_main_retriever(query_image, query_kind="image")

    def _search_with_main_retriever(self, query, query_kind):
        """Dispatch either text or image retrieval through the configured main retriever."""
        self._log_module_output(
            None,
            "retriever_call",
            query_kind=query_kind,
            query_preview=query if isinstance(query, str) else None,
            image_info=self._describe_image_for_log(query) if not isinstance(query, str) else None,
            target_modal=(
                self.config["image_retrieval_target_modal"] if query_kind == "image" else "text"
            ),
        )
        return super()._search_with_main_retriever(query, query_kind)

    def _query_similarity(self, query_a, query_b):
        """Compute a normalized similarity score between two query strings.

        The method mixes cheap lexical normalization with the sentence-encoder path
        implemented by ``_get_query_similarity_encoder`` when available.
        """
        left = self._normalize_query_text(query_a)
        right = self._normalize_query_text(query_b)
        if not left or not right:
            return 0.0
        try:
            encoder = self._get_query_similarity_encoder()
            if encoder is None:
                raise ValueError("Text retriever encoder is unavailable.")

            emb = encoder.encode([left, right], batch_size=2, is_query=True)
            left_emb, right_emb = emb[0], emb[1]
            left_norm = np.linalg.norm(left_emb)
            right_norm = np.linalg.norm(right_emb)
            if left_norm == 0.0 or right_norm == 0.0:
                return 0.0
            similarity = float(np.dot(left_emb, right_emb) / (left_norm * right_norm))
            return max(-1.0, min(1.0, similarity))
        except Exception:
            return SequenceMatcher(None, left, right).ratio()

    def _get_query_similarity_encoder(self):
        """Lazily construct and cache the sentence encoder used for query similarity."""
        source_router = getattr(self, "source_text_retriever", None)
        if source_router is None:
            return None

        retriever = getattr(source_router, "retriever", None)
        if retriever is None:
            source = getattr(self, "current_source", None)
            retriever = source_router._switch_source(source)
        return getattr(retriever, "encoder", None)

    def _normalize_evidence_unit(self, text):
        """Canonicalize one evidence unit before de-duplication or semantic matching."""
        return self._normalize_query_text(text).rstrip(".")

    def _split_retrieval_content_into_units(self, retrieval_content):
        """Split formatted retrieval text into smaller evidence units for gain analysis."""
        text = str(retrieval_content or "").strip()
        if not text:
            return []

        doc_matches = list(re.finditer(r"(?m)^Doc\d+:\n", text))
        units = []
        if doc_matches:
            for idx, match in enumerate(doc_matches):
                start = match.end()
                end = doc_matches[idx + 1].start() if idx + 1 < len(doc_matches) else len(text)
                chunk = text[start:end].strip()
                if chunk:
                    units.append(chunk)
        else:
            units = [chunk.strip() for chunk in re.split(r"\n\s*\n+", text) if chunk.strip()]

        return units

    def _semantic_similarity(self, text_a, text_b):
        """Estimate semantic similarity between two evidence units."""
        left = self._normalize_query_text(text_a)
        right = self._normalize_query_text(text_b)
        if not left or not right:
            return None
        try:
            encoder = self._get_query_similarity_encoder()
            if encoder is None:
                raise ValueError("Text retriever encoder is unavailable.")
            emb = encoder.encode([left, right], batch_size=2, is_query=True)
            left_emb, right_emb = emb[0], emb[1]
            left_norm = np.linalg.norm(left_emb)
            right_norm = np.linalg.norm(right_emb)
            if left_norm == 0.0 or right_norm == 0.0:
                return None
            similarity = float(np.dot(left_emb, right_emb) / (left_norm * right_norm))
            return max(-1.0, min(1.0, similarity))
        except Exception:
            return SequenceMatcher(None, left, right).ratio()

    def _find_semantic_match(self, unit, candidates):
        """Find the first semantically equivalent evidence unit in a candidate pool.

        Returns:
            Tuple of ``(matched_candidate_or_None, similarity_score)``.
        """
        unit_key = self._normalize_evidence_unit(unit)
        best_match = None
        best_score = None
        for candidate in candidates:
            candidate_key = self._normalize_evidence_unit(candidate)
            if unit_key == candidate_key:
                return candidate, 1.0
            sim = self._semantic_similarity(unit, candidate)
            if sim is None:
                continue
            if best_score is None or sim > best_score:
                best_score = sim
                best_match = candidate
            if sim >= self.info_gain_similarity_threshold:
                return candidate, sim
        return None, best_score

    def _extract_thought_block_after_search(self, messages):
        """Extract the first assistant thought that appears after a retrieval feedback turn."""
        for message in reversed(messages):
            if message.get("role") != "assistant":
                continue
            content = message.get("content", "")
            if not isinstance(content, str):
                continue

            thought_match = re.search(
                r"<Thought>\s*(.*?)\s*(?:</Thought>|(?=<Search>|<Sub-Question>|<End>|<Final Answer>|Final Answer:|$))",
                content,
                re.DOTALL | re.IGNORECASE,
            )
            if thought_match:
                thought_text = thought_match.group(1).strip()
                if thought_text:
                    return thought_text

            if "<Search>" in content:
                search_match = re.search(r"<Search>\s*(.*?)\s*</Search>", content, re.DOTALL | re.IGNORECASE)
                if search_match:
                    search_text = search_match.group(1).strip()
                    if search_text:
                        return search_text
                return content.strip()
        return ""

    def _extract_last_missing_info_feedback(self, messages):
        """Recover the most recent feedback string useful for missing-info rewrites."""
        last_assistant_feedback = self._extract_thought_block_after_search(messages)
        if last_assistant_feedback:
            return last_assistant_feedback

        return (
            "Rewrite the text query to target the missing entity, attribute, relation, time, or location more explicitly."
        )

    def _extract_thought_content(self, content):
        """Parse and return the textual payload inside a ``<Thought>`` block."""
        if not isinstance(content, str):
            return ""
        thought_match = re.search(
            r"<Thought>\s*(.*?)\s*(?:</Thought>|(?=<Search>|<Sub-Question>|<End>|<Final Answer>|Final Answer:|$))",
            content,
            re.DOTALL | re.IGNORECASE,
        )
        if not thought_match:
            return ""
        return thought_match.group(1).strip()

    def _extract_last_thought_from_messages(self, messages, response=""):
        """Return the latest assistant thought from either the message list or a fresh response."""
        if response:
            thought_text = self._extract_thought_content(response)
            if thought_text:
                return thought_text
        for message in reversed(messages):
            if message.get("role") != "assistant":
                continue
            thought_text = self._extract_thought_content(message.get("content", ""))
            if thought_text:
                return thought_text
        return ""

    def _clean_final_answer_text(self, text):
        """Normalize a model final-answer string before persistence or evaluation."""
        text = str(text or "").strip()
        if not text:
            return ""

        final_answer_match = re.search(r"(?:<Final Answer>|Final Answer:)\s*(.*?)(?=<|$)", text, re.DOTALL)
        if final_answer_match:
            text = final_answer_match.group(1).strip()

        text = re.sub(r"</?(?:Thought|Search|Sub-Question|Final Answer)>", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _finalize_answer_with_fallback(self, question, messages, response, trajectory=None):
        """Produce a usable final answer when the current response is incomplete.

        Returns:
            Cleaned final-answer string chosen from the current response or fallback state.
        """
        search_state_markers = (
            "text retrieval:",
            "image retrieval:",
            "no retrieval",
            "<search>",
            "</search>",
            "search>",
        )
        last_thought = self._extract_last_thought_from_messages(messages, response=response)
        response_text = str(response or "")
        normalized_response = self._normalize_query_text(response_text)
        response_is_search_state = "<Search>" in response_text or (
            bool(normalized_response)
            and any(marker in normalized_response for marker in search_state_markers)
        )

        fallback_candidate = last_thought or ("" if response_is_search_state else response_text)
        fallback_answer = self._clean_final_answer_text(fallback_candidate)
        normalized_fallback = self._normalize_query_text(fallback_answer)
        if not fallback_answer or (
            normalized_fallback
            and any(marker in normalized_fallback for marker in search_state_markers)
        ):
            fallback_answer = "The available evidence is insufficient to provide a reliable final answer."

        finalization_prompt = (
            "You are finalizing an answer for a multimodal QA agent.\n"
            "Return exactly one concise final answer sentence or short paragraph.\n"
            "Do not output XML tags, <Thought>, <Search>, or planning text.\n"
            "If the evidence is insufficient, say so directly.\n\n"
            f"Question: {question}\n"
            f"Last reasoning: {last_thought or '[empty]'}\n"
            f"Last model state: {('[search state omitted]' if response_is_search_state else (self._clean_final_answer_text(response_text) or '[empty]'))}\n"
        )
        try:
            finalized = self._generate_text(
                [
                    {"role": "system", "content": [{"type": "text", "text": "You are finalizing a multimodal QA answer. Follow formatting constraints strictly."}]},
                    {"role": "user", "content": [{"type": "text", "text": finalization_prompt}]},
                ]
            )
            finalized = self._clean_final_answer_text(finalized)
            if (
                finalized
                and "<Search>" not in finalized
                and "<Thought>" not in finalized
                and not (
                    (normalized_finalized := self._normalize_query_text(finalized))
                    and any(marker in normalized_finalized for marker in search_state_markers)
                )
            ):
                self._log_module_output(
                    trajectory,
                    "finalize_answer",
                    status="generated",
                    fallback_answer=fallback_answer,
                    finalized_answer=finalized,
                )
                return finalized
        except Exception as exc:
            self._log_module_output(
                trajectory,
                "finalize_answer",
                status="generation_failed",
                fallback_answer=fallback_answer,
                error=f"{exc.__class__.__name__}: {exc}",
            )

        self._log_module_output(
            trajectory,
            "finalize_answer",
            status="fallback",
            fallback_answer=fallback_answer,
        )
        return fallback_answer

    def _build_search_response(self, retrieval_mode, query_txt=""):
        """Construct a synthetic assistant search action in the pipeline tag format."""
        if retrieval_mode == "text_retrieval":
            normalized_query = str(query_txt or "").strip()
            if not normalized_query:
                return (
                    "<Search>\n"
                    "No Retrieval.\n"
                    "</Search>"
                )
            return (
                "<Search>\n"
                f"Text Retrieval: {normalized_query}\n"
                "</Search>"
            )
        if retrieval_mode == "image_retrieval":
            normalized_query = str(query_txt or "").strip()
            if not normalized_query:
                return (
                    "<Search>\n"
                    "No Retrieval.\n"
                    "</Search>"
                )
            return (
                "<Search>\n"
                f"Image Retrieval: {normalized_query}\n"
                "</Search>"
            )
        return (
            "<Search>\n"
            "No Retrieval.\n"
            "</Search>"
        )

    def _extract_feedback_query_candidate(self, reference_query, vlm_feedback):
        """Heuristically derive a more useful text query from retrieval feedback text."""
        feedback = str(vlm_feedback or "").strip()
        if not feedback:
            return ""

        feedback = re.split(r"Latest retrieval content\s*:", feedback, maxsplit=1, flags=re.IGNORECASE)[0]
        feedback = re.sub(r"\s+", " ", feedback).strip()
        if not feedback:
            return ""

        banned_phrases = {
            "clue more explicitly",
            "missing clue",
            "the missing clue",
            "more explicitly",
            "query",
            "rewrite the query",
        }

        def normalize_candidate(text):
            """Normalize one heuristic rewrite candidate extracted from feedback text."""
            candidate = str(text or "").strip(" :,-")
            candidate = re.sub(r"^(the|a|an)\s+", "", candidate, flags=re.IGNORECASE)
            candidate = re.sub(
                r"\b(?:more explicitly|explicitly|missing clue|the missing clue|query|retrieval)\b",
                "",
                candidate,
                flags=re.IGNORECASE,
            )
            candidate = re.sub(r"\s+", " ", candidate).strip(" :,-")
            normalized = self._normalize_query_text(candidate)
            if not normalized or normalized in banned_phrases:
                return ""
            if len(normalized.split()) < 3:
                return ""
            return candidate

        patterns = [
            r"(?:need to find|need to search for|search for|look for)\s+(.+?)(?:[.?!]|$)",
            r"(?:does not provide|did not provide|missing)\s+(.+?)(?:[.?!]|$)",
            r"(?:target|focus on|specifically for)\s+(.+?)(?:[.?!]|$)",
        ]
        candidates = []
        for pattern in patterns:
            for match in re.finditer(pattern, feedback, re.IGNORECASE):
                candidate = normalize_candidate(match.group(1))
                if candidate:
                    candidates.append(candidate)

        for candidate in candidates:
            if self._query_similarity(candidate, reference_query) < self.query_similarity_threshold:
                return candidate
            combined_candidate = f"{candidate} exact value".strip()
            if self._query_similarity(combined_candidate, reference_query) < self.query_similarity_threshold:
                return combined_candidate
        return ""

    def _tokenize_query_terms(self, query):
        """Split a query into normalized lexical terms for specificity checks."""
        return re.findall(r"[a-z0-9]+", self._normalize_query_text(query))

    def _is_specificity_increasing_rewrite(self, reference_query, candidate_query):
        """Check whether a candidate rewrite is meaningfully more specific than the original."""
        previous = str(reference_query or "").strip()
        candidate = str(candidate_query or "").strip()
        if not candidate:
            return False
        if not previous:
            return True

        normalized_candidate = self._normalize_query_text(candidate)
        generic_filler_patterns = [
            r"\bmore specific information\b",
            r"\bspecific details\b",
            r"\bmore details\b",
            r"\bmore precise data\b",
            r"\bexact value\b",
            r"\bto answer the question\b",
        ]
        if any(re.search(pattern, normalized_candidate) for pattern in generic_filler_patterns):
            candidate_wo_filler = normalized_candidate
            for pattern in generic_filler_patterns:
                candidate_wo_filler = re.sub(pattern, " ", candidate_wo_filler)
            if candidate_wo_filler.strip() == self._normalize_query_text(previous):
                return False

        previous_terms = set(self._tokenize_query_terms(previous))
        candidate_terms = set(self._tokenize_query_terms(candidate))
        if not candidate_terms:
            return False

        new_terms = candidate_terms - previous_terms
        non_specific_terms = {
            "a", "an", "the", "this", "that", "these", "those", "more", "specific", "information",
            "details", "detail", "exact", "value", "question", "answer", "about", "regarding",
            "what", "which", "who", "where", "when", "how", "find", "identify", "determine",
        }
        informative_new_terms = [term for term in new_terms if term not in non_specific_terms]
        return len(informative_new_terms) > 0

    def _build_missing_info_rewrite_message(self, query_to_rewrite, vlm_feedback):
        """Build the rewrite prompt used when retrieval indicates missing information."""
        prompt_template = self.rewrite_prompt
        system_prompt = prompt_template
        user_prompt = (
            "Apply the system instructions to this input.\n"
            f"query_to_rewrite: {query_to_rewrite or ''}\n"
            f"vlm_feedback: {vlm_feedback or ''}"
        )
        rewrite_response = self._generate_text([
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ])
        rewrite_response = self._truncate_after_search(rewrite_response)
        if "<Search>" not in rewrite_response:
            return self._build_search_response("text_retrieval", query_to_rewrite)
        search_body_match = re.search(r"<Search>\s*(.*?)\s*</Search>", rewrite_response, re.DOTALL | re.IGNORECASE)
        if not search_body_match:
            return self._build_search_response("text_retrieval", query_to_rewrite)
        search_body = search_body_match.group(1).strip()
        valid_search = (
            "Text Retrieval:" in search_body
            or search_body.lower().startswith("image retrieval:")
            or search_body == "No Retrieval."
            or search_body == "No Retrieval"
        )
        if not valid_search:
            return self._build_search_response("text_retrieval", query_to_rewrite)
        return rewrite_response

    def _force_diversified_rewrite(self, reference_query, vlm_feedback, trajectory=None):
        """Force a non-duplicate next move when ordinary rewrites stay too close.

        Returns:
            Tuple of ``(diagnosis_label, rewritten_search_response)``.
        """
        diversification_prompt = (
            "You are rewriting a failed text retrieval query.\n"
            "Return exactly one <Search> block.\n"
            "Prefer Text Retrieval with a genuinely different query that targets a new missing clue, "
            "entity, relation, time, location, or attribute.\n"
            "Do not repeat or lightly paraphrase the previous query.\n"
            "If the missing clue is visual and cannot be expressed well in text, return Image Retrieval: <anchored phrase>.\n"
            "Use No Retrieval only if no justified retrieval step remains.\n\n"
            f"Reference query: {reference_query or ''}\n"
            f"Feedback: {vlm_feedback or ''}\n"
        )
        try:
            rewrite_response = self._generate_text(
                [
                    {"role": "system", "content": [{"type": "text", "text": "You are rewriting a failed text retrieval query. Follow the required output format strictly."}]},
                    {"role": "user", "content": [{"type": "text", "text": diversification_prompt}]},
                ]
            )
            rewrite_response = self._truncate_after_search(rewrite_response)
        except Exception as exc:
            self._log_module_output(
                trajectory,
                "diversified_query_rewrite",
                status="generation_error",
                reference_query=reference_query,
                vlm_feedback=vlm_feedback,
                error=f"{exc.__class__.__name__}: {exc}",
            )
            rewrite_response = ""

        retrieval_mode, query_txt = self._extract_search_mode_and_query(rewrite_response)
        if retrieval_mode == "text_retrieval":
            similarity = self._query_similarity(query_txt, reference_query) if reference_query else 0.0
            if (
                query_txt
                and similarity < self.query_similarity_threshold
                and self._is_specificity_increasing_rewrite(reference_query, query_txt)
            ):
                normalized_response = self._build_search_response("text_retrieval", query_txt)
                self._log_module_output(
                    trajectory,
                    "diversified_query_rewrite",
                    status="accepted_model_rewrite",
                    reference_query=reference_query,
                    vlm_feedback=vlm_feedback,
                    rewritten_query=query_txt,
                    similarity=similarity,
                )
                return "QUERY_DIVERSIFIED", normalized_response
            self._log_module_output(
                trajectory,
                "diversified_query_rewrite",
                status="rejected_model_rewrite_not_more_specific",
                reference_query=reference_query,
                vlm_feedback=vlm_feedback,
                rewritten_query=query_txt,
                similarity=similarity,
            )

        heuristic_query = self._extract_feedback_query_candidate(reference_query, vlm_feedback)
        if heuristic_query:
            similarity = self._query_similarity(heuristic_query, reference_query) if reference_query else 0.0
            if (
                similarity < self.query_similarity_threshold
                and self._is_specificity_increasing_rewrite(reference_query, heuristic_query)
            ):
                normalized_response = self._build_search_response("text_retrieval", heuristic_query)
                self._log_module_output(
                    trajectory,
                    "diversified_query_rewrite",
                    status="accepted_feedback_fallback",
                    reference_query=reference_query,
                    vlm_feedback=vlm_feedback,
                    rewritten_query=heuristic_query,
                    similarity=similarity,
                )
                return "QUERY_DIVERSIFIED", normalized_response
            self._log_module_output(
                trajectory,
                "diversified_query_rewrite",
                status="rejected_feedback_rewrite_not_more_specific",
                reference_query=reference_query,
                vlm_feedback=vlm_feedback,
                rewritten_query=heuristic_query,
                similarity=similarity,
            )

        if self._ambiguity_check(reference_query, trajectory=trajectory):
            self._log_module_output(
                trajectory,
                "diversified_query_rewrite",
                status="fallback_image_retrieval",
                reference_query=reference_query,
                vlm_feedback=vlm_feedback,
                rewritten_query=None,
            )
            return "UNRESOLVED_VISUAL_REFERENCE", self._build_search_response("image_retrieval", reference_query)

        fallback_query = reference_query or self._extract_feedback_query_candidate("", vlm_feedback)
        if fallback_query:
            if "exact value" not in self._normalize_query_text(fallback_query):
                fallback_query = f"{fallback_query} exact value"
            self._log_module_output(
                trajectory,
                "diversified_query_rewrite",
                status="fallback_previous_query",
                reference_query=reference_query,
                vlm_feedback=vlm_feedback,
                rewritten_query=fallback_query,
            )
            return "QUERY_REWRITE_FALLBACK", self._build_search_response("text_retrieval", fallback_query)

        self._log_module_output(
            trajectory,
            "diversified_query_rewrite",
            status="fallback_image_retrieval_last_resort",
            reference_query=reference_query,
            vlm_feedback=vlm_feedback,
            rewritten_query=None,
        )
        return "UNRESOLVED_VISUAL_REFERENCE", self._build_search_response("image_retrieval", reference_query)

    def _infer_rebuild_diagnosis(self, reference_query, rebuilt_response, vlm_feedback="", trajectory=None):
        """Classify the rebuild result and normalize it into a valid search action."""
        self._increment_item_controller_stat("missing_info_validation_count")
        retrieval_mode, query_txt = self._extract_search_mode_and_query(rebuilt_response)
        if retrieval_mode == "image_retrieval":
            return "UNRESOLVED_VISUAL_REFERENCE", self._build_search_response("image_retrieval", query_txt)
        if retrieval_mode == "no_retrieval":
            return "RETRIEVAL_PATH_EXHAUSTED", self._build_search_response("no_retrieval")

        if retrieval_mode != "text_retrieval":
            return "RETRIEVAL_PATH_EXHAUSTED", self._build_search_response("no_retrieval")

        similarity = self._query_similarity(query_txt, reference_query) if reference_query else 0.0
        if reference_query and similarity > self.query_similarity_threshold:
            return self._force_diversified_rewrite(
                reference_query,
                vlm_feedback,
                trajectory=trajectory,
            )

        return "QUERY_MISSING_CLUE", rebuilt_response

    def _extract_search_mode_and_query(self, response):
        """Parse one tagged assistant response into retrieval mode and query text."""
        if not response:
            return None, ""
        search_match = re.search(r"<Search>\s*(.*?)\s*</Search>", response, re.DOTALL | re.IGNORECASE)
        if not search_match:
            return None, ""
        search_body = search_match.group(1).strip()
        lowered = search_body.lower()
        if lowered.startswith("text retrieval"):
            return "text_retrieval", self._extract_retrieval_query(response, "Text Retrieval")
        if lowered.startswith("image retrieval"):
            return "image_retrieval", self._extract_retrieval_query(response, "Image Retrieval")
        if lowered.startswith("no retrieval"):
            return "no_retrieval", ""
        return None, ""

    def _extract_text_retrieval_query_from_search(self, response):
        """Convenience wrapper that extracts only the text-retrieval query from a response."""
        if not response:
            return ""
        return self._extract_retrieval_query(response, "Text Retrieval")

    def _force_text_retrieval_after_image_reflection(
        self,
        *,
        reference_query,
        response,
        vlm_feedback="",
        trajectory=None,
    ):
        """Normalize an image-reflection output into a guaranteed text-retrieval action."""
        retrieval_mode, query_txt = self._extract_search_mode_and_query(response)
        if retrieval_mode == "text_retrieval" and query_txt:
            return response

        candidate_query = ""
        if query_txt:
            candidate_query = query_txt
        if not candidate_query:
            candidate_query = self._extract_feedback_query_candidate(reference_query, vlm_feedback)
        if not candidate_query:
            candidate_query = reference_query
        candidate_query = str(candidate_query or "").strip()
        if not candidate_query:
            candidate_query = "identify the missing visual clue in text"

        forced_response = self._build_search_response("text_retrieval", candidate_query)
        self._log_module_output(
            trajectory,
            "force_text_retrieval_after_image_reflection",
            reference_query=reference_query,
            original_response=response,
            original_mode=retrieval_mode,
            forced_query=candidate_query,
            forced_response=forced_response,
        )
        return forced_response

    def _is_retrieval_feedback_message(self, message):
        """Detect user messages that simply feed retrieval results back to the model."""
        if message.get("role") != "user":
            return False
        content = message.get("content", [])
        if not isinstance(content, list):
            return False
        text_parts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
        text = "\n".join(text_parts).strip()
        if not text:
            return False
        return (
            text.startswith("Contents of retrieved documents:")
            or text.startswith("Retrieval was attempted but returned no useful results.")
            or text.startswith("You chose `No Retrieval`")
        )

    def _should_remove_search_state(self, message, bad_query="", bad_query_reason=""):
        """Decide whether one assistant search state should be pruned from history.

        Returns:
            Tuple of ``(should_remove, reason_string)``.
        """
        if message.get("role") != "assistant":
            return False, ""
        content = message.get("content", "")
        if not isinstance(content, str) or "<Search>" not in content:
            return False, ""

        retrieval_mode, query_txt = self._extract_search_mode_and_query(content)
        if retrieval_mode != "text_retrieval":
            return False, ""

        normalized_query = self._normalize_query_text(query_txt)
        if (
            bad_query_reason == "duplicate_prev_query"
            and bad_query
            and normalized_query == self._normalize_query_text(bad_query)
        ):
            return True, "duplicate_prev_query"
        if (
            bad_query_reason == "delta_f_zero"
            and bad_query
            and normalized_query == self._normalize_query_text(bad_query)
        ):
            return True, "delta_f_zero"
        return False, ""

    def _extract_search_segment(self, content):
        """Extract only the ``<Search>...</Search>`` segment from a mixed assistant message."""
        if not isinstance(content, str):
            return ""
        start = content.find("<Search>")
        if start == -1:
            return ""
        end = content.find("</Search>", start)
        if end == -1:
            return content[start:].strip()
        end += len("</Search>")
        return content[start:end].strip()

    def _remove_search_action_from_message(self, message):
        """Drop the search action from a mixed reasoning/search message but keep reasoning text.

        Returns:
            A rewritten message dict, or ``None`` when nothing meaningful remains.
        """
        content = message.get("content", "")
        if not isinstance(content, str) or "<Search>" not in content:
            return None

        start = content.find("<Search>")
        end = content.find("</Search>", start)
        if end == -1:
            end = len(content)
        else:
            end += len("</Search>")

        prefix = content[:start].rstrip()
        suffix = content[end:].lstrip()
        kept_parts = [part for part in (prefix, suffix) if part.strip()]
        if not kept_parts:
            return None
        return {**message, "content": "\n".join(kept_parts).strip()}

    def _apply_missing_info_rebuild(self, original_message, rebuilt_response):
        """Merge a rebuilt search action back into the preserved reasoning prefix."""
        preserved_message = self._remove_search_action_from_message(original_message)
        rebuilt_search = self._extract_search_segment(rebuilt_response)
        if preserved_message is None or not rebuilt_search:
            return rebuilt_response
        preserved_content = preserved_message.get("content", "")
        if not isinstance(preserved_content, str) or not preserved_content.strip():
            return rebuilt_response
        return f"{preserved_content}\n{rebuilt_search}".strip()

    def _context_prune_messages(
        self,
        messages,
        bad_query="",
        bad_query_reason="",
        replacement_query="",
        trajectory=None,
    ):
        """Prune failed retrieval context with stack-based search-state replacement.

        Returns:
            Tuple of ``(pruned_messages, removed_messages, removed_turn_count)``.
        """
        if not self.pruning_module:
            self._log_module_output(
                trajectory,
                "context_pruning",
                bad_query=bad_query,
                bad_query_reason=bad_query_reason,
                replacement_query=replacement_query,
                removed_count=0,
                removed_messages=[],
                removed_turns=0,
                skipped_state_pruning=True,
                reason="pruning_module_disabled",
            )
            return messages, [], 0

        if not bad_query:
            self._log_module_output(
                trajectory,
                "context_pruning",
                bad_query=bad_query,
                bad_query_reason=bad_query_reason,
                replacement_query=replacement_query,
                removed_count=0,
                removed_messages=[],
                removed_turns=0,
            )
            return messages, [], 0

        context_stack = []
        removed_messages = []
        removed_turns = 0
        preserved_anchor = False
        repaired_search_message = None
        if replacement_query:
            repaired_search_message = {
                "role": "assistant",
                "content": self._build_search_response("text_retrieval", replacement_query),
            }

        i = 0
        while i < len(messages):
            message = messages[i]
            context_stack.append(message)

            should_remove = False
            reason = ""
            if message.get("role") == "assistant":
                content = message.get("content", "")
                if isinstance(content, str) and "<Search>" in content:
                    should_remove, reason = self._should_remove_search_state(
                        message,
                        bad_query=bad_query,
                        bad_query_reason=bad_query_reason,
                    )

            if should_remove:
                if not preserved_anchor:
                    preserved_anchor = True
                    i += 1
                    continue

                bad_message = context_stack.pop()
                removed_messages.append({"reason": reason, "message": bad_message})
                removed_turns += 1

                if i + 1 < len(messages) and self._is_retrieval_feedback_message(messages[i + 1]):
                    removed_messages.append(messages[i + 1])
                    i += 1

                if i + 1 < len(messages):
                    next_message = messages[i + 1]
                    if next_message.get("role") == "assistant":
                        next_content = next_message.get("content", "")
                        if isinstance(next_content, str) and "<Thought>" in next_content:
                            removed_messages.append({"reason": "post_search_thought", "message": next_message})
                            i += 1

                if repaired_search_message is not None:
                    context_stack.append(dict(repaired_search_message))

            i += 1

        self._log_module_output(
            trajectory,
            "context_pruning",
            bad_query=bad_query,
            bad_query_reason=bad_query_reason,
            replacement_query=replacement_query,
            preserved_anchor=preserved_anchor,
            removed_count=len(removed_messages),
            removed_messages=removed_messages,
            removed_turns=removed_turns,
        )
        return context_stack, removed_messages, removed_turns

    def _rebuild_messages(
        self,
        messages,
        mode,
        reference_query="",
        vlm_feedback="",
        bad_query="",
        bad_query_reason="",
        replacement_query="",
        reflection_kind="text_retrieval",
        trajectory=None,
    ):
        """Rebuild message history for either missing-info repair or reflection recovery.

        Returns:
            Tuple of ``(rebuilt_messages, rebuilt_response_string)``.
        """
        self._increment_item_controller_stat("rebuild_messages_total_count")
        if mode == "missing_info":
            self._increment_item_controller_stat("rebuild_messages_missing_info_count")
            raw_rebuilt_response = self._build_missing_info_rewrite_message(reference_query, vlm_feedback)
            diagnosis, rebuilt_response = self._infer_rebuild_diagnosis(
                reference_query,
                raw_rebuilt_response,
                vlm_feedback=vlm_feedback,
                trajectory=trajectory,
            )
            self._log_module_output(
                trajectory,
                "rebuild_messages",
                mode=mode,
                reference_query=reference_query,
                vlm_feedback=vlm_feedback,
                diagnosis=diagnosis,
                raw_rebuilt_response=raw_rebuilt_response,
                rebuilt_response=rebuilt_response,
            )
            rebuilt_messages = list(messages)
            for idx in range(len(rebuilt_messages) - 1, -1, -1):
                if rebuilt_messages[idx].get("role") != "assistant":
                    continue
                content = rebuilt_messages[idx].get("content", "")
                if isinstance(content, str) and "<Search>" in content:
                    rebuilt_messages = rebuilt_messages[: idx + 1]
                    rebuilt_messages[idx] = {
                        "role": "assistant",
                        "content": self._apply_missing_info_rebuild(rebuilt_messages[idx], rebuilt_response),
                    }
                    break
            else:
                rebuilt_messages.append({"role": "assistant", "content": rebuilt_response})
            return rebuilt_messages, rebuilt_response

        self._increment_item_controller_stat("rebuild_messages_reflection_count")
        trimmed_messages, removed_messages, removed_turns = self._context_prune_messages(
            messages,
            bad_query=bad_query,
            bad_query_reason=bad_query_reason,
            replacement_query=replacement_query,
            trajectory=trajectory,
        )
        failed_thought = self._extract_last_thought_from_messages(removed_messages) or self._extract_last_thought_from_messages(messages)
        if reflection_kind == "image_retrieval":
            reflection_response = (
                "<ReflectionInstruction>\n"
                "Repeated image retrieval was detected. The current image-retrieval path is not working and the current retrieval results are not sufficient to move forward. "
                f"Failed image query: {reference_query}.\n"
                f"Previous thought before the failed retrieval: {failed_thought or '[empty]'}\n"
                "Return exactly one <Thought> block first, then return exactly one <Search> block or one <Final Answer> block. "
                "Do not omit the required tags. Do not output more than one action block after the thought block. "
                f"You must not issue the same or near-duplicate Image Retrieval query as: {reference_query}. "
                "Choose a genuinely different next step. Prefer Text Retrieval if the missing clue can be described in text. "
                "If you choose Image Retrieval again, you must use a clearly different anchored phrase that focuses on a different visual clue, object, attribute, region, or accessory. "
                "If the existing evidence is already sufficient, give a Final Answer. Do not repeat the same failed image-retrieval plan.\n"
                "</ReflectionInstruction>"
            )
        else:
            reflection_response = (
                "<ReflectionInstruction>\n"
                "Repeated or low-gain text retrieval was detected. The recent repeated retrieval turns have been discarded. "
                "The current query path is not working and the current retrieval results are not sufficient to move forward. "
                f"Failed query: {reference_query}.\n"
                f"Previous thought before the failed retrieval: {failed_thought or '[empty]'}\n"
                "Return exactly one <Thought> block first, then return exactly one <Search> block or one <Final Answer> block. "
                "Do not omit the required tags. Do not output more than one action block after the thought block. "
                f"You must not ask the same question again, and you must not issue the same or near-duplicate Text Retrieval query as: {reference_query}. "
                "Choose a genuinely different next step. If you choose Text Retrieval, ask a different question that targets a new missing clue, entity, relation, location, time, or attribute instead of rephrasing the same request. "
                "If a different text query is not well justified, choose Image Retrieval: <anchored phrase>, No Retrieval, or give a Final Answer if the evidence is already sufficient. "
                "Do not repeat the same failed plan.\n"
                "</ReflectionInstruction>"
            )
        self._log_module_output(
            trajectory,
            "rebuild_messages",
            mode=mode,
            reflection_kind=reflection_kind,
            reference_query=reference_query,
            vlm_feedback=vlm_feedback,
            rebuilt_response=reflection_response,
            failed_thought=failed_thought,
            removed_turns=removed_turns,
            removed_messages=removed_messages,
        )
        rebuilt_messages = trimmed_messages
        rebuilt_messages.append({"role": "user", "content": [{"type": "text", "text": reflection_response}]})
        return rebuilt_messages, reflection_response

    def _ambiguity_check(self, query_txt, trajectory=None):
        """Ask the ambiguity detector prompt whether a text query is visually ambiguous."""
        query_txt = str(query_txt or "").strip()
        if not query_txt:
            self._log_module_output(trajectory, "ambiguity_check", query=query_txt, status="empty_query")
            return False

        prompt = self.ambiguity_check_prompt.format(
            text_query=query_txt,
        )
        try:
            response = self._generate_text(
                [
                    {"role": "system", "content": [{"type": "text", "text": "You are an ambiguity detector. Return strict JSON as required."}]},
                    {"role": "user", "content": [{"type": "text", "text": prompt}]},
                ]
            )
        except Exception as exc:
            self.terminal.info(f"Ambiguity check generation failed, default to non-ambiguous: {exc}")
            self._log_module_output(
                trajectory,
                "ambiguity_check",
                query=query_txt,
                status="generation_failed",
                error=str(exc),
            )
            return False

        try:
            payload = json.loads(str(response or "").strip())
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", str(response or ""), re.DOTALL)
            if not json_match:
                self.terminal.info("Ambiguity check returned invalid JSON; default to non-ambiguous.")
                self._log_module_output(
                    trajectory,
                    "ambiguity_check",
                    query=query_txt,
                    status="json_parse_failed",
                )
                return False
            try:
                payload = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                self.terminal.info("Ambiguity check regex JSON parse failed; default to non-ambiguous.")
                self._log_module_output(
                    trajectory,
                    "ambiguity_check",
                    query=query_txt,
                    status="json_parse_failed",
                )
                return False

        entity_ambiguous = str(payload.get("entity_ambiguous", "")).strip()
        remark = str(payload.get("remark", "") or "").strip()
        is_ambiguous = entity_ambiguous == "Yes"
        self._log_module_output(
            trajectory,
            "ambiguity_check",
            query=query_txt,
            entity_ambiguous=entity_ambiguous,
            remark=remark,
        )
        return is_ambiguous

    def _extract_image_retrieval_query(self, response):
        """Extract the image-retrieval query phrase from a tagged assistant response."""
        return self._extract_retrieval_query(response, "Image Retrieval")

    def _roi(self, image_query, trajectory=None):
        """Run ROI cropping for the current image and return the cropped PIL image if available."""
        roi_cfg = self.roi_preprocess_config
        current_image = getattr(self, "_current_image", None)
        self._log_module_output(
            trajectory,
            "roi_enter",
            image_query=image_query,
            roi_enabled=bool(roi_cfg and roi_cfg.get("enabled", False)),
            original_image=self._describe_image_for_log(current_image),
        )
        if not roi_cfg or not roi_cfg.get("enabled", False):
            self._log_module_output(
                trajectory,
                "roi",
                image_query=image_query,
                status="disabled",
            )
            return getattr(self, "_current_image", None)
        if not image_query:
            self._log_module_output(
                trajectory,
                "roi",
                image_query=image_query,
                status="empty_query",
            )
            return getattr(self, "_current_image", None)

        img = getattr(self, "_current_image", None)
        if img is None:
            self.terminal.info("ROI preprocessing skipped: current image is unavailable.")
            self._log_module_output(
                trajectory,
                "roi",
                image_query=image_query,
                status="missing_image",
            )
            return None

        python_bin = roi_cfg.get("python_bin")
        script_path = roi_cfg.get("script_path")
        if not python_bin or not script_path:
            self.terminal.info("ROI preprocessing skipped: missing python_bin or script_path.")
            self._log_module_output(
                trajectory,
                "roi",
                image_query=image_query,
                status="missing_config",
                python_bin=python_bin,
                script_path=script_path,
            )
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

        command = self._build_roi_worker_command(serve=False)
        if command is None:
            self._log_module_output(
                trajectory,
                "roi",
                image_query=image_query,
                status="missing_command",
            )
            return img
        command.extend([
            "--image-path", src_path,
            "--phrase", image_query,
            "--output-path", crop_path,
            "--json-output", meta_path,
        ])

        env = self._build_roi_worker_env()

        try:
            self._log_module_output(
                trajectory,
                "roi_subprocess_start",
                image_query=image_query,
                src_path=src_path,
                output_path=crop_path,
                meta_path=meta_path,
                command=command,
            )
            completed = self._run_roi_preprocess_command(
                command=command,
                env=env,
                src_path=src_path,
                crop_path=crop_path,
                meta_path=meta_path,
                image_query=image_query,
            )
            if completed and completed.stdout.strip():
                self.terminal.info(f"ROI preprocess stdout:\n{completed.stdout.strip()}")
            if completed and completed.stderr.strip():
                self.terminal.info(f"ROI preprocess stderr:\n{completed.stderr.strip()}")

            if not os.path.exists(crop_path):
                self._log_module_output(
                    trajectory,
                    "roi",
                    image_query=image_query,
                    status="no_crop_output",
                )
                return img

            metadata = None
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                self.terminal.info(f"ROI metadata: {metadata}")

            with Image.open(crop_path) as cropped:
                saved_path = self._save_roi_debug_image(crop_path, image_query, trajectory=trajectory)
                self._log_module_output(
                    trajectory,
                    "roi",
                    image_query=image_query,
                    status="ok",
                    metadata=metadata,
                    crop_path=crop_path,
                    saved_path=saved_path,
                    image_size=list(cropped.size),
                )
                return cropped.convert("RGB")
        except subprocess.CalledProcessError as exc:
            stdout = exc.stdout.strip() if exc.stdout else ""
            stderr = exc.stderr.strip() if exc.stderr else ""
            if stdout:
                self.terminal.info(f"ROI preprocess stdout before failure:\n{stdout}")
            if stderr:
                self.terminal.info(f"ROI preprocess stderr before failure:\n{stderr}")
            self.terminal.info(f"ROI preprocessing failed, fallback to original image: {exc}")
            self._log_module_output(
                trajectory,
                "roi",
                image_query=image_query,
                status="subprocess_failed",
                error=str(exc),
                stdout=stdout,
                stderr=stderr,
            )
            return img
        except Exception as exc:
            self.terminal.info(f"ROI preprocessing failed, fallback to original image: {exc}")
            self._log_module_output(
                trajectory,
                "roi",
                image_query=image_query,
                status="failed",
                error=str(exc),
            )
            return img

    def _log_image_retrieval_pipeline(
        self,
        trajectory,
        stage,
        image_query="",
        query_image=None,
        retrieved_docs=None,
        note="",
    ):
        """Emit a structured trace record for one stage of the image-retrieval pipeline."""
        payload = {
            "stage": stage,
            "image_query": image_query,
            "query_image": self._describe_image_for_log(query_image),
            "note": note,
        }
        if isinstance(retrieved_docs, list):
            payload["retrieved_doc_count"] = len(retrieved_docs)
            payload["retrieved_doc_preview"] = self._serialize_for_log(retrieved_docs[:2])
        elif retrieved_docs is not None:
            payload["retrieved_doc_preview"] = self._serialize_for_log(retrieved_docs)
        self._log_module_output(trajectory, "image_retrieval_pipeline", **payload)
        if retrieved_docs is not None:
            self.terminal.info(
                f"[image_retrieval_raw_result:{stage}] {json.dumps(self._serialize_for_log(retrieved_docs), ensure_ascii=False)}"
            )

    def _flatten(self, image_retrieval_docs, trajectory=None):
        """Flatten retrieved multimodal evidence into a text block for alignment or prompts."""
        if not image_retrieval_docs:
            flattened_text = ""
            status = "empty_docs"
            doc_count = 0
        else:
            flattened_body = self._format_image_retrieval_content(image_retrieval_docs)
            flattened_text = (
                "According to the retrieval results for your question, the entity-related information in the image is:\n"
                f"{flattened_body}"
                if flattened_body
                else ""
            )
            status = "ok"
            doc_count = len(image_retrieval_docs) if isinstance(image_retrieval_docs, list) else 1

        self._log_module_output(
            None,
            "flatten",
            status=status,
            doc_count=doc_count,
            flattened_text=flattened_text,
        )
        return flattened_text

    def _coerce_alignment_list(self, value):
        """Normalize alignment-tool outputs into a simple list representation."""
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        items = []
        for item in value:
            item = str(item or "").strip()
            if item:
                items.append(item)
        return items

    def _normalize_alignment_confidence(self, value):
        """Normalize alignment confidence values into a bounded numeric score."""
        value = self._normalize_query_text(value)
        if value in {"high", "medium", "low"}:
            return value
        return ""

    def _normalize_alignment_status(self, value):
        """Normalize alignment status labels into a stable lowercase vocabulary."""
        value = self._normalize_query_text(value)
        if value in {"resolved", "low_confidence", "unresolved"}:
            return value
        return ""

    def _build_structured_alignment_query(
        self,
        original_query,
        canonical_entity="",
        alternative_entities=None,
        entity_status="",
        entity_confidence="",
        target_attribute="",
        refined_queries=None,
        grounding_cues=None,
    ):
        """Assemble a rewritten query from structured alignment outputs."""
        refined_queries = self._coerce_alignment_list(refined_queries)
        grounding_cues = self._coerce_alignment_list(grounding_cues)
        alternative_entities = self._coerce_alignment_list(alternative_entities)
        canonical_entity = str(canonical_entity or "").strip()
        target_attribute = str(target_attribute or "").strip()
        entity_status = self._normalize_alignment_status(entity_status)
        entity_confidence = self._normalize_alignment_confidence(entity_confidence)

        allow_entity_commit = entity_status == "resolved" and entity_confidence in {"high", "medium"}
        allow_descriptive_refine = entity_status in {"resolved", "low_confidence", "unresolved"}

        for refined_query in refined_queries:
            if not self._is_specificity_increasing_rewrite(original_query, refined_query):
                continue
            normalized_refined = self._normalize_query_text(refined_query)
            normalized_entity = self._normalize_query_text(canonical_entity)
            if allow_entity_commit:
                return refined_query
            if allow_descriptive_refine and normalized_entity and normalized_entity not in normalized_refined:
                return refined_query

        if allow_entity_commit and canonical_entity and target_attribute:
            return f"{canonical_entity} {target_attribute}".strip()

        if allow_entity_commit and canonical_entity:
            original_terms = self._normalize_query_text(original_query)
            query_without_visual_refs = re.sub(
                r"\b(?:this|these|it|its|the image|in the image|pictured|shown|depicted)\b",
                " ",
                original_terms,
                flags=re.IGNORECASE,
            )
            query_without_visual_refs = re.sub(r"\s+", " ", query_without_visual_refs).strip()
            if query_without_visual_refs:
                return f"{canonical_entity} {query_without_visual_refs}".strip()
            return canonical_entity

        if entity_status == "low_confidence" and alternative_entities:
            descriptive_query = " OR ".join(alternative_entities[:2])
            if target_attribute:
                descriptive_query = f"{descriptive_query} {target_attribute}".strip()
            if self._is_specificity_increasing_rewrite(original_query, descriptive_query):
                return descriptive_query

        if grounding_cues:
            best_cue = grounding_cues[0]
            if self._is_specificity_increasing_rewrite(original_query, best_cue):
                return best_cue

        for refined_query in refined_queries:
            if refined_query:
                return refined_query

        return ""

    def _estimate_information_gain(
        self,
        query_txt,
        reference_query,
        retrieval_content,
        cumulative_retrieval_units,
    ):
        """Measure how much novel evidence the latest retrieval added.

        Returns:
            Tuple of ``(information_gain_ratio, accepted_units_for_history)``.
        """
        current_text = self._normalize_query_text(retrieval_content)
        if not current_text or len(current_text) < self.min_effective_retrieval_chars:
            return 0.0, []

        if reference_query:
            query_similarity = self._query_similarity(query_txt, reference_query)
            if query_similarity > self.query_similarity_threshold:
                return 0.0, []

        current_units = self._split_retrieval_content_into_units(retrieval_content)
        if not current_units:
            return 0.0, []

        accepted_units = []
        accepted_unit_keys = set()
        novel_units = []

        for unit in current_units:
            unit_key = self._normalize_evidence_unit(unit)
            if not unit_key or unit_key in accepted_unit_keys:
                continue

            within_step_match, _ = self._find_semantic_match(unit, accepted_units)
            if within_step_match is not None:
                accepted_unit_keys.add(unit_key)
                continue

            history_match, _ = self._find_semantic_match(unit, cumulative_retrieval_units)
            accepted_units.append(unit)
            accepted_unit_keys.add(unit_key)
            if history_match is None:
                novel_units.append(unit)

        if not accepted_units:
            return 0.0, []

        return len(novel_units) / len(accepted_units), accepted_units

    def _run_reflection_step(self, messages, trajectory):
        """Run one reflection-generation step and append the response to message history."""
        try:
            response = self._generate_text(messages)
            response = self._truncate_after_search(response)
            self.terminal.info(f"Response after reflection: {response}")
            self._record_response_actions(trajectory, response)
            messages.append({"role": "assistant", "content": response})
            return messages, response
        except Exception as exc:
            error_text = f"{exc.__class__.__name__}: {exc}"
            fallback_response = ""
            for message in reversed(messages):
                if message.get("role") != "assistant":
                    continue
                content = message.get("content", "")
                if isinstance(content, str) and content.strip():
                    fallback_response = content
                    break
            self.terminal.info(f"Reflection step failed, fallback to previous assistant response: {error_text}")
            self._log_module_output(
                trajectory,
                "reflection_fallback",
                status="generation_failed",
                error=error_text,
                fallback_response=fallback_response,
            )
            return messages, fallback_response

    def _extract_controller_query_after(self, response):
        """Extract the actual next retrieval query from a real assistant response.

        Only queries inside a valid ``<Search>`` block are returned. Reflection
        instructions or other prompt text should therefore never leak into
        ``controller_actions.jsonl`` as ``query_after``.
        """
        action_mode, query_txt = self._parse_search_action(response)
        if action_mode in {"text_retrieval", "image_retrieval"}:
            return query_txt
        return None

    def _extract_text_parts_from_message(self, message):
        """Return flattened text parts from one message content payload."""
        content = message.get("content", "")
        if isinstance(content, str):
            return [content]
        if not isinstance(content, list):
            return []

        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
        return text_parts

    def _extract_retrieval_feedback_payload(self, message):
        """Parse one user followup message back into a retrieval-result trajectory node."""
        if not self._is_retrieval_feedback_message(message):
            return None

        text = "\n".join(self._extract_text_parts_from_message(message)).strip()
        if text.startswith("You chose `No Retrieval`"):
            return {"action": "no_retrieval_result", "mode": "no_retrieval", "content": None}

        if text.startswith("Based on the retrieval you requested, the retrieved evidence is:\n"):
            evidence = text.split("Based on the retrieval you requested, the retrieved evidence is:\n", 1)[1].strip()
            return {"action": "retrieval_result", "mode": None, "content": evidence}

        if text.startswith("Retrieval was attempted but returned no useful results."):
            return {"action": "retrieval_result", "mode": None, "content": None}

        if text.startswith("Contents of retrieved documents:"):
            evidence = text.split("Contents of retrieved documents:", 1)[1].strip()
            return {"action": "retrieval_result", "mode": None, "content": evidence}

        return None

    def _build_effective_trajectory_from_messages(self, messages):
        """Reconstruct an evaluator-friendly trajectory from the current pruned messages."""
        effective_trajectory = []
        pending_search = None

        for message in messages:
            role = message.get("role")
            if role == "assistant":
                content = message.get("content", "")
                if not isinstance(content, str) or not content.strip():
                    continue
                nodes = self._extract_action_nodes(content)
                for node in nodes:
                    action = node.get("action")
                    if action not in {"thought", "sub-question", "search", "final_answer"}:
                        continue
                    effective_trajectory.append(node)
                    if action == "search":
                        search_mode, search_query = self._extract_search_mode_and_query(
                            f"<Search>\n{node.get('content', '')}\n</Search>"
                        )
                        if search_mode == "text_retrieval":
                            pending_search = {
                                "action": "text_retrieval_result",
                                "mode": "text_retrieval",
                                "query": search_query or None,
                            }
                        elif search_mode == "image_retrieval":
                            pending_search = {
                                "action": "image_retrieval_result",
                                "mode": "image_retrieval",
                                "query": search_query or None,
                            }
                        elif search_mode == "no_retrieval":
                            pending_search = {
                                "action": "no_retrieval_result",
                                "mode": "no_retrieval",
                                "query": None,
                            }
                        else:
                            pending_search = None
                continue

            if role == "user" and pending_search is not None:
                feedback_payload = self._extract_retrieval_feedback_payload(message)
                if feedback_payload is None:
                    continue
                result_action = pending_search["action"]
                result_mode = pending_search["mode"]
                result_query = pending_search.get("query")
                if feedback_payload["action"] == "no_retrieval_result":
                    result_action = "no_retrieval_result"
                    result_mode = "no_retrieval"
                    result_query = None
                effective_trajectory.append(
                    {
                        "action": result_action,
                        "mode": result_mode,
                        "query": result_query,
                        "content": feedback_payload.get("content"),
                    }
                )
                pending_search = None

        return effective_trajectory

    def _build_simple_trajectory_from_messages(self, messages):
        """Convert assistant-tagged messages into a simplified action trajectory list."""
        simple_trajectory = []
        allowed_actions = {"thought", "sub-question", "search", "final_answer"}
        for message in messages:
            if message.get("role") != "assistant":
                continue
            content = message.get("content", "")
            if not isinstance(content, str) or not content.strip():
                continue
            nodes = self._extract_action_nodes(content)
            for node in nodes:
                action = node.get("action")
                if action in allowed_actions:
                    simple_trajectory.append(node)
        return simple_trajectory

    def _build_effective_trajectory_from_runtime(self, trajectory):
        """Filter the runtime trajectory down to the canonical public action schema.

        The runtime list may contain controller/module debug nodes. For persisted
        `omnisearch_trajectories.jsonl`, keep only the same user-facing steps as the
        reference format: reasoning/search nodes plus explicit retrieval-result nodes.
        """
        if not trajectory:
            return []

        allowed_actions = {
            "thought",
            "sub-question",
            "search",
            "text_retrieval_result",
            "image_retrieval_result",
            "no_retrieval_result",
            "final_answer",
        }
        filtered_trajectory = []
        for step in trajectory:
            if not isinstance(step, dict):
                continue
            action = step.get("action")
            if action not in allowed_actions:
                continue
            filtered_trajectory.append(dict(step))
        return filtered_trajectory

    def _build_effective_trajectory(self, messages, fallback_trajectory=None):
        """Choose the best available evaluator-friendly trajectory representation."""
        effective_from_runtime = self._build_effective_trajectory_from_runtime(fallback_trajectory)
        effective_from_messages = self._build_effective_trajectory_from_messages(messages)

        runtime_retrieval_count = sum(
            1
            for step in effective_from_runtime
            if step.get("action") in {"text_retrieval_result", "image_retrieval_result", "no_retrieval_result"}
        )
        message_retrieval_count = sum(
            1
            for step in effective_from_messages
            if step.get("action") in {"text_retrieval_result", "image_retrieval_result", "no_retrieval_result"}
        )

        if effective_from_runtime and runtime_retrieval_count > message_retrieval_count:
            return effective_from_runtime
        if effective_from_messages:
            return effective_from_messages
        if effective_from_runtime:
            return effective_from_runtime
        return self._build_simple_trajectory_from_messages(messages)

    def iterative_infer(self, question, id, image_id):
        """Run the full bounded controller loop for one sample image/question pair.

        Args:
            question: User question for the current sample.
            id: Sample identifier used in trajectory logs.
            image_id: Image filename stem used to load the associated image.

        Returns:
            Final answer string for the sample, or ``None`` when the image is missing.
        """
        img_path = os.path.join(self.data_dir, self.dataset_name, "images", f"{image_id}.jpg")
        if not os.path.exists(img_path):
            return None

        img = Image.open(img_path).convert("RGB")
        self._current_image = img
        self._current_item_id = id
        self._current_question = question
        self._init_item_controller_stats(id, question)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Input Question: {question}"},
                    {"type": "image", "image": img},
                ],
            },
        ]

        trajectory = []
        public_trajectory = []
        last_public_response_span = None

        def append_public_response_actions(response_text):
            nonlocal last_public_response_span
            nodes = self._extract_action_nodes(response_text)
            start = len(public_trajectory)
            for node in nodes:
                if node.get("action") in {"thought", "sub-question", "search", "final_answer"}:
                    public_trajectory.append(dict(node))
            last_public_response_span = (start, len(public_trajectory))

        def replace_last_public_response_actions(response_text):
            nonlocal last_public_response_span
            if last_public_response_span is None:
                append_public_response_actions(response_text)
                return
            start, end = last_public_response_span
            replacement = []
            for node in self._extract_action_nodes(response_text):
                if node.get("action") in {"thought", "sub-question", "search", "final_answer"}:
                    replacement.append(dict(node))
            public_trajectory[start:end] = replacement
            last_public_response_span = (start, start + len(replacement))

        def append_public_retrieval_result(retrieval_mode, query_txt, retrieval_content):
            mode_name_map = {
                "text_retrieval": "text_retrieval_result",
                "image_retrieval": "image_retrieval_result",
                "no_retrieval": "no_retrieval_result",
            }
            public_trajectory.append(
                {
                    "action": mode_name_map.get(retrieval_mode, "retrieval_result"),
                    "mode": retrieval_mode,
                    "query": query_txt if query_txt else None,
                    "content": retrieval_content,
                }
            )

        def sync_public_trajectory_from_messages(current_messages):
            nonlocal last_public_response_span
            public_trajectory[:] = self._build_effective_trajectory_from_messages(current_messages)
            last_public_response_span = None

        start_time = time.time()
        accepted_query = {"text": None, "source": "original"}
        attempted_query = {"text": None, "source": "original"}
        draft_query_source = "original"
        cumulative_retrieval_units = []
        conversation_num = 0
        similar_query_streak = 0
        low_gain_streak = 0
        consecutive_image_retrieval_count = 0
        last_retrieval_mode = None
        last_image_query = None
        response = self._generate_text(messages)
        response = self._truncate_after_search(response)
        self.terminal.info(f"First Response: {response}")
        self._record_response_actions(trajectory, response)
        append_public_response_actions(response)
        messages.append({"role": "assistant", "content": response})
        while conversation_num < self.max_turns:
            if "Final Answer" in response or "<Final Answer>" in response:
                break

            need_txt_ret = "Text Retrieval" in response
            need_img_ret = "Image Retrieval" in response
            need_no_ret = "No Retrieval" in response

            if need_txt_ret:
                retrieval_mode = "text_retrieval"
                self.terminal.info("Start Text Retrieval...")
                draft_query = self._extract_retrieval_query(response, "Text Retrieval")
                self.terminal.info(f"Query Text: {draft_query}")

                similarity_anchor = attempted_query if attempted_query["text"] else accepted_query

                if similarity_anchor["text"] is not None:
                    similarity = self._query_similarity(draft_query, similarity_anchor["text"])
                    self.terminal.info(f"Query similarity with previous query: {similarity:.4f}")
                    if similarity > self.query_similarity_threshold:
                        exact_repeat_after_repair = (
                            similarity_anchor["source"] == "repaired"
                            and self._normalize_query_text(draft_query) == self._normalize_query_text(similarity_anchor["text"])
                        )
                        if similarity_anchor["source"] != "repaired":
                            similar_query_streak += 1
                        self._log_module_output(
                            trajectory,
                            "query_similarity_gate",
                            query=draft_query,
                            reference_query=similarity_anchor["text"],
                            similarity=similarity,
                            streak=similar_query_streak,
                            similarity_anchor_source=similarity_anchor["source"],
                            similarity_anchor_type=("attempted" if attempted_query["text"] else "accepted"),
                            exact_repeat_after_repair=exact_repeat_after_repair,
                        )
                        if self.reflection_recovery_module and similarity_anchor["source"] == "repaired" and exact_repeat_after_repair:
                            bad_query = draft_query
                            messages, rebuilt_response = self._rebuild_messages(
                                messages,
                                mode="reflection",
                                reference_query=bad_query,
                                vlm_feedback=self._extract_last_missing_info_feedback(messages),
                                bad_query=bad_query,
                                bad_query_reason="duplicate_prev_query",
                                trajectory=trajectory,
                            )
                            similar_query_streak = 0
                            self.terminal.info("Repaired query was repeated exactly; switched to reflection.")
                            sync_public_trajectory_from_messages(messages)
                            messages, response = self._run_reflection_step(messages, trajectory)
                            append_public_response_actions(response)
                            self._append_controller_action(
                                trajectory,
                                controller_name="recovery",
                                trigger_condition="repaired_query_exact_repeat",
                                mode="reflection",
                                reference_query=similarity_anchor["text"],
                                query_before=bad_query,
                                query_after=self._extract_controller_query_after(response),
                                reflection_response=response,
                                similar_query_streak=similar_query_streak,
                            )
                            attempted_query = {"text": None, "source": "original"}
                            draft_query_source = "original"
                            conversation_num += 1
                            continue
                        if self.missing_info_repair_module:
                            vlm_feedback = self._extract_last_missing_info_feedback(messages)
                            original_query = draft_query
                            messages, rebuilt_response = self._rebuild_messages(
                                messages,
                                mode="missing_info",
                                reference_query=draft_query,
                                vlm_feedback=vlm_feedback,
                                trajectory=trajectory,
                            )
                            rewritten_query = self._extract_text_retrieval_query_from_search(rebuilt_response)
                            self._append_controller_action(
                                trajectory,
                                controller_name="repair",
                                trigger_condition="similar_query_streak>=1_missing_info",
                                mode="missing_info",
                                reference_query=draft_query,
                                query_before=original_query,
                                query_after=rewritten_query,
                                similar_query_streak=similar_query_streak,
                            )
                            response = rebuilt_response
                            self._record_response_actions(trajectory, response)
                            replace_last_public_response_actions(response)
                            attempted_query = {"text": rewritten_query or None, "source": "repaired"}
                            draft_query_source = "repaired"
                            self.terminal.info("Similar query similarity exceeded threshold; rebuilt messages with missing-info rewrite.")
                            conversation_num += 1
                            continue
                    else:
                        similar_query_streak = 0

                accepted_draft_query = draft_query

                retrieved_docs = self._search_text_docs(accepted_draft_query)
                retrieval_content = self._format_retrieval_content(retrieved_docs)
                retrieval_content = self._clip_text(retrieval_content, self.retrieval_char_limit)
                self._log_retrieval_preview(retrieval_content)
                self._record_retrieval_result(
                    trajectory=trajectory,
                    retrieval_mode=retrieval_mode,
                    query_txt=accepted_draft_query,
                    retrieval_content=retrieval_content,
                )
                append_public_retrieval_result(
                    retrieval_mode=retrieval_mode,
                    query_txt=accepted_draft_query,
                    retrieval_content=retrieval_content,
                )

                info_gain, accepted_units = self._estimate_information_gain(
                    accepted_draft_query,
                    accepted_query["text"],
                    retrieval_content,
                    cumulative_retrieval_units,
                )
                self.terminal.info(f"Retrieval information increment: {info_gain:.4f}")
                self._log_module_output(
                    trajectory,
                    "information_increment",
                    query=accepted_draft_query,
                    reference_query=accepted_query["text"],
                    information_increment=info_gain,
                )
                if info_gain < self.failure_threshold:
                    low_gain_streak += 1
                    self._log_module_output(
                        trajectory,
                        "low_gain_gate",
                        query=accepted_draft_query,
                        reference_query=accepted_query["text"],
                        information_increment=info_gain,
                        streak=low_gain_streak,
                        accepted_query_source=draft_query_source,
                    )
                    if self.reflection_recovery_module and draft_query_source == "repaired":
                        bad_query = accepted_draft_query
                        messages, rebuilt_response = self._rebuild_messages(
                            messages,
                            mode="reflection",
                            reference_query=bad_query,
                            vlm_feedback=retrieval_content,
                            bad_query=bad_query,
                            bad_query_reason="delta_f_zero" if info_gain == 0.0 else "",
                            trajectory=trajectory,
                        )
                        low_gain_streak = 0
                        self.terminal.info("Repaired query still produced low gain; switched to reflection.")
                        sync_public_trajectory_from_messages(messages)
                        messages, response = self._run_reflection_step(messages, trajectory)
                        append_public_response_actions(response)
                        self._append_controller_action(
                            trajectory,
                            controller_name="recovery",
                            trigger_condition="repaired_query_low_gain",
                            mode="reflection",
                            reference_query=accepted_query["text"],
                            query_before=bad_query,
                            query_after=self._extract_controller_query_after(response),
                            reflection_response=response,
                            low_gain_streak=low_gain_streak,
                            information_increment=info_gain,
                            failure_threshold=self.failure_threshold,
                            rebuilt_as_search=bool(rebuilt_response.startswith("<Search>")),
                        )
                        attempted_query = {"text": None, "source": "original"}
                        draft_query_source = "original"
                        conversation_num += 1
                        continue
                    if self.reflection_recovery_module and low_gain_streak >= 2:
                        bad_query = accepted_draft_query
                        messages, rebuilt_response = self._rebuild_messages(
                            messages,
                            mode="reflection",
                            reference_query=bad_query,
                            vlm_feedback=retrieval_content,
                            bad_query=bad_query,
                            bad_query_reason="delta_f_zero" if info_gain == 0.0 else "",
                            trajectory=trajectory,
                        )
                        low_gain_streak = 0
                        self.terminal.info("Low-gain streak reached 2; switched to reflection.")
                        sync_public_trajectory_from_messages(messages)
                        messages, response = self._run_reflection_step(messages, trajectory)
                        append_public_response_actions(response)
                        self._append_controller_action(
                            trajectory,
                            controller_name="recovery",
                            trigger_condition="low_gain_streak>=2",
                            mode="reflection",
                            reference_query=accepted_query["text"],
                            query_before=bad_query,
                            query_after=self._extract_controller_query_after(response),
                            reflection_response=response,
                            low_gain_streak=low_gain_streak,
                            information_increment=info_gain,
                            failure_threshold=self.failure_threshold,
                            rebuilt_as_search=bool(rebuilt_response.startswith("<Search>")),
                        )
                        attempted_query = {"text": None, "source": "original"}
                        draft_query_source = "original"
                        conversation_num += 1
                        continue
                    if self.missing_info_repair_module:
                        original_query = accepted_draft_query
                        vlm_feedback = (
                            "The latest retrieval brought little or no new information compared with the previous retrieval. "
                            "Rewrite the query to target the missing clue more explicitly.\n"
                            f"Latest retrieval content:\n{retrieval_content}"
                        )
                        messages, rebuilt_response = self._rebuild_messages(
                            messages,
                            mode="missing_info",
                            reference_query=accepted_draft_query,
                            vlm_feedback=vlm_feedback,
                            trajectory=trajectory,
                        )
                        rewritten_query = self._extract_text_retrieval_query_from_search(rebuilt_response)
                        self._append_controller_action(
                            trajectory,
                            controller_name="repair",
                            trigger_condition="low_gain_streak>=1_missing_info",
                            mode="missing_info",
                            reference_query=accepted_query["text"],
                            query_before=original_query,
                            query_after=rewritten_query,
                            low_gain_streak=low_gain_streak,
                            information_increment=info_gain,
                            failure_threshold=self.failure_threshold,
                        )
                        response = rebuilt_response
                        self._record_response_actions(trajectory, response)
                        replace_last_public_response_actions(response)
                        attempted_query = {"text": rewritten_query or None, "source": "repaired"}
                        draft_query_source = "repaired"
                        self.terminal.info("Low-gain signal exceeded threshold; rebuilt messages with missing-info rewrite.")
                        conversation_num += 1
                        continue
                else:
                    low_gain_streak = 0

                accepted_query = {
                    "text": accepted_draft_query,
                    "source": draft_query_source,
                }
                attempted_query = dict(accepted_query)
                cumulative_retrieval_units.extend(accepted_units)
                similar_query_streak = 0
                consecutive_image_retrieval_count = 0
                last_retrieval_mode = retrieval_mode
                last_image_query = None
                followup_message = {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self._build_followup_message(retrieval_mode, retrieval_content),
                        }
                    ],
                }
                messages.append(followup_message)

                try:
                    response = self._generate_text(messages)
                    response = self._truncate_after_search(response)
                    self.terminal.info(f"Response: {response}")
                    self._record_response_actions(trajectory, response)
                    append_public_response_actions(response)
                    messages.append({"role": "assistant", "content": response})
                    attempted_query = dict(accepted_query) if accepted_query["text"] else {"text": None, "source": "original"}
                    draft_query_source = "original"
                except Exception as e:
                    error_text = f"{e.__class__.__name__}: {e}"
                    self.terminal.info(f"Inference error, hidden states ignored: {error_text}")
                    trajectory.append({"action": "error", "content": error_text})
                    record = {
                        "question": question,
                        "id": id,
                        "final_answer": response,
                        "status": "generation_error",
                        "error": error_text,
                        "duration_seconds": time.time() - start_time,
                        "trajectory": public_trajectory,
                    }
                    self._write_trajectory(record)
                    self._write_item_controller_stats(
                        status="generation_error",
                        duration_seconds=time.time() - start_time,
                        final_answer=response,
                    )
                    return response, messages

            elif need_img_ret or need_no_ret:
                retrieval_content = ""
                retrieval_mode = None
                query_txt = ""

                if need_img_ret:
                    retrieval_mode = "image_retrieval"
                    self.terminal.info("Start Image Retrieval...")
                    query_txt = self._extract_retrieval_query(response, "Image Retrieval")
                    self.terminal.info(f"Image Query: {query_txt}")
                    image_query_similarity = None
                    if last_retrieval_mode == "image_retrieval":
                        consecutive_image_retrieval_count += 1
                        if last_image_query:
                            image_query_similarity = self._query_similarity(query_txt, last_image_query)
                            self.terminal.info(
                                f"Image query similarity with previous image query: {image_query_similarity:.4f}"
                            )
                        self._log_module_output(
                            trajectory,
                            "image_query_similarity_gate",
                            query=query_txt,
                            reference_query=last_image_query,
                            similarity=image_query_similarity,
                            consecutive_image_retrieval_count=consecutive_image_retrieval_count,
                            trigger_condition="consecutive_image_retrieval>=1",
                        )
                        if self.reflection_recovery_module and consecutive_image_retrieval_count >= 1:
                            bad_query = query_txt
                            image_reflection_feedback = self._extract_last_missing_info_feedback(messages)
                            messages, rebuilt_response = self._rebuild_messages(
                                messages,
                                mode="reflection",
                                reference_query=bad_query,
                                vlm_feedback=image_reflection_feedback,
                                bad_query=bad_query,
                                reflection_kind="image_retrieval",
                                trajectory=trajectory,
                            )
                            self.terminal.info(
                                "Consecutive image retrieval detected; rebuilt messages and switched to reflection."
                            )
                            sync_public_trajectory_from_messages(messages)
                            messages, response = self._run_reflection_step(messages, trajectory)
                            response = self._force_text_retrieval_after_image_reflection(
                                reference_query=bad_query,
                                response=response,
                                vlm_feedback=image_reflection_feedback,
                                trajectory=trajectory,
                            )
                            if messages and messages[-1].get("role") == "assistant":
                                messages[-1] = {"role": "assistant", "content": response}
                            self._append_controller_action(
                                trajectory,
                                controller_name="recovery",
                                trigger_condition="consecutive_image_retrieval>=1",
                                mode="reflection",
                                reference_query=last_image_query,
                                query_before=bad_query,
                                query_after=self._extract_controller_query_after(response),
                                reflection_response=response,
                                consecutive_image_retrieval_count=consecutive_image_retrieval_count,
                                query_similarity=image_query_similarity,
                            )
                            self._record_response_actions(trajectory, response)
                            append_public_response_actions(response)
                            attempted_query = {"text": None, "source": "original"}
                            draft_query_source = "original"
                            conversation_num += 1
                            continue
                    else:
                        consecutive_image_retrieval_count = 0

                    self._log_image_retrieval_pipeline(
                        trajectory,
                        stage="main_image_retrieval_before_search",
                        image_query=query_txt,
                        query_image=img,
                        note="about to run main image retrieval",
                    )
                    retrieved_docs = self._search_image_docs(img, query_txt, trajectory=trajectory)
                    self._log_image_retrieval_pipeline(
                        trajectory,
                        stage="main_image_retrieval_raw_result",
                        image_query=query_txt,
                        query_image=None,
                        retrieved_docs=retrieved_docs,
                    )
                    retrieved_docs = self._sanitize_retrieved_docs(
                        retrieved_docs,
                        retrieval_mode=retrieval_mode,
                        trajectory=trajectory,
                    )
                    retrieval_content = self._format_retrieval_content(retrieved_docs)
                    retrieval_content = self._clip_text(retrieval_content, self.retrieval_char_limit)
                    self._log_retrieval_preview(retrieval_content)
                    last_retrieval_mode = retrieval_mode
                    last_image_query = query_txt
                else:
                    retrieval_mode = "no_retrieval"
                    retrieval_content = None
                    consecutive_image_retrieval_count = 0
                    last_retrieval_mode = retrieval_mode
                    last_image_query = None

                self._record_retrieval_result(
                    trajectory=trajectory,
                    retrieval_mode=retrieval_mode,
                    query_txt=query_txt,
                    retrieval_content=retrieval_content,
                )
                append_public_retrieval_result(
                    retrieval_mode=retrieval_mode,
                    query_txt=query_txt,
                    retrieval_content=retrieval_content,
                )
                followup_message = {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self._build_followup_message(retrieval_mode, retrieval_content),
                        }
                    ],
                }
                messages.append(followup_message)
                try:
                    response = self._generate_text(messages)
                    response = self._truncate_after_search(response)
                    self.terminal.info(f"Response: {response}")
                    self._record_response_actions(trajectory, response)
                    append_public_response_actions(response)
                    messages.append({"role": "assistant", "content": response})
                    draft_query_source = "original"
                except Exception as e:
                    error_text = f"{e.__class__.__name__}: {e}"
                    self.terminal.info(f"Inference error, hidden states ignored: {error_text}")
                    trajectory.append({"action": "error", "content": error_text})
                    record = {
                        "question": question,
                        "id": id,
                        "final_answer": response,
                        "status": "generation_error",
                        "error": error_text,
                        "duration_seconds": time.time() - start_time,
                        "trajectory": public_trajectory,
                    }
                    self._write_trajectory(record)
                    self._write_item_controller_stats(
                        status="generation_error",
                        duration_seconds=time.time() - start_time,
                        final_answer=response,
                    )
                    return response, messages
            else:
                break

            conversation_num += 1

        pattern = r"(?:<Final Answer>|Final Answer:)\s*(.*?)(?=<|$)"
        final_answer_match = re.search(pattern, response, re.DOTALL)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip().replace("\n", "")
            self.terminal.info(f"Final Answer: {final_answer}")
            record = {
                "question": question,
                "id": id,
                "final_answer": final_answer,
                "status": "ok",
                "duration_seconds": time.time() - start_time,
                "trajectory": public_trajectory,
            }
            self._write_trajectory(record)
            self._write_item_controller_stats(
                status="ok",
                duration_seconds=time.time() - start_time,
                final_answer=final_answer,
            )
            return final_answer, messages

        finalized_answer = self._finalize_answer_with_fallback(
            question,
            messages,
            response,
            trajectory=trajectory,
        )
        self.terminal.info(
            f"Warning: reached end of agent loop for item {conversation_num} without a 'Final Answer'. returning last response"
        )
        record = {
            "question": question,
            "id": id,
            "final_answer": finalized_answer,
            "status": "missing_final_answer",
            "duration_seconds": time.time() - start_time,
            "trajectory": public_trajectory,
        }
        self._write_trajectory(record)
        self._write_item_controller_stats(
            status="missing_final_answer",
            duration_seconds=time.time() - start_time,
            final_answer=finalized_answer,
        )
        return finalized_answer, messages

    def _is_invalid_retrieved_doc(self, doc):
        """Decide whether a retrieved document is empty, null-like, or unusable."""
        if doc is None:
            return True
        if isinstance(doc, str):
            normalized = re.sub(r"\s+", " ", doc).strip().lower()
            return normalized in {"", "none", "none none"}
        if isinstance(doc, dict):
            entities = doc.get("entities")
            if isinstance(entities, list):
                for entity in entities:
                    if not isinstance(entity, dict):
                        continue
                    if str(entity.get("entity_name", "")).strip():
                        return False
                    entity_attributes = entity.get("entity_attributes", {}) or {}
                    if isinstance(entity_attributes, dict):
                        for value in entity_attributes.values():
                            if value not in (None, "", [], {}):
                                return False
            text_fields = [
                str(doc.get("title", "")).strip(),
                str(doc.get("text", "")).strip(),
                str(doc.get("contents", "")).strip(),
                str(doc.get("content", "")).strip(),
                str(doc.get("body", "")).strip(),
                str(doc.get("passage", "")).strip(),
            ]
            meaningful = [field for field in text_fields if field and field.lower() != "none"]
            return len(meaningful) == 0
        return False

    def _sanitize_retrieved_docs(self, retrieved_docs, retrieval_mode, trajectory=None):
        """Filter invalid retrieval results before later formatting or alignment logic.

        Returns:
            Sanitized document list, or a single sanitized document when the input was scalar.
        """
        if retrieved_docs is None:
            self._log_module_output(
                trajectory,
                "retrieval_sanitize",
                retrieval_mode=retrieval_mode,
                status="none_input",
            )
            return retrieved_docs

        if not isinstance(retrieved_docs, list):
            docs = [retrieved_docs]
            unwrap_single = True
        else:
            docs = list(retrieved_docs)
            unwrap_single = False

        filtered_docs = [doc for doc in docs if not self._is_invalid_retrieved_doc(doc)]
        removed_count = len(docs) - len(filtered_docs)
        if removed_count > 0:
            self._log_module_output(
                trajectory,
                "retrieval_sanitize",
                retrieval_mode=retrieval_mode,
                status="filtered_invalid_docs",
                input_count=len(docs),
                output_count=len(filtered_docs),
                removed_count=removed_count,
            )

        if unwrap_single:
            return filtered_docs[0] if filtered_docs else None
        return filtered_docs

    def _search_image_docs(self, img, image_query="", trajectory=None):
        """Run ROI preprocessing on an image query, then search with the main retriever."""
        roi_img = self._roi(image_query, trajectory=trajectory)
        query_image = roi_img or img
        self._log_image_retrieval_pipeline(
            trajectory,
            stage="search_image_docs_after_roi",
            image_query=image_query,
            query_image=query_image,
            note="main image retrieval will use this image after ROI preprocessing",
        )
        return self._search_with_main_retriever(query_image, query_kind="image")

    def _search_with_main_retriever(self, query, query_kind):
        """Dispatch either text or image retrieval through the configured main retriever."""
        self._log_module_output(
            None,
            "retriever_call",
            query_kind=query_kind,
            query_preview=query if isinstance(query, str) else None,
            image_info=self._describe_image_for_log(query) if not isinstance(query, str) else None,
            target_modal=(
                self.config["image_retrieval_target_modal"] if query_kind == "image" else "text"
            ),
        )
        return super()._search_with_main_retriever(query, query_kind)

    def _query_similarity(self, query_a, query_b):
        """Compute a normalized similarity score between two query strings.

        The method mixes cheap lexical normalization with the sentence-encoder path
        implemented by ``_get_query_similarity_encoder`` when available.
        """
        left = self._normalize_query_text(query_a)
        right = self._normalize_query_text(query_b)
        if not left or not right:
            return 0.0
        try:
            encoder = self._get_query_similarity_encoder()
            if encoder is None:
                raise ValueError("Text retriever encoder is unavailable.")

            emb = encoder.encode([left, right], batch_size=2, is_query=True)
            left_emb, right_emb = emb[0], emb[1]
            left_norm = np.linalg.norm(left_emb)
            right_norm = np.linalg.norm(right_emb)
            if left_norm == 0.0 or right_norm == 0.0:
                return 0.0
            similarity = float(np.dot(left_emb, right_emb) / (left_norm * right_norm))
            return max(-1.0, min(1.0, similarity))
        except Exception:
            return SequenceMatcher(None, left, right).ratio()

    def _get_query_similarity_encoder(self):
        """Lazily construct and cache the sentence encoder used for query similarity."""
        source_router = getattr(self, "source_text_retriever", None)
        if source_router is None:
            return None

        retriever = getattr(source_router, "retriever", None)
        if retriever is None:
            source = getattr(self, "current_source", None)
            retriever = source_router._switch_source(source)
        return getattr(retriever, "encoder", None)

    def _normalize_evidence_unit(self, text):
        """Canonicalize one evidence unit before de-duplication or semantic matching."""
        return self._normalize_query_text(text).rstrip(".")

    def _split_retrieval_content_into_units(self, retrieval_content):
        """Split formatted retrieval text into smaller evidence units for gain analysis."""
        text = str(retrieval_content or "").strip()
        if not text:
            return []

        doc_matches = list(re.finditer(r"(?m)^Doc\d+:\n", text))
        units = []
        if doc_matches:
            for idx, match in enumerate(doc_matches):
                start = match.end()
                end = doc_matches[idx + 1].start() if idx + 1 < len(doc_matches) else len(text)
                chunk = text[start:end].strip()
                if chunk:
                    units.append(chunk)
        else:
            units = [chunk.strip() for chunk in re.split(r"\n\s*\n+", text) if chunk.strip()]

        return units

    def _semantic_similarity(self, text_a, text_b):
        """Estimate semantic similarity between two evidence units."""
        left = self._normalize_query_text(text_a)
        right = self._normalize_query_text(text_b)
        if not left or not right:
            return None
        try:
            encoder = self._get_query_similarity_encoder()
            if encoder is None:
                raise ValueError("Text retriever encoder is unavailable.")
            emb = encoder.encode([left, right], batch_size=2, is_query=True)
            left_emb, right_emb = emb[0], emb[1]
            left_norm = np.linalg.norm(left_emb)
            right_norm = np.linalg.norm(right_emb)
            if left_norm == 0.0 or right_norm == 0.0:
                return None
            similarity = float(np.dot(left_emb, right_emb) / (left_norm * right_norm))
            return max(-1.0, min(1.0, similarity))
        except Exception:
            return SequenceMatcher(None, left, right).ratio()

    def _find_semantic_match(self, unit, candidates):
        """Find the first semantically equivalent evidence unit in a candidate pool.

        Returns:
            Tuple of ``(matched_candidate_or_None, similarity_score)``.
        """
        unit_key = self._normalize_evidence_unit(unit)
        best_match = None
        best_score = None
        for candidate in candidates:
            candidate_key = self._normalize_evidence_unit(candidate)
            if unit_key == candidate_key:
                return candidate, 1.0
            sim = self._semantic_similarity(unit, candidate)
            if sim is None:
                continue
            if best_score is None or sim > best_score:
                best_score = sim
                best_match = candidate
            if sim >= self.info_gain_similarity_threshold:
                return candidate, sim
        return None, best_score

    def _extract_thought_block_after_search(self, messages):
        """Extract the first assistant thought that appears after a retrieval feedback turn."""
        for message in reversed(messages):
            if message.get("role") != "assistant":
                continue
            content = message.get("content", "")
            if not isinstance(content, str):
                continue

            thought_match = re.search(
                r"<Thought>\s*(.*?)\s*(?:</Thought>|(?=<Search>|<Sub-Question>|<End>|<Final Answer>|Final Answer:|$))",
                content,
                re.DOTALL | re.IGNORECASE,
            )
            if thought_match:
                thought_text = thought_match.group(1).strip()
                if thought_text:
                    return thought_text

            if "<Search>" in content:
                search_match = re.search(r"<Search>\s*(.*?)\s*</Search>", content, re.DOTALL | re.IGNORECASE)
                if search_match:
                    search_text = search_match.group(1).strip()
                    if search_text:
                        return search_text
                return content.strip()
        return ""

    def _extract_last_missing_info_feedback(self, messages):
        """Recover the most recent feedback string useful for missing-info rewrites."""
        last_assistant_feedback = self._extract_thought_block_after_search(messages)
        if last_assistant_feedback:
            return last_assistant_feedback

        return (
            "Rewrite the text query to target the missing entity, attribute, relation, time, or location more explicitly."
        )

    def _extract_thought_content(self, content):
        """Parse and return the textual payload inside a ``<Thought>`` block."""
        if not isinstance(content, str):
            return ""
        thought_match = re.search(
            r"<Thought>\s*(.*?)\s*(?:</Thought>|(?=<Search>|<Sub-Question>|<End>|<Final Answer>|Final Answer:|$))",
            content,
            re.DOTALL | re.IGNORECASE,
        )
        if not thought_match:
            return ""
        return thought_match.group(1).strip()

    def _extract_last_thought_from_messages(self, messages, response=""):
        """Return the latest assistant thought from either the message list or a fresh response."""
        if response:
            thought_text = self._extract_thought_content(response)
            if thought_text:
                return thought_text
        for message in reversed(messages):
            if message.get("role") != "assistant":
                continue
            thought_text = self._extract_thought_content(message.get("content", ""))
            if thought_text:
                return thought_text
        return ""

    def _clean_final_answer_text(self, text):
        """Normalize a model final-answer string before persistence or evaluation."""
        text = str(text or "").strip()
        if not text:
            return ""

        final_answer_match = re.search(r"(?:<Final Answer>|Final Answer:)\s*(.*?)(?=<|$)", text, re.DOTALL)
        if final_answer_match:
            text = final_answer_match.group(1).strip()

        text = re.sub(r"</?(?:Thought|Search|Sub-Question|Final Answer)>", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        return text
