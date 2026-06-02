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
    def info(self, message):
        print(message)

    def module(self, module_name, payload):
        self.info(f"[{module_name}] {json.dumps(payload, ensure_ascii=False)}")


class ATL_CI_Ablation(OmniSearchPipeline):
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        super().__init__(config, prompt_template=prompt_template, retriever=retriever, generator=generator)

        prompt_path = self.config["all_prompts_path"]
        with open(prompt_path, "rb") as f:
            self.prompt_book = tomllib.load(f)

        self.rewrite_prompt = self.prompt_book["system_rewrite_similar_text_query"]
        self.ambiguity_check_prompt = self.prompt_book["system_ambiguity_check"]

        self.query_similarity_threshold = float(self.config.get("omni_query_similarity_threshold"))
        self.failure_threshold = float(self.config.get("omni_failure_threshold"))
        self.max_repair_count = int(self.config.get("omni_max_repair_count"))
        self.max_turns = int(self.config.get("omni_max_turns"))
        self.min_effective_retrieval_chars = int(self.config.get("omni_min_effective_retrieval_chars"))
        self.info_gain_similarity_threshold = float(
            self.config.get("omni_info_gain_similarity_threshold")
        )
        self.tightening_module = self._as_bool(self.config.get("tightening_module", True), default=True)
        self.rebuild_module = self._as_bool(self.config.get("rebuild_module", True), default=True)
        self.terminal = TerminalPrinter()
        self._controller_trace_filename = str(
            self.config.get("omni_controller_trace_filename", "controller_actions.jsonl")
        ).strip() or "controller_actions.jsonl"
        self._current_item_id = None
        self._current_question = None
        self.terminal.info(
            f"[AblationConfig] tightening_module={self.tightening_module}, "
            f"rebuild_module={self.rebuild_module}"
        )

    def _as_bool(self, value, default=True):
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

    def _log_module_output(self, trajectory, module_name, **payload):
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
        if not query:
            return ""
        return re.sub(r"\s+", " ", str(query).strip().lower())

    def _describe_image_for_log(self, image_obj):
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
        source_router = getattr(self, "source_text_retriever", None)
        if source_router is None:
            return None

        retriever = getattr(source_router, "retriever", None)
        if retriever is None:
            source = getattr(self, "current_source", None)
            retriever = source_router._switch_source(source)
        return getattr(retriever, "encoder", None)

    def _normalize_evidence_unit(self, text):
        return self._normalize_query_text(text).rstrip(".")

    def _split_retrieval_content_into_units(self, retrieval_content):
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
        last_assistant_feedback = self._extract_thought_block_after_search(messages)
        if last_assistant_feedback:
            return last_assistant_feedback

        return (
            "Rewrite the text query to target the missing entity, attribute, relation, time, or location more explicitly."
        )

    def _extract_thought_content(self, content):
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
        text = str(text or "").strip()
        if not text:
            return ""

        final_answer_match = re.search(r"(?:<Final Answer>|Final Answer:)\s*(.*?)(?=<|$)", text, re.DOTALL)
        if final_answer_match:
            text = final_answer_match.group(1).strip()

        text = re.sub(r"</?(?:Thought|Search|Sub-Question|Final Answer)>", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _looks_like_search_state(self, text):
        normalized = self._normalize_query_text(text)
        if not normalized:
            return False
        markers = (
            "text retrieval:",
            "image retrieval:",
            "no retrieval",
            "<search>",
            "</search>",
            "search>",
        )
        return any(marker in normalized for marker in markers)

    def _finalize_answer_with_fallback(self, question, messages, response, trajectory=None):
        last_thought = self._extract_last_thought_from_messages(messages, response=response)
        response_text = str(response or "")
        response_is_search_state = "<Search>" in response_text or self._looks_like_search_state(response_text)

        fallback_candidate = last_thought or ("" if response_is_search_state else response_text)
        fallback_answer = self._clean_final_answer_text(fallback_candidate)
        if not fallback_answer or self._looks_like_search_state(fallback_answer):
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
                and not self._looks_like_search_state(finalized)
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
        return re.findall(r"[a-z0-9]+", self._normalize_query_text(query))

    def _is_specificity_increasing_rewrite(self, reference_query, candidate_query):
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
        if not response:
            return ""
        return self._extract_retrieval_query(response, "Text Retrieval")

    def _is_retrieval_feedback_message(self, message):
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

    def _trim_recent_retrieval_turns(self, messages, max_turns_to_trim=None):
        trimmed_messages = list(messages)
        removed_messages = []
        removed_turns = 0
        trim_budget = self.max_repair_count + 1 if max_turns_to_trim is None else max_turns_to_trim

        while trimmed_messages and removed_turns < trim_budget:
            removed_in_this_turn = False

            if trimmed_messages and trimmed_messages[-1].get("role") == "assistant":
                assistant_content = trimmed_messages[-1].get("content", "")
                if isinstance(assistant_content, str) and (
                    "<Search>" in assistant_content or "Final Answer" in assistant_content
                ):
                    removed_messages.append(trimmed_messages.pop())
                    removed_in_this_turn = True

            if trimmed_messages and self._is_retrieval_feedback_message(trimmed_messages[-1]):
                removed_messages.append(trimmed_messages.pop())
                removed_in_this_turn = True

            if not removed_in_this_turn:
                break
            removed_turns += 1

        removed_messages.reverse()
        return trimmed_messages, removed_messages, removed_turns

    def _should_remove_search_state(self, message, duplicate_query="", zero_gain_query=""):
        if message.get("role") != "assistant":
            return False, ""
        content = message.get("content", "")
        if not isinstance(content, str) or "<Search>" not in content:
            return False, ""

        retrieval_mode, query_txt = self._extract_search_mode_and_query(content)
        if retrieval_mode != "text_retrieval":
            return False, ""

        normalized_query = self._normalize_query_text(query_txt)
        if duplicate_query and normalized_query == self._normalize_query_text(duplicate_query):
            return True, "duplicate_prev_query"
        if zero_gain_query and normalized_query == self._normalize_query_text(zero_gain_query):
            return True, "delta_f_zero"
        return False, ""

    def _prune_trajectory_states(self, messages, duplicate_query="", zero_gain_query="", trajectory=None):
        if not duplicate_query and not zero_gain_query:
            return messages

        pruned_messages = []
        removed_messages = []
        expect_feedback_after_removed_search = False
        expect_next_thought_after_removed_search = False

        for message in messages:
            if expect_next_thought_after_removed_search and message.get("role") == "assistant":
                content = message.get("content", "")
                if isinstance(content, str) and "<Thought>" in content:
                    removed_messages.append({"reason": "post_search_thought", "message": message})
                    expect_next_thought_after_removed_search = False
                    continue
                expect_next_thought_after_removed_search = False

            if expect_feedback_after_removed_search and self._is_retrieval_feedback_message(message):
                removed_messages.append(message)
                expect_feedback_after_removed_search = False
                expect_next_thought_after_removed_search = True
                continue

            expect_feedback_after_removed_search = False
            should_remove, reason = self._should_remove_search_state(
                message,
                duplicate_query=duplicate_query,
                zero_gain_query=zero_gain_query,
            )
            if should_remove:
                removed_messages.append({"reason": reason, "message": message})
                expect_feedback_after_removed_search = True
                continue

            pruned_messages.append(message)

        self._log_module_output(
            trajectory,
            "prune_trajectory_states",
            duplicate_query=duplicate_query,
            zero_gain_query=zero_gain_query,
            removed_count=len(removed_messages),
            removed_messages=removed_messages,
        )
        return pruned_messages

    def _rebuild_messages(
        self,
        messages,
        mode,
        reference_query="",
        vlm_feedback="",
        trajectory=None,
    ):
        if mode == "missing_info":
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
                    rebuilt_messages[idx] = {"role": "assistant", "content": rebuilt_response}
                    break
            else:
                rebuilt_messages.append({"role": "assistant", "content": rebuilt_response})
            return rebuilt_messages, rebuilt_response

        trimmed_messages, removed_messages, removed_turns = self._trim_recent_retrieval_turns(messages)
        reflection_response = (
            "<Thought>\n"
            "Repeated or low-gain text retrieval was detected. The recent repeated retrieval turns have been discarded. "
            "The current query path is not working and the current retrieval results are not sufficient to move forward. "
            f"You must not ask the same question again, and you must not issue the same or near-duplicate Text Retrieval query as: {reference_query}. "
            "Choose a genuinely different next step. If you choose Text Retrieval, ask a different question that targets a new missing clue, entity, relation, location, time, or attribute instead of rephrasing the same request. "
            "If a different text query is not well justified, choose Image Retrieval: <anchored phrase>, No Retrieval, or give a Final Answer if the evidence is already sufficient. "
            "Do not repeat the same failed plan.\n"
            "</Thought>"
        )
        self._log_module_output(
            trajectory,
            "rebuild_messages",
            mode=mode,
            reference_query=reference_query,
            vlm_feedback=vlm_feedback,
            rebuilt_response=reflection_response,
            removed_turns=removed_turns,
            removed_messages=removed_messages,
        )
        rebuilt_messages = trimmed_messages
        rebuilt_messages.append({"role": "user", "content": [{"type": "text", "text": reflection_response}]})
        return rebuilt_messages, reflection_response

    def _ambiguity_check(self, query_txt, trajectory=None):
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
        return self._extract_retrieval_query(response, "Image Retrieval")

    def _roi(self, image_query, trajectory=None):
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
        value = self._normalize_query_text(value)
        if value in {"high", "medium", "low"}:
            return value
        return ""

    def _normalize_alignment_status(self, value):
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

    def _alignment_layer(self, text_query, enhanced_information, trajectory=None):
        original_query = str(text_query or "").strip()
        if not original_query:
            self._log_module_output(
                trajectory,
                "alignment_layer",
                status="empty_query",
                original_query=original_query,
            )
            return ""

        enhanced_information = str(enhanced_information or "").strip()
        if not enhanced_information:
            self._log_module_output(
                trajectory,
                "alignment_layer",
                status="empty_enhanced_information",
                original_query=original_query,
            )
            return original_query

        system_prompt = str(self.prompt_book.get("system_alignment_layer", "")).strip()

        if not system_prompt:
            self._log_module_output(
                trajectory,
                "alignment_layer",
                status="empty_prompt",
                original_query=original_query,
                enhanced_information=enhanced_information,
            )
            return original_query

        user_prompt = (
            f"Original Query:\n{original_query}\n\n"
            f"Enhanced Information:\n{enhanced_information}\n\n"
            "Return JSON only.\n"
            "Prefer this schema:\n"
            "{\n"
            '  "ambiguous_references": ["..."],\n'
            '  "grounding_cues": ["..."],\n'
            '  "entity_status": "resolved|low_confidence|unresolved",\n'
            '  "canonical_entity": "...",\n'
            '  "alternative_entities": ["..."],\n'
            '  "entity_confidence": "high|medium|low",\n'
            '  "entity_type": "...",\n'
            '  "target_attribute": "...",\n'
            '  "refined_queries": ["..."]\n'
            "}\n"
            "Use `canonical_entity` for the grounded entity name when possible.\n"
            "Use `alternative_entities` when multiple plausible entities remain.\n"
            "Use `entity_confidence` to indicate how strongly the enhanced information supports the canonical entity.\n"
            "Use `target_attribute` for the missing attribute, relation, time, location, taxonomy, or value being asked for.\n"
            "If one candidate is plausible but still uncertain, set `entity_status` to `low_confidence` instead of pretending the entity is resolved.\n"
            "If the entity cannot be resolved from the enhanced information, set `entity_status` to `unresolved`.\n"
            "Return JSON only."
        )

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ]

        try:
            response = self._generate_text(messages)
        except Exception as exc:
            self.terminal.info(f"Alignment layer generation failed, fallback to original query: {exc}")
            self._log_module_output(
                trajectory,
                "alignment_layer",
                status="generation_failed",
                original_query=original_query,
                enhanced_information=enhanced_information,
                error=str(exc),
            )
            return original_query

        response = str(response or "").strip()
        if not response:
            self._log_module_output(
                trajectory,
                "alignment_layer",
                status="empty_response",
                original_query=original_query,
                enhanced_information=enhanced_information,
            )
            return original_query

        try:
            payload = json.loads(response)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if not json_match:
                self.terminal.info("Alignment layer returned non-JSON output; fallback to original query.")
                self._log_module_output(
                    trajectory,
                    "alignment_layer",
                    status="non_json_response",
                    original_query=original_query,
                    enhanced_information=enhanced_information,
                )
                return original_query
            try:
                payload = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                self.terminal.info("Alignment layer JSON parsing failed; fallback to original query.")
                self._log_module_output(
                    trajectory,
                    "alignment_layer",
                    status="json_parse_failed",
                    original_query=original_query,
                    enhanced_information=enhanced_information,
                )
                return original_query

        if not isinstance(payload, dict):
            self._log_module_output(
                trajectory,
                "alignment_layer",
                status="invalid_payload",
                original_query=original_query,
                enhanced_information=enhanced_information,
            )
            return original_query

        refined_queries = self._coerce_alignment_list(payload.get("refined_queries", []))
        grounding_cues = self._coerce_alignment_list(payload.get("grounding_cues", []))
        canonical_entity = str(payload.get("canonical_entity", "") or "").strip()
        alternative_entities = self._coerce_alignment_list(payload.get("alternative_entities", []))
        entity_type = str(payload.get("entity_type", "") or "").strip()
        entity_status = self._normalize_alignment_status(payload.get("entity_status", ""))
        entity_confidence = self._normalize_alignment_confidence(payload.get("entity_confidence", ""))
        target_attribute = str(payload.get("target_attribute", "") or "").strip()

        aligned_query = self._build_structured_alignment_query(
            original_query=original_query,
            canonical_entity=canonical_entity,
            alternative_entities=alternative_entities,
            entity_status=entity_status,
            entity_confidence=entity_confidence,
            target_attribute=target_attribute,
            refined_queries=refined_queries,
            grounding_cues=grounding_cues,
        )
        if aligned_query:
            self._log_module_output(
                trajectory,
                "alignment_layer",
                status="ok",
                original_query=original_query,
                enhanced_information=enhanced_information,
                entity_status=entity_status,
                entity_confidence=entity_confidence,
                canonical_entity=canonical_entity,
                alternative_entities=alternative_entities,
                entity_type=entity_type,
                target_attribute=target_attribute,
                grounding_cues=grounding_cues,
                refined_queries=refined_queries,
                aligned_query=aligned_query,
            )
            return aligned_query

        self._log_module_output(
            trajectory,
            "alignment_layer",
            status="fallback_original_query",
            original_query=original_query,
            enhanced_information=enhanced_information,
            entity_status=entity_status,
            entity_confidence=entity_confidence,
            canonical_entity=canonical_entity,
            alternative_entities=alternative_entities,
            entity_type=entity_type,
            target_attribute=target_attribute,
            grounding_cues=grounding_cues,
            refined_queries=refined_queries,
        )
        return original_query

    def _update_history(self, retrieval_query, retrieval_content):
        return {
            "query": retrieval_query,
            "content": retrieval_content,
        }

    def _estimate_information_gain(
        self,
        query_txt,
        reference_query,
        retrieval_content,
        cumulative_retrieval_units,
    ):
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

    def _build_simple_trajectory_from_messages(self, messages):
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

    def _build_effective_trajectory(self, messages, fallback_trajectory=None):
        if fallback_trajectory:
            return fallback_trajectory
        return self._build_simple_trajectory_from_messages(messages)

    def iterative_infer(self, question, id, image_id):
        img_path = os.path.join(self.data_dir, self.dataset_name, "images", f"{image_id}.jpg")
        if not os.path.exists(img_path):
            return None

        img = Image.open(img_path).convert("RGB")
        self._current_image = img
        self._current_item_id = id
        self._current_question = question
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
        start_time = time.time()
        last_retrieval_query = None
        cumulative_retrieval_units = []
        repair_count = 0
        conversation_num = 0
        similar_query_streak = 0
        low_gain_streak = 0
        response = self._generate_text(messages)
        response = self._truncate_after_search(response)
        self.terminal.info(f"First Response: {response}")
        self._record_response_actions(trajectory, response)
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
                current_query = self._extract_retrieval_query(response, "Text Retrieval")
                self.terminal.info(f"Query Text: {current_query}")

                if last_retrieval_query is not None:
                    similarity = self._query_similarity(current_query, last_retrieval_query)
                    self.terminal.info(f"Query similarity with previous query: {similarity:.4f}")
                    if similarity > self.query_similarity_threshold:
                        similar_query_streak += 1
                        self._log_module_output(
                            trajectory,
                            "query_similarity_gate",
                            query=current_query,
                            reference_query=last_retrieval_query,
                            similarity=similarity,
                            streak=similar_query_streak,
                        )
                        if self.rebuild_module and similar_query_streak >= 2:
                            messages, rebuilt_response = self._rebuild_messages(
                                messages,
                                mode="reflection",
                                reference_query=last_retrieval_query,
                                vlm_feedback=self._extract_last_missing_info_feedback(messages),
                                trajectory=trajectory,
                            )
                            rewritten_query = self._extract_text_retrieval_query_from_search(rebuilt_response)
                            self._append_controller_action(
                                trajectory,
                                controller_name="recovery",
                                trigger_condition="similar_query_streak>=2",
                                mode="reflection",
                                reference_query=last_retrieval_query,
                                query_before=current_query,
                                query_after=rewritten_query,
                                similar_query_streak=similar_query_streak,
                                repair_count_before=repair_count,
                                max_repair_count=self.max_repair_count,
                            )
                            repair_count = 0
                            similar_query_streak = 0
                            self.terminal.info("Similar query streak reached 2; switched to reflection.")
                            messages, response = self._run_reflection_step(messages, trajectory)
                            conversation_num += 1
                            continue
                        if self.rebuild_module:
                            vlm_feedback = self._extract_last_missing_info_feedback(messages)
                            original_query = current_query
                            messages, rebuilt_response = self._rebuild_messages(
                                messages,
                                mode="missing_info",
                                reference_query=current_query,
                                vlm_feedback=vlm_feedback,
                                trajectory=trajectory,
                            )
                            rewritten_query = self._extract_text_retrieval_query_from_search(rebuilt_response)
                            self._append_controller_action(
                                trajectory,
                                controller_name="repair",
                                trigger_condition="similar_query_streak>=1_missing_info",
                                mode="missing_info",
                                reference_query=current_query,
                                query_before=original_query,
                                query_after=rewritten_query,
                                similar_query_streak=similar_query_streak,
                                repair_count_before=repair_count,
                                max_repair_count=self.max_repair_count,
                            )
                            response = rebuilt_response
                            self._record_response_actions(trajectory, response)
                            repair_count += 1
                            self.terminal.info("Similar query similarity exceeded threshold; rebuilt messages with missing-info rewrite.")
                            conversation_num += 1
                            continue
                    else:
                        similar_query_streak = 0

                search_query = current_query
                if self.tightening_module and self._ambiguity_check(current_query, trajectory=trajectory):
                    image_query = self._extract_image_retrieval_query(response)
                    roi_img = self._roi(image_query or current_query, trajectory=trajectory)
                    self._log_image_retrieval_pipeline(
                        trajectory,
                        stage="ambiguity_alignment_before_search",
                        image_query=image_query or current_query,
                        query_image=roi_img or img,
                        note="about to run auxiliary image retrieval for ambiguity alignment",
                    )
                    image_docs = self._search_with_main_retriever(roi_img or img, query_kind="image")
                    self._log_image_retrieval_pipeline(
                        trajectory,
                        stage="ambiguity_alignment_raw_result",
                        image_query=image_query or current_query,
                        query_image=roi_img or img,
                        retrieved_docs=image_docs,
                    )
                    image_docs = self._sanitize_retrieved_docs(
                        image_docs,
                        retrieval_mode="image_retrieval",
                        trajectory=trajectory,
                    )
                    enhanced_information = self._flatten(image_docs, trajectory=trajectory)
                    search_query = self._alignment_layer(current_query, enhanced_information, trajectory=trajectory)
                    self._append_controller_action(
                        trajectory,
                        controller_name="tightening",
                        trigger_condition="ambiguity_check==True",
                        mode="ambiguity_alignment",
                        query_before=current_query,
                        query_after=search_query,
                        image_query=(image_query or current_query),
                        used_roi_image=roi_img is not None,
                    )

                retrieved_docs = self._search_text_docs(search_query)
                retrieval_content = self._format_retrieval_content(retrieved_docs)
                retrieval_content = self._clip_text(retrieval_content, self.retrieval_char_limit)
                self._log_retrieval_preview(retrieval_content)
                self._record_retrieval_result(
                    trajectory=trajectory,
                    retrieval_mode=retrieval_mode,
                    query_txt=search_query,
                    retrieval_content=retrieval_content,
                )
                self._update_history(search_query, retrieval_content)

                info_gain, accepted_units = self._estimate_information_gain(
                    search_query,
                    last_retrieval_query,
                    retrieval_content,
                    cumulative_retrieval_units,
                )
                self.terminal.info(f"Retrieval information increment: {info_gain:.4f}")
                self._log_module_output(
                    trajectory,
                    "information_increment",
                    query=search_query,
                    reference_query=last_retrieval_query,
                    information_increment=info_gain,
                )
                if info_gain < self.failure_threshold:
                    low_gain_streak += 1
                    self._log_module_output(
                        trajectory,
                        "low_gain_gate",
                        query=search_query,
                        reference_query=last_retrieval_query,
                        information_increment=info_gain,
                        streak=low_gain_streak,
                    )
                    if self.rebuild_module and low_gain_streak >= 2:
                        messages, rebuilt_response = self._rebuild_messages(
                            messages,
                            mode="reflection",
                            reference_query=search_query,
                            vlm_feedback=retrieval_content,
                            trajectory=trajectory,
                        )
                        zero_gain_query = search_query if info_gain == 0.0 else ""
                        messages = self._prune_trajectory_states(
                            messages,
                            zero_gain_query=zero_gain_query,
                            trajectory=trajectory,
                        )
                        rewritten_query = self._extract_text_retrieval_query_from_search(rebuilt_response)
                        self._append_controller_action(
                            trajectory,
                            controller_name="recovery",
                            trigger_condition="low_gain_streak>=2",
                            mode="reflection",
                            reference_query=last_retrieval_query,
                            query_before=search_query,
                            query_after=rewritten_query,
                            low_gain_streak=low_gain_streak,
                            information_increment=info_gain,
                            failure_threshold=self.failure_threshold,
                            repair_count_before=repair_count,
                            max_repair_count=self.max_repair_count,
                            rebuilt_as_search=bool(rebuilt_response.startswith("<Search>")),
                        )
                        repair_count = 0
                        low_gain_streak = 0
                        self.terminal.info("Low-gain streak reached 2; switched to reflection.")
                        messages, response = self._run_reflection_step(messages, trajectory)
                        conversation_num += 1
                        continue
                    if self.rebuild_module:
                        original_query = search_query
                        vlm_feedback = (
                            "The latest retrieval brought little or no new information compared with the previous retrieval. "
                            "Rewrite the query to target the missing clue more explicitly.\n"
                            f"Latest retrieval content:\n{retrieval_content}"
                        )
                        zero_gain_query = search_query if info_gain == 0.0 else ""
                        messages, rebuilt_response = self._rebuild_messages(
                            messages,
                            mode="missing_info",
                            reference_query=search_query,
                            vlm_feedback=vlm_feedback,
                            trajectory=trajectory,
                        )
                        rewritten_query = self._extract_text_retrieval_query_from_search(rebuilt_response)
                        self._append_controller_action(
                            trajectory,
                            controller_name="repair",
                            trigger_condition="low_gain_streak>=1_missing_info",
                            mode="missing_info",
                            reference_query=last_retrieval_query,
                            query_before=original_query,
                            query_after=rewritten_query,
                            low_gain_streak=low_gain_streak,
                            information_increment=info_gain,
                            failure_threshold=self.failure_threshold,
                            repair_count_before=repair_count,
                            max_repair_count=self.max_repair_count,
                        )
                        response = rebuilt_response
                        self._record_response_actions(trajectory, response)
                        repair_count += 1
                        self.terminal.info("Low-gain signal exceeded threshold; rebuilt messages with missing-info rewrite.")
                        conversation_num += 1
                        continue
                else:
                    low_gain_streak = 0

                last_retrieval_query = search_query
                cumulative_retrieval_units.extend(accepted_units)
                repair_count = 0
                similar_query_streak = 0
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
                    messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_text = f"{e.__class__.__name__}: {e}"
                    self.terminal.info(f"Inference error, hidden states ignored: {error_text}")
                    trajectory.append({"action": "error", "content": error_text})
                    effective_trajectory = self._build_effective_trajectory(messages, fallback_trajectory=trajectory)
                    record = {
                        "question": question,
                        "id": id,
                        "final_answer": response,
                        "status": "generation_error",
                        "error": error_text,
                        "duration_seconds": time.time() - start_time,
                        "trajectory": effective_trajectory,
                    }
                    self._write_trajectory(record)
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
                else:
                    retrieval_mode = "no_retrieval"
                    retrieval_content = None

                self._record_retrieval_result(
                    trajectory=trajectory,
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
                    messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_text = f"{e.__class__.__name__}: {e}"
                    self.terminal.info(f"Inference error, hidden states ignored: {error_text}")
                    trajectory.append({"action": "error", "content": error_text})
                    effective_trajectory = self._build_effective_trajectory(messages, fallback_trajectory=trajectory)
                    record = {
                        "question": question,
                        "id": id,
                        "final_answer": response,
                        "status": "generation_error",
                        "error": error_text,
                        "duration_seconds": time.time() - start_time,
                        "trajectory": effective_trajectory,
                    }
                    self._write_trajectory(record)
                    return response, messages
            else:
                break

            conversation_num += 1

        pattern = r"(?:<Final Answer>|Final Answer:)\s*(.*?)(?=<|$)"
        final_answer_match = re.search(pattern, response, re.DOTALL)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip().replace("\n", "")
            self.terminal.info(f"Final Answer: {final_answer}")
            effective_trajectory = self._build_effective_trajectory(messages, fallback_trajectory=trajectory)
            record = {
                "question": question,
                "id": id,
                "final_answer": final_answer,
                "status": "ok",
                "duration_seconds": time.time() - start_time,
                "trajectory": effective_trajectory,
            }
            self._write_trajectory(record)
            return final_answer, messages

        finalized_answer = self._finalize_answer_with_fallback(
            question,
            messages,
            response,
            trajectory=trajectory,
        )
        if finalized_answer:
            self.terminal.info(f"Final Answer (fallback): {finalized_answer}")
            effective_trajectory = self._build_effective_trajectory(messages, fallback_trajectory=trajectory)
            record = {
                "question": question,
                "id": id,
                "final_answer": finalized_answer,
                "status": "ok",
                "duration_seconds": time.time() - start_time,
                "trajectory": effective_trajectory,
            }
            self._write_trajectory(record)
            return finalized_answer, messages

        self.terminal.info(
            f"Warning: reached end of agent loop for item {conversation_num} without a 'Final Answer'. returning last response"
        )
        effective_trajectory = self._build_effective_trajectory(messages, fallback_trajectory=trajectory)
        record = {
            "question": question,
            "id": id,
            "final_answer": response,
            "status": "missing_final_answer",
            "duration_seconds": time.time() - start_time,
            "trajectory": effective_trajectory,
        }
        self._write_trajectory(record)
        return response, messages
