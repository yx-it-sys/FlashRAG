from flashrag.pipeline.omni_pipeline import OmniSearchPipeline
import inspect
import json
import os
import re
import time
import tomllib
from PIL import Image


class AmbRewritePipeline(OmniSearchPipeline):
    """Baseline that augments ambiguous text retrieval with image-grounded query rewriting."""
    AMBIGUITY_PROMPT_BOOK_PATH = "/home/you/FlashRAG/exps/idea10/prompts/new_method/prompt_books.toml"

    DEFAULT_DETECTOR_SYSTEM_PROMPT = (
        "You are a strict detector for image-dependent references in retrieval queries. "
        "Return JSON only."
    )
    DEFAULT_DETECTOR_USER_PROMPT = (
        "Determine whether the text retrieval query contains a reference that cannot be resolved "
        "without grounding in the image.\n\n"
        "Query:\n{query}\n\n"
        "Return JSON with this schema only:\n"
        "{\n"
        '  "needs_image_grounding": "Yes" or "No",\n'
        '  "ambiguous_references": ["..."],\n'
        '  "reason": "...",\n'
        '  "image_focus": "short phrase for the referred region/object, empty if none",\n'
        '  "target_attribute": "attribute/relation/value being asked for, empty if none"\n'
        "}\n"
        "Mark Yes only when the query includes image-dependent references such as pronouns, "
        "demonstratives, or phrases like 'in the image', 'pictured', 'its color', "
        "'the structure in the image', or similar."
    )
    DEFAULT_RESOLVER_SYSTEM_PROMPT = (
        "You identify the most likely entity referred to by an ambiguous query using image retrieval evidence. "
        "Return JSON only."
    )
    DEFAULT_RESOLVER_USER_PROMPT = (
        "Resolve the ambiguous image-dependent reference in the text retrieval query.\n\n"
        "Original Query:\n{query}\n\n"
        "Detector Analysis:\n{detector_analysis}\n\n"
        "Image Retrieval Evidence:\n{image_evidence}\n\n"
        "Return JSON only with this schema:\n"
        "{{\n"
        '  "resolved_entity": "...",\n'
        '  "confidence": "high" or "medium" or "low",\n'
        '  "supporting_evidence": ["..."],\n'
        '  "rewrite_query": "..."\n'
        "}}\n"
        "The rewritten query must replace the ambiguous reference with the resolved entity name and remain suitable for text retrieval."
    )

    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        super().__init__(config, prompt_template=prompt_template, retriever=retriever, generator=generator)
        self.trajectory_path = os.path.join(self.output_dir, "omnisearch_trajectories.jsonl")
        self.prompt_book = {}
        with open(self.AMBIGUITY_PROMPT_BOOK_PATH, "rb") as f:
            self.prompt_book = tomllib.load(f)
        self.amb_detector_system_prompt = self.DEFAULT_DETECTOR_SYSTEM_PROMPT
        self.amb_detector_user_prompt = self.prompt_book.get(
            "system_ambiguity_check",
            self.DEFAULT_DETECTOR_USER_PROMPT,
        )
        self.amb_resolver_system_prompt = self._config_value(
            self.config,
            "amb_resolver_system_prompt",
            self.DEFAULT_RESOLVER_SYSTEM_PROMPT,
        )
        self.amb_resolver_user_prompt = self._config_value(
            self.config,
            "amb_resolver_user_prompt",
            self.DEFAULT_RESOLVER_USER_PROMPT,
        )
        self.amb_rewrite_enabled = bool(self._config_value(self.config, "amb_rewrite_enabled", True))
        self.amb_rewrite_max_image_docs = int(self._config_value(self.config, "amb_rewrite_max_image_docs", 3))
        self.amb_rewrite_max_text_docs = int(self._config_value(self.config, "amb_rewrite_max_text_docs", 3))
        self.amb_candidate_field_priority = (
            "entity_name",
            "title",
            "page_name",
            "name",
            "subject",
            "caption",
        )

    def _extract_json_payload(self, raw_text):
        text = str(raw_text or "").strip()
        if not text:
            return {}
        try:
            payload = json.loads(text)
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            payload = json.loads(match.group(0))
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _coerce_str_list(self, value):
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        results = []
        for item in value:
            item = str(item or "").strip()
            if item:
                results.append(item)
        return results

    def _normalize_yes_no(self, value):
        normalized = str(value or "").strip().lower()
        if normalized == "yes":
            return "Yes"
        if normalized == "no":
            return "No"
        return ""

    def _is_valid_rewrite(self, original_query, rewritten_query, resolved_entity):
        original_query = str(original_query or "").strip()
        rewritten_query = str(rewritten_query or "").strip()
        resolved_entity = str(resolved_entity or "").strip()
        if not rewritten_query or not resolved_entity:
            return False
        if rewritten_query.lower() == original_query.lower():
            return False
        return resolved_entity.lower() in rewritten_query.lower()

    def _search_with_target_modal(self, query, query_kind, target_modal=None):
        search_callable = getattr(self.retriever, "_search", self.retriever.search)
        search_signature = inspect.signature(search_callable)
        search_params = search_signature.parameters
        supports_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in search_params.values()
        )
        kwargs = {}
        topk = self.config["image_retrieval_topk"] if query_kind == "image" else self.config["text_retrieval_topk"]
        if topk is not None and ("num" in search_params or supports_kwargs):
            kwargs["num"] = topk
        if "query_type" in search_params or supports_kwargs:
            kwargs["query_type"] = query_kind
        if target_modal is None and query_kind == "image":
            target_modal = self.config["image_retrieval_target_modal"]
        if target_modal is not None and ("target_modal" in search_params or supports_kwargs):
            kwargs["target_modal"] = target_modal
        return self.retriever.search(query, **kwargs)

    def _normalize_docs_to_list(self, docs):
        if docs is None:
            return []
        if isinstance(docs, list):
            return list(docs)
        return [docs]

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
                str(doc.get("page_name", "")).strip(),
                str(doc.get("name", "")).strip(),
                str(doc.get("text", "")).strip(),
                str(doc.get("contents", "")).strip(),
                str(doc.get("content", "")).strip(),
                str(doc.get("body", "")).strip(),
                str(doc.get("passage", "")).strip(),
                str(doc.get("caption", "")).strip(),
            ]
            meaningful = [field for field in text_fields if field and field.lower() != "none"]
            return len(meaningful) == 0
        return False

    def _sanitize_retrieved_docs(self, retrieved_docs):
        docs = self._normalize_docs_to_list(retrieved_docs)
        return [doc for doc in docs if not self._is_invalid_retrieved_doc(doc)]

    def _extract_candidate_name_from_doc(self, doc):
        if isinstance(doc, str):
            first_line = str(doc).strip().splitlines()[0].strip() if str(doc).strip() else ""
            return first_line
        if not isinstance(doc, dict):
            return ""

        for field in self.amb_candidate_field_priority:
            value = str(doc.get(field, "") or "").strip()
            if value and value.lower() != "none":
                return value

        for field in ("text", "contents", "content", "body", "passage"):
            value = str(doc.get(field, "") or "").strip()
            if not value or value.lower() == "none":
                continue
            first_line = value.splitlines()[0].strip()
            if first_line:
                return first_line[:160]
        return ""

    def _extract_candidate_attributes_from_doc(self, doc):
        if not isinstance(doc, dict):
            return []
        attr_parts = []
        attrs = doc.get("entity_attributes", {}) or {}
        if isinstance(attrs, dict):
            for key, value in attrs.items():
                if value in (None, "", [], {}):
                    continue
                attr_parts.append(f"{key}: {value}")
        for key in ("title", "caption", "image_url", "url", "source"):
            value = doc.get(key)
            if value in (None, "", [], {}):
                continue
            text = str(value).strip()
            if not text or text.lower() == "none":
                continue
            attr_parts.append(f"{key}: {text}")
        return attr_parts[:5]

    def _search_auxiliary_image_docs(self, img):
        try:
            docs = self._search_with_target_modal(img, query_kind="image")
            docs = self._sanitize_retrieved_docs(docs)
            if docs:
                return docs, "image_query_configured_target"
        except NotImplementedError as exc:
            print(f"Primary image retrieval path is unsupported, fallback to image-to-image retrieval: {exc}")
        except Exception as exc:
            print(f"Primary image retrieval path failed, fallback to image-to-image retrieval: {exc}")

        try:
            docs = self._search_with_target_modal(img, query_kind="image", target_modal="image")
            docs = self._sanitize_retrieved_docs(docs)
            if docs:
                return docs, "image_query_image_target_fallback"
        except Exception as exc:
            print(f"Image-to-image fallback retrieval failed: {exc}")

        return [], "empty"

    def _build_detector_messages(self, query_txt):
        detector_prompt = self.amb_detector_user_prompt
        if "{text_query}" in detector_prompt:
            detector_prompt = detector_prompt.format(text_query=query_txt)
        else:
            detector_prompt = detector_prompt.format(query=query_txt)
        return [
            {"role": "system", "content": [{"type": "text", "text": self.amb_detector_system_prompt}]},
            {
                "role": "user",
                "content": [{"type": "text", "text": detector_prompt}],
            },
        ]

    def _build_resolver_messages(self, query_txt, detector_payload, image_evidence):
        detector_analysis = json.dumps(detector_payload, ensure_ascii=False, indent=2)
        user_prompt = self.amb_resolver_user_prompt.format(
            query=query_txt,
            detector_analysis=detector_analysis,
            image_evidence=image_evidence,
        )
        return [
            {"role": "system", "content": [{"type": "text", "text": self.amb_resolver_system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ]

    def _detect_query_needs_image_grounding(self, query_txt):
        if not self.amb_rewrite_enabled:
            return {
                "needs_image_grounding": "No",
                "ambiguous_references": [],
                "reason": "disabled",
                "image_focus": "",
                "target_attribute": "",
            }

        try:
            response = self._generate_text(self._build_detector_messages(query_txt))
        except Exception as exc:
            print(f"Ambiguity detector failed, fallback to No: {exc}")
            return {
                "needs_image_grounding": "No",
                "ambiguous_references": [],
                "reason": f"detector_failed: {exc}",
                "image_focus": "",
                "target_attribute": "",
            }

        payload = self._extract_json_payload(response)
        needs_image_grounding = self._normalize_yes_no(payload.get("needs_image_grounding", ""))
        if not needs_image_grounding:
            needs_image_grounding = self._normalize_yes_no(payload.get("entity_ambiguous", ""))
        if not needs_image_grounding:
            needs_image_grounding = "No"
        return {
            "needs_image_grounding": needs_image_grounding,
            "ambiguous_references": self._coerce_str_list(payload.get("ambiguous_references")),
            "reason": str(payload.get("reason", payload.get("remark", "")) or "").strip(),
            "image_focus": str(payload.get("image_focus", "") or "").strip(),
            "target_attribute": str(payload.get("target_attribute", "") or "").strip(),
        }

    def _extract_entity_candidates_from_image_docs(self, image_docs):
        docs = self._sanitize_retrieved_docs(image_docs)
        if not docs:
            return []
        candidates = []
        for doc in docs[: self.amb_rewrite_max_image_docs]:
            if isinstance(doc, dict):
                entities = doc.get("entities")
                if isinstance(entities, list):
                    for entity in entities:
                        if not isinstance(entity, dict):
                            continue
                        name = str(entity.get("entity_name", "") or "").strip()
                        if not name:
                            continue
                        attrs = entity.get("entity_attributes", {}) or {}
                        attr_parts = []
                        if isinstance(attrs, dict):
                            for key, value in attrs.items():
                                if value in (None, "", [], {}):
                                    continue
                                attr_parts.append(f"{key}: {value}")
                        candidate_text = name
                        if attr_parts:
                            candidate_text += " | " + "; ".join(attr_parts[:5])
                        candidates.append(candidate_text)

            generic_name = self._extract_candidate_name_from_doc(doc)
            if not generic_name:
                continue
            attr_parts = self._extract_candidate_attributes_from_doc(doc)
            candidate_text = generic_name
            if attr_parts:
                candidate_text += " | " + "; ".join(attr_parts)
            candidates.append(candidate_text)
        deduped = []
        seen = set()
        for item in candidates:
            normalized = item.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(item)
        return deduped

    def _resolve_entity_and_rewrite_query(self, query_txt, detector_payload, image_docs):
        image_docs = self._sanitize_retrieved_docs(image_docs)
        image_evidence = self._format_image_retrieval_content(image_docs)
        image_evidence = self._clip_text(image_evidence, self.retrieval_char_limit)
        candidates = self._extract_entity_candidates_from_image_docs(image_docs)
        if candidates:
            image_evidence = (
                "Candidate entities:\n- "
                + "\n- ".join(candidates)
                + "\n\n"
                + image_evidence
            )

        try:
            response = self._generate_text(
                self._build_resolver_messages(query_txt, detector_payload, image_evidence)
            )
        except Exception as exc:
            print(f"Ambiguity resolver failed, skip rewrite: {exc}")
            return {
                "resolved_entity": "",
                "confidence": "low",
                "supporting_evidence": [],
                "rewrite_query": "",
            }

        payload = self._extract_json_payload(response)
        return {
            "resolved_entity": str(payload.get("resolved_entity", "") or "").strip(),
            "confidence": str(payload.get("confidence", "") or "").strip().lower(),
            "supporting_evidence": self._coerce_str_list(payload.get("supporting_evidence")),
            "rewrite_query": str(payload.get("rewrite_query", "") or "").strip(),
        }

    def _build_augmented_text_retrieval(self, img, original_query):
        detector_payload = self._detect_query_needs_image_grounding(original_query)
        detector_yes = detector_payload.get("needs_image_grounding") == "Yes"
        module_trace = {
            "triggered": detector_yes,
            "detector": detector_payload,
            "image_query": "",
            "executed_text_query": original_query,
            "resolved_entity": "",
            "rewritten_query": "",
            "rewrite_confidence": "",
            "appended_doc_count": 0,
        }

        if not detector_yes:
            base_docs = self._search_text_docs(original_query)
            base_content = self._format_retrieval_content(base_docs)
            base_content = self._clip_text(base_content, self.retrieval_char_limit)
            return base_docs, base_content, module_trace

        print("Ambiguous query detected. Start auxiliary image retrieval for rewrite...")
        image_docs, image_retrieval_mode = self._search_auxiliary_image_docs(img)
        module_trace["image_retrieval_mode"] = image_retrieval_mode
        resolver_payload = self._resolve_entity_and_rewrite_query(original_query, detector_payload, image_docs)
        rewritten_query = resolver_payload.get("rewrite_query", "")
        resolved_entity = resolver_payload.get("resolved_entity", "")
        module_trace["resolved_entity"] = resolved_entity
        module_trace["rewritten_query"] = rewritten_query
        module_trace["rewrite_confidence"] = resolver_payload.get("confidence", "")

        if not self._is_valid_rewrite(original_query, rewritten_query, resolved_entity):
            print("Ambiguity rewrite skipped: invalid or non-specific rewrite. Fallback to original text retrieval.")
            base_docs = self._search_text_docs(original_query)
            base_content = self._format_retrieval_content(base_docs)
            base_content = self._clip_text(base_content, self.retrieval_char_limit)
            return base_docs, base_content, module_trace

        rewritten_docs = self._search_text_docs(rewritten_query)
        rewritten_content = self._format_retrieval_content(rewritten_docs)
        rewritten_content = self._clip_text(rewritten_content, self.retrieval_char_limit)
        module_trace["executed_text_query"] = rewritten_query
        module_trace["appended_doc_count"] = len(rewritten_docs) if isinstance(rewritten_docs, list) else int(bool(rewritten_docs))
        return rewritten_docs, rewritten_content, module_trace

    def iterative_infer(self, question, id, image_id):
        self._start_sample_generation_accounting()
        try:
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
                {"role": "user", "content": [
                    {"type": "text", "text": f"Input Question: {question}"},
                    {"type": "image", "image": img}
                ]},
            ]
            trajectory = []
            start_time = time.time()
            response = ""
            try:
                response = self._generate_text(messages)
                response = self._truncate_after_search(response)
                print(f"First Response: {response}")
                self._record_response_actions(trajectory, response)
                messages.append({"role": "assistant", "content": response})
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
                        retrieved_docs, retrieval_content, amb_trace = self._build_augmented_text_retrieval(img, query_txt)
                        query_txt = str(amb_trace.get("executed_text_query", query_txt) or query_txt)
                        print(f"Executed Text Query: {query_txt}")
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
                        "type": "text",
                        "text": self._build_followup_message(retrieval_mode, retrieval_content),
                    })

                    messages.append({"role": "user", "content": contents})

                    try:
                        response = self._generate_text(messages)
                        response = self._truncate_after_search(response)
                        print(f"Response: {response}")
                        self._record_response_actions(trajectory, response)
                        messages.append({"role": "assistant", "content": response})
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
        finally:
            self._finish_sample_generation_accounting()
