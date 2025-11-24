from flashrag.utils import get_retriever
import tomllib
from typing import List
from utils import general_generate, parse_json
from functools import partial
from transformers import pipeline
import nltk
import re

class RAGPipeline():
    def __init__(self, config, model, tokenizer, ret_thresh, extractor=None, retriever=None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        if extractor is None:
            extractor = pipeline("question-answering", model="ddeepset/deberta-v3-base-squad2")
        self.extractor = extractor
        if retriever is None:
            retriever = get_retriever(self.config)

        self.retriever = retriever
        self.top_k = 5
        self.ret_thresh = ret_thresh

        search_func = partial(self.agentic_search, self.retriever, self.extractor, self.top_k, self.ret_thresh)
        self.context_vars = {
            "search": search_func,
        }
        with open('prompts/reasoning.toml', 'rb') as f:
            self.prompt = tomllib.load(f)
    
    def agentic_search(self, intent: str, entity: List[str], constraints: List[str]) -> str:
        retrieved_intent_content, intent_content_scores = self.retriever.search(query=intent, num=self.top_k, return_score=True)
        intent_docs = []

        for doc, score in zip(retrieved_intent_content, intent_content_scores):
            if score >= self.ret_thresh:
                intent_docs.append(doc['contents'])
        
        relevant_sentences = []
        for i, doc in enumerate(intent_docs):
            sentences = nltk.sent_tokenize(doc)
            filtered_sentences = []
            for sent in sentences:
                result = self.extractor(question=intent, context=sent)
                if result['score'] > 0.5:
                    filtered_sentences.append(sent)
            relevant_sentences.append(f"Hint {i+1}:\n{'\n'.join(filtered_sentences)}")
        intent_docs = "\n\n".join(relevant_sentences)

        background_knowledge = []
        for entity in entity:
            retrieved_docs = []
            retrieved_entity_content, entiity_content_scores = self.retriever.search(query=entity, num=self.top_k, return_score=True)
            for doc, score in zip(retrieved_entity_content, entiity_content_scores):
                if score >= self.ret_thresh:
                    retrieved_docs.append(doc['contents'])
            background_knowledge.append({entity: retrieved_docs})
            
        sections = ["=== Background Knowledge for Entities ==="]
        for item in background_knowledge:
            for entity, docs in item.items():
                if not docs:
                    continue
                sections.append(f"\n## Introduction to: {entity}")
                
                for i, doc in enumerate(docs, 1):
                    clean_doc = doc.strip().replace('\n', ' ')
                    sections.append(f"{i}. {clean_doc}")
            
        background_knowledge = "\n".join(sections)

        final_context = f"=== Intent Related Information ===\n{intent_docs}\n\n{background_knowledge}"

        # LLM 登场
        # 首先根据constraints对final_context进行过滤和调整
        messages = [
            {"role": "system", "content": self.prompt['system_prompt']},
            {"role": "user", "content": self.prompt['user_prompt'].format(intention=intent, constraints='\n'.join(constraints), context=final_context)}
        ]
        response = general_generate(messages, self.model, self.tokenizer)
        json_text = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        json_dict = parse_json(json_text)
        if json_dict is None:
            print("I can't answer! json_dict is None.")
        return json_dict.get('final_answer', '')
        # 最终生成当前Plan的结果
        # 内部抛出异常时，外部捕获

    def run(self, draft_plan: List[str], user_query: str, context: str):
        for plan in draft_plan:
            try:
                exec(plan, {}, self.context_vars)
            except Exception as e:
                print(f"Error executing plan line: {plan}\nError: {e}")
                break
            
        # messages = [
        #     {"role": "system", "content": self.prompt['system_prompt']},
        #     {"role": "user", "content": self.prompt['user_prompt'].format(user_query=user_query, draft_plan=draft_plan, context=context)}
        # ]
        # response = general_generate(messages, self.model, self.tokenizer)
        # reasoning_dict = self.parse_reasoning(response)
        # action_type = reasoning_dict.get('action_type', '').lower()
        # current_step = reasoning_dict.get('current_step', '')
        # action_content = reasoning_dict.get('action_content', '')

        # if action_type == "search":
        #     retrieved_content, scores = self.retriever.search(query=action_content, num=self.top_k, return_score=True)
        #     retrieved_docs = []

        #     for doc, score in zip(retrieved_content, scores):
        #         if score >= self.ret_thresh:
        #             retrieved_docs.append(doc['contents'])
            
        #     shaped_docs = self.reshape(retrieved_docs)

        # elif



