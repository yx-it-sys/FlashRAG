from flashrag.utils import get_retriever
import tomllib
from typing import List
from utils import general_generate, agentic_search
from functools import partial
from transformers import pipeline

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

        search_func = partial(agentic_search, self.retriever, self.extractor, self.top_k, self.ret_thresh)
        self.context_vars = {
            "search": search_func,
        }
        with open('prompts/reasoning.toml', 'rb') as f:
            self.prompt = tomllib.load(f)
    
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



