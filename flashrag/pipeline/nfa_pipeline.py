from typing import TypedDict, Annotated, List
from flashrag.prompt import MMPromptTemplate
from flashrag.pipeline import BasicMultiModalPipeline
from flashrag.utils import get_retriever, get_generator
import operator
import re
import torch
import os
import json
import warnings
import tomllib

class NFAPipeline(BasicMultiModalPipeline):
    def __init__(self, config, prompt_path: str, prompt_template=None, retriever=None, generator=None):
        super().__init__(config, prompt_template)
        
        self.config = config
        self.generator = get_generator(config) if generator is None else generator
        self.retriever = get_retriever(config) if retriever is None else retriever

        if prompt_template is None:
            prompt_template = MMPromptTemplate(config)
        self.prompt_template = prompt_template

    def _state_initial(self, run_state: dict) -> str:
        return "S_Plan"
    
    def _state_plan(self, run_state: dict) -> str:
        response_dict = self.generator.generate(run_state["prompts_with_query"])
        response = response_dict["output_text"][0]
        sub_question, need_search = self._parse_plan(response)

        run_state["plan"] = {"sub_question": sub_question, "need_search": need_search}
        
        if need_search:
            return "S_Retrieve"
        else:
            return "S_generate"
    
    def _state_retrieve(self, run_state: dict) -> str:
        query = run_state["plan"]["sub_question"]

        search_text = self.retriever.search(query, 2)
        search_text = search_text[0]["contents"]