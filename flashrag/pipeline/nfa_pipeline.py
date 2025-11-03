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
        user_prompt = self.task_prompts["plan"].format(query=run_state["current_query"])
        response_dict = self.generator.generate()
        return "S_Plan"