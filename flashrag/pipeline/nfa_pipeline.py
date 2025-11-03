from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langchain_core.pydantic_v1 import BaseModel, Field
from flashrag.pipeline import BasicMultiModalPipeline
from flashrag.utils import get_retriever, get_generator
import operator
import re
import torch
import os
import json
import warnings

class NFAPipeline(BasicMultiModalPipeline):
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        super().__init__(config, prompt_template)
        
        self.config = config
        self.generator = get_generator(config) if generator is None else generator
        self.retriever = get_retriever(config) if retriever is None else retriever

        self.prompt_template = prompt_template

        self.current_state = "S_Initial"

    def _state_initial(self) -> str:
        return "S_Plan"
    
    def _state_plan(self) -> str:
        response = self.generator()