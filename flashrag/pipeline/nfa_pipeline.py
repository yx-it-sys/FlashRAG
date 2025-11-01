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

class Assessment(BaseModel):
    assessment: str = Field(
         description="The assessment of the documents, must be one of 'good', 'insufficient', or 'irrelevant'."
    )

class AutomatonState(TypedDict):
    initial_query: str
    plan: str
    documents: List[str]
    assessment: str
    generation: str
    if_final: bool
    messages: Annotated[List[BaseMessage], operator.add]

class NFAPipeline(BasicMultiModalPipeline):
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        super().__init__(config, prompt_template)
        
        self.config = config
        self.generator = get_generator(config) if generator is None else generator
        self.retriever = get_retriever(config) if retriever is None else retriever

        self.prompt_template = prompt_template

        self.workflow = StateGraph(AutomatonState)
        self._build_graph()
        self.app = self.workflow.compile()
        print("自动机结构图:")
        self.app.get_graph().print_ascii()

    def _build_graph(self):
        self.workflow.add_node("plan", self._plan_node)
        self.workflow.add_node("retrieval", self._retrieval_node)
        self.workflow.add_node("assess", self._assess_node)
        self.workflow.add_node("generate", self._generate_node)

        self.workflow.set_entry_point("plan")
        self.workflow.add_edge("plan", "retrieval")
        self.workflow.add_edge("retrieval", "assess")

        self.workflow.add_conditional_edges(
            "assess",
            self._assess_router,
            {
                "retrieval": "retrieval",
                "generate": "generate"
            }
        )

        self.workflow.add_conditional_edges(
            "generate",
            self._generate_router,
            {
                "plan": "plan",
                END: END
            }
        )

    def _plan_node(self, state: AutomatonState) -> dict:
        if not state.get("messages"):
            initial_message = self.prompt_template.get_string(state['initial_query'])
        else:
            initial_message = []

        response_dict = self.generator.generate(state['initial_query'], type="plan")[0]
        current_plan = response_dict["output_text"][0]
        plan_message = AIMessage(content=f"Plan: I need to find out '{current_plan}'.")
        return {"plan": current_plan, "messages": initial_message + [plan_message]}

    def _retrieval_node(self, state: AutomatonState) ->dict:
        retrieved_docs = self.retriever.search(state["plan"], 2)[0]
        docs_as_string = "\n".join([json.dumps(doc, indent=2) for doc in retrieved_docs])
        retrieval_message = ToolMessage(
            content=f"Retrieved {len(retrieved_docs)} documents:\n{docs_as_string}",
            tool_call_id="retrieval_tool"
        )
        return {"documents": retrieved_docs, "messages": [retrieval_message]}

    def _format_docs_for_prompt(self, subquery: str, docs: List[dict]) -> str:
        formatted_list = []
        for i, doc in enumerate(docs):
            content = doc.get('contents')
            formatted_list.append(f"Document {i+1}:\n{content}")
        return f"{subquery}\n\n---\n\n".join(formatted_list)
    
    def _assess_node(self, state: AutomatonState) ->dict:
        current_plan = state['plan']
        documents = state['documents']

        formatted_docs = self._format_docs_for_prompt(current_plan, documents)
        decision = self.generator.generate([formatted_docs], type="assess")
        if decision not in ['relevant', 'irrelevant']:
            warnings.warn(f"The assessment model does not generate expected result (relevant, irrelevant): {decision}, set to default (irrelevant).")
            decision = 'irrelevant'
        assessment_message = AIMessage(content=f"Assessment: The documents are '{decision}'.")
        
        return {
            "assessment": decision,
            "messages": [assessment_message]
        }
    
    def _generate_node(self, state: AutomatonState) -> dict:
        documents = state['documents']
        current_plan = state['plan']
        context = state.get('messages', [])
        formatted_docs = self._format_docs_for_prompt(current_plan)
        response = self.generator.generate(formatted_docs, type="answer")[0]
        generated_answer = response["output_text"][0]
        # 解析其中是否出现"Final Answer" or "<Final Answer>"
        if re.search(r"<Final Answer>|Final Answer:", generated_answer, re.IGNORECASE):
            task_is_final = True
        else:
            task_is_final = False
        generation_message = AIMessage(content=generated_answer)
        return {"generation": generated_answer, "is_final": task_is_final, "messages": [generation_message]}

    def _assess_router(self, state: AutomatonState) -> str:
        if state["assessment"] == 'continue':
            return "generate"
        else:
            state['assessment'] = 'continue'
            return 'retrieval'
        
    def _generate_router(self, state: AutomatonState) -> str:
        if state['id_final']:
            return END
        else:
            return "plan"

    def run(self, dataset):
        data_items_list = []
        for item in dataset:
            data_items_list.append({
                "id": item.id,
                "question": item.question,
                "answers": item.golden_answers[0],
                "input_prompt": self.prompt_template.get_string(item, self.config)
                }
            )

        for input_prompt in data_items_list:
            final_state = self.app.invoke(input_prompt)
            print(f"Final State: {final_state}")



        