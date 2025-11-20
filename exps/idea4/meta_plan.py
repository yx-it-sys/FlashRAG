import outlines
from pydantic import BaseModel, Field
from typing import List
import tomllib

class Constraint(BaseModel):
    text: str = Field(..., description="Text segment from query representing know facts")
    type: str = Field(..., description="Entity type: Person, Org, Time, etc.")
class Variable(BaseModel):
    target_role: str = Field(..., description="The target role")
    attributes: List[str] = Field(default=[], description="Adjectives modifying the target")
    dependency: str = Field(..., description="Relation to constraints")


class QuerySymbolTable(BaseModel):
    constraints: List[Constraint]
    target_variables: List[Variable]
    initent: str = Field(..., description="Retrieval, Comparison, Judgement or  Aggregation")
class SyntaxTreeGenerator():
    def __init__(self, model, prompt_path):
        self.model = outlines.models.transformers(
            model,
            device="cuda",
            model_kwargs = {
                "load_in_4bit": True,
                "trust_remote_code": True
            }
        )

        self.generator = outlines.generate.json(self.model, QuerySymbolTable)

        with open(prompt_path, "rb") as f:
            self.prompt = tomllib.load(f)
    
    def generate(self, query: str):
        messages = [
            {
                "role": "system",
                "content": self.prompt["system_prompt"]
            },
            {
                "role": "user",
                "content": self.prompt["user_prompt"].format(query=query)
            }
        ]

        inputs = self.model.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        result = self.generator(inputs)
        return result
    