from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class TemplateType(str, Enum):
    SINGLE_HOP = "Single-hop"
    MULTI_HOP = "Multi-hop"
    COMPARISON = "Selection"
    SELECTION = "Selection"
    VERIFICATION = "verification"
    INTERSECTION = "Intersection"

class OpType(str, Enum):
    SEARCH = "SEARCH"
    LOOKUP = "LOOKUP"
    FILTER = "FILTER"
    VERIFY = "VERIFY"
    COMPARE = "COMPARE"


class PlanNode(BaseModel):
    id: str
    op: OpType
    inputs: Dict[str, str]
    description: str = Field(..., description="Human-readable description for debugging/logging")
    output_desc: str

class NeuroSymbolicTemplete(BaseModel):
    # Meta data. For Router and Classifier
    type: TemplateType
    description: str

    # Symbol Table Interface
    required_slots: List[str]

    # Formal logic (for execution engines/code)
    dag_structure = List[PlanNode]

    # Few shot examples
    few_shot_examples: List[Dict[str, Any]]

