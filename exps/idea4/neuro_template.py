from meta_template import NeuroSymbolicTemplete, TemplateType, PlanNode, OpType

MULTIHOP = NeuroSymbolicTemplete(
    type=TemplateType.MULTI_HOP,
    description="Handles multi-hop queries",

    # Symbol Table Interface
    required_slots = ["anchor_entity", "missing_attribute", "result_filter"],
    
    # Formal logic
    dag_structure = [
        PlanNode(
            id="step_1",
            op=OpType.SEARCH,
            inputs={"query": "The $missing_attribute of $anchor_entity"},
            description="Identify the bridge entity"
        ),
        PlanNode(
            id="step_2",
            op=OpType.SEARCH,
            inputs={"query": ""},
            description="Retrieve details of the bridge entity"
        ),
        PlanNode(
            id="step_3",
            op=OpType.FILTER,
            inputs={"candidates": "@step_2", "condition": "$filter_condiction"},
            description="Filter candidates based on user constraints"
        )
    ],

    few_shot_examples=[
        {
            "query": "My Neighbor Totoro was produced by a Japanese animation film studio founded in what year?",
            "symbol_table": {
                "start_node": ""
            }
        }
    ]

)