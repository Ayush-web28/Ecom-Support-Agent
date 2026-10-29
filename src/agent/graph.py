"""
LangGraph State Machine Assembly for the Customer Support Agent.
Wires together nodes, conditional routing edges, and persistent checkpointer memory.
"""

from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.agent.state import AgentState
from src.agent.nodes import (
    classify_node,
    greeting_node,
    escalate_node,
    rewrite_node,
    retrieve_node,
    grade_documents_node,
    generate_node,
    rephrase_query_node,
)


# =====================================================================
# CONDITIONAL ROUTING FUNCTIONS
# =====================================================================
def route_after_classification(state: AgentState) -> Literal["greeting_node", "escalate_node", "rewrite_node"]:
    """Routes based on the predicted intent category."""
    category = state.get("category")
    if category == "greeting":
        return "greeting_node"
    elif category == "escalate":
        return "escalate_node"
    else:
        # Both "policy" and "manual" require contextual rewriting & RAG
        return "rewrite_node"


def route_after_grading(state: AgentState) -> Literal["generate_node", "rephrase_query_node", "escalate_node"]:
    """
    Corrective RAG (CRAG) Routing:
    - If documents are relevant -> Proceed to grounded generation.
    - If documents are irrelevant and retry_count < 1 -> Rephrase query and try retrieval again.
    - If retry failed -> Escalate to human support gracefully.
    """
    if state.get("has_relevant_docs", False):
        return "generate_node"

    retry_count = state.get("retry_count", 0)
    if retry_count < 1:
        return "rephrase_query_node"

    # Retries exhausted and no good documents found -> Escalate
    return "escalate_node"


# =====================================================================
# GRAPH BUILDER
# =====================================================================
def build_support_graph(checkpointer=None):
    """
    Constructs and compiles the complete customer support LangGraph workflow.
    """
    workflow = StateGraph(AgentState)

    # 1. Add all functional nodes
    workflow.add_node("classify_node", classify_node)
    workflow.add_node("greeting_node", greeting_node)
    workflow.add_node("escalate_node", escalate_node)
    workflow.add_node("rewrite_node", rewrite_node)
    workflow.add_node("retrieve_node", retrieve_node)
    workflow.add_node("grade_documents_node", grade_documents_node)
    workflow.add_node("generate_node", generate_node)
    workflow.add_node("rephrase_query_node", rephrase_query_node)

    # 2. Add edges from START
    workflow.add_edge(START, "classify_node")

    # 3. Add conditional routing from classify_node
    workflow.add_conditional_edges(
        "classify_node",
        route_after_classification,
        {
            "greeting_node": "greeting_node",
            "escalate_node": "escalate_node",
            "rewrite_node": "rewrite_node",
        }
    )

    # Direct edges for terminal leaf nodes
    workflow.add_edge("greeting_node", END)
    workflow.add_edge("escalate_node", END)

    # Retrieval pipeline sequence
    workflow.add_edge("rewrite_node", "retrieve_node")
    workflow.add_edge("retrieve_node", "grade_documents_node")

    # Corrective RAG routing from grade_documents_node
    workflow.add_conditional_edges(
        "grade_documents_node",
        route_after_grading,
        {
            "generate_node": "generate_node",
            "rephrase_query_node": "rephrase_query_node",
            "escalate_node": "escalate_node",
        }
    )

    # Rephrase loop back to retrieve
    workflow.add_edge("rephrase_query_node", "retrieve_node")

    # Answer generation to END
    workflow.add_edge("generate_node", END)

    # Compile with memory checkpointer
    if checkpointer is None:
        checkpointer = MemorySaver()

    app = workflow.compile(checkpointer=checkpointer)
    return app
