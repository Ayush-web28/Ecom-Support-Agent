"""
Agent State Definition for LangGraph.
Tracks conversational state, retrieved context, routing categories, and escalation status.
"""

from typing import Annotated, List, Literal, Optional, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    State tracked across all nodes in the customer support LangGraph workflow.
    """
    # Conversational messages with LangGraph's add_messages reducer
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Current raw user query
    current_query: str

    # Contextual query rewritten to be standalone (from Phase 4)
    standalone_query: str

    # Intent category determined by router (from Phase 3)
    category: Literal["greeting", "policy", "manual", "escalate"]

    # Router confidence score
    confidence: float

    # Documents retrieved from ChromaDB (from Phase 2)
    documents: List[Document]

    # Evaluated document relevance flag
    has_relevant_docs: bool

    # Retry counter to prevent infinite search loops
    retry_count: int

    # Final generated answer
    generation: str

    # Human-in-the-loop escalation flags
    escalate_to_human: bool
    escalation_reason: Optional[str]
