"""
LangGraph Nodes for Intelligent Customer Support Agent.
Implements classification, query rewriting, retrieval, document grading (CRAG),
grounded answer generation, and human support escalation.
"""

import os
import time
import random
from typing import Dict, Any, List
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from src.agent.state import AgentState
from src.agent.router import classify_intent
from src.agent.rewriter import rewrite_query
from src.rag.vector_store import search_documents

load_dotenv()


def get_llm(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    """Returns Gemini LLM instance with retry parameters."""
    api_key = os.getenv("GEMINI_API_KEY")
    return ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        temperature=temperature,
        google_api_key=api_key,
        max_retries=6,
    )


def safe_invoke(runnable: Any, input_data: Any, max_attempts: int = 5) -> Any:
    """Invokes a LangChain runnable with exponential backoff on 429 rate limits."""
    for attempt in range(max_attempts):
        try:
            return runnable.invoke(input_data)
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "RateLimitError" in type(e).__name__:
                wait_time = 2.0 * (attempt + 1)
                time.sleep(wait_time)
            else:
                raise e
    return runnable.invoke(input_data)


def extract_text_content(content: Any) -> str:
    """Safely extracts text string from LLM message content across different formats."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in content]
        return "".join(texts).strip()
    return str(content).strip()


# =====================================================================
# NODE 1: INTENT CLASSIFICATION
# =====================================================================
def classify_node(state: AgentState) -> Dict[str, Any]:
    """Classifies user intent using the Pydantic router."""
    query = state["current_query"]

    # Build concise history string
    history_lines = []
    for msg in state.get("messages", []):
        role = "Customer" if isinstance(msg, HumanMessage) else "Assistant"
        history_lines.append(f"{role}: {msg.content}")
    history_text = "\n".join(history_lines[-6:]) if history_lines else ""

    decision = classify_intent(query, chat_history=history_text)

    is_escalate = decision.category == "escalate"
    escalate_reason = decision.reasoning if is_escalate else None

    return {
        "category": decision.category,
        "confidence": decision.confidence,
        "escalate_to_human": is_escalate,
        "escalation_reason": escalate_reason,
        "retry_count": state.get("retry_count", 0),
    }


# =====================================================================
# NODE 2: GREETING HANDLER
# =====================================================================
def greeting_node(state: AgentState) -> Dict[str, Any]:
    """Generates a warm, direct greeting without accessing vector storage."""
    query = state["current_query"]
    llm = get_llm(temperature=0.7)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI customer support assistant for an online electronics and lifestyle store. "
                   "Provide a warm, welcoming, and concise response to the customer's greeting, and ask how you can help them today with our products, shipping, returns, or warranty."),
        ("human", "{query}"),
    ])

    response = safe_invoke(prompt | llm, {"query": query})
    answer = extract_text_content(response.content)

    return {
        "generation": answer,
        "messages": [AIMessage(content=answer)],
    }


# =====================================================================
# NODE 3: HUMAN ESCALATION HANDOFF
# =====================================================================
def escalate_node(state: AgentState) -> Dict[str, Any]:
    """
    Creates a simulated support ticket and hands off the conversation to a human agent.
    """
    ticket_id = f"TICKET-{random.randint(10000, 99999)}"
    reason = state.get("escalation_reason") or "Customer requested human support or inquiry could not be resolved automatically."

    response_text = (
        f"I understand your concern and have escalated this matter to our live support team.\n\n"
        f"📌 **Support Ticket Created:** `#{ticket_id}`\n"
        f"**Reason:** {reason}\n\n"
        f"A human support representative has received your conversation history and will assist you shortly. "
        f"Thank you for your patience!"
    )

    return {
        "generation": response_text,
        "escalate_to_human": True,
        "messages": [AIMessage(content=response_text)],
    }


# =====================================================================
# NODE 4: CONTEXTUAL QUERY REWRITER
# =====================================================================
def rewrite_node(state: AgentState) -> Dict[str, Any]:
    """Rewrites follow-up questions into standalone queries using conversation history."""
    query = state["current_query"]
    messages = state.get("messages", [])

    standalone = rewrite_query(query, chat_history=messages)

    return {
        "standalone_query": standalone,
    }


# =====================================================================
# NODE 5: VECTOR RETRIEVAL
# =====================================================================
def retrieve_node(state: AgentState) -> Dict[str, Any]:
    """Retrieves relevant chunks from ChromaDB filtered by category."""
    query = state.get("standalone_query") or state["current_query"]
    category = state.get("category", "policy")

    # Map category to doc_type filter
    doc_type = category if category in ["policy", "manual"] else None

    docs = search_documents(query=query, doc_type=doc_type, top_k=3)

    return {
        "documents": docs,
    }


# =====================================================================
# NODE 6: DOCUMENT RELEVANCE GRADER (CRAG PATTERN)
# =====================================================================
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    is_relevant: bool = Field(
        description="Whether the document contains information relevant to answering the question."
    )
    explanation: str = Field(
        description="Brief explanation of why the document is or is not relevant."
    )


def grade_documents_node(state: AgentState) -> Dict[str, Any]:
    """
    Evaluates whether the retrieved documents actually contain relevant information.
    If no documents are relevant, triggers query rephrasing or fallback escalation.
    """
    query = state.get("standalone_query") or state["current_query"]
    docs = state.get("documents", [])

    if not docs:
        return {"has_relevant_docs": False}

    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(GradeDocuments)

    grader_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an evaluator assessing whether a retrieved document snippet contains facts relevant "
                   "to answering the user question. If the snippet contains relevant information or addresses the topic, grade is_relevant as True, else False."),
        ("human", "User Question: {question}\n\nRetrieved Document Content:\n{document}"),
    ])

    chain = grader_prompt | structured_llm

    # Combine document snippets
    combined_docs = "\n\n---\n\n".join([d.page_content for d in docs])
    grade = safe_invoke(chain, {"question": query, "document": combined_docs})

    return {
        "has_relevant_docs": grade.is_relevant,
    }


# =====================================================================
# NODE 7: GROUNDED ANSWER GENERATION
# =====================================================================
def generate_node(state: AgentState) -> Dict[str, Any]:
    """
    Generates a concise, helpful answer strictly grounded in the retrieved documents,
    complete with document citations.
    """
    query = state.get("standalone_query") or state["current_query"]
    docs = state.get("documents", [])

    # Format context with source citations
    context_blocks = []
    sources = set()
    for doc in docs:
        src = doc.metadata.get("source", "knowledge_base")
        sources.add(src)
        context_blocks.append(f"[{src}]\n{doc.page_content}")

    context_str = "\n\n".join(context_blocks)

    llm = get_llm(temperature=0.2)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful and polite customer support agent for our online store.
Answer the customer's question based strictly on the provided context below.

Rules:
1. Ground your answer ONLY in the provided context. Do NOT fabricate rules or specifications.
2. Cite the source document at the end of your response (e.g., "[Source: return_and_refund.md]").
3. Keep the tone courteous, clear, and professional.
4. If the context does not fully answer the question, state what is known and offer further assistance.

Context:
{context}
"""),
        ("human", "{question}"),
    ])

    response = safe_invoke(prompt | llm, {
        "context": context_str,
        "question": query,
    })

    answer = extract_text_content(response.content)

    return {
        "generation": answer,
        "messages": [AIMessage(content=answer)],
    }


# =====================================================================
# NODE 8: QUERY REPHRASER (FALLBACK RETRY)
# =====================================================================
def rephrase_query_node(state: AgentState) -> Dict[str, Any]:
    """
    Rephrases the search query to be broader if initial retrieval failed.
    """
    query = state.get("standalone_query") or state["current_query"]
    retry_count = state.get("retry_count", 0) + 1

    llm = get_llm(temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a query optimizer. The previous search query failed to retrieve relevant documents. "
                   "Rephrase the question to use broader, more general keywords suited for e-commerce search."),
        ("human", "Original Question: {query}\n\nBroader Search Query:"),
    ])

    response = safe_invoke(prompt | llm, {"query": query})
    new_query = extract_text_content(response.content).strip('"').strip("'")

    return {
        "standalone_query": new_query,
        "retry_count": retry_count,
    }
