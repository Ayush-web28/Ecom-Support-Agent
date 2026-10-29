"""
Conversational Query Rewriting Module.

Takes the customer's raw query and recent dialogue history,
and reformulates follow-up questions containing pronouns (anaphora)
into standalone, unambiguous semantic search queries for vector retrieval.
"""

import os
import time
from typing import List, Union
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

REWRITER_SYSTEM_PROMPT = """You are an expert query reformulation assistant for an e-commerce support search engine.

Given a conversation history between a customer and a support assistant, and the customer's latest follow-up question:
1. If the follow-up question refers to previous context using pronouns (e.g., "it", "that", "this", "they", "those", "does that apply", "how about the fee"), rewrite it into a complete, standalone question that can be understood without the chat history.
2. Include the specific product name, model number, or policy topic mentioned in the prior messages.
3. If the latest question is already completely standalone and needs no context, output it unchanged.
4. Output ONLY the rewritten standalone question and nothing else. Do not add explanations or quotes.

Example 1:
History:
User: What is the return policy for electronics?
Assistant: Electronics must be returned within 14 days in original condition.
User: Does that apply to open-box items too?
Rewritten: Does the 14-day electronics return policy apply to open-box items?

Example 2:
History:
User: How long does the battery last on AeroSound Pro headphones?
Assistant: The AeroSound Pro lasts up to 30 hours with ANC on and 40 hours with ANC off.
User: How long to recharge it from empty?
Rewritten: How long does it take to fully recharge the AeroSound Pro headphones from empty?
"""


def get_rewriter_llm() -> ChatGoogleGenerativeAI:
    """Initializes the Gemini LLM for query reformulation."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing from environment or .env file.")

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.0,
        google_api_key=api_key,
        max_retries=0,
    )


def format_history_for_rewriter(history: Union[List[BaseMessage], List[dict], str]) -> str:
    """Formats diverse message history representations into clean dialogue lines."""
    if isinstance(history, str):
        return history.strip() or "No previous history."

    if not history:
        return "No previous history."

    lines = []
    for msg in history:
        if isinstance(msg, BaseMessage):
            role = "Customer" if isinstance(msg, HumanMessage) else "Assistant"
            lines.append(f"{role}: {msg.content}")
        elif isinstance(msg, dict):
            role = "Customer" if msg.get("role") in ["user", "human"] else "Assistant"
            lines.append(f"{role}: {msg.get('content', '')}")

    return "\n".join(lines)


def rewrite_query(query: str, chat_history: Union[List[BaseMessage], List[dict], str] = "") -> str:
    """
    Rewrites a contextual follow-up query into a standalone search query.

    Args:
        query: The customer's current question.
        chat_history: The recent conversation history.

    Returns:
        str: The standalone reformulated search query.
    """
    # If there is no conversation history, query is already standalone
    formatted_history = format_history_for_rewriter(chat_history)
    if formatted_history == "No previous history." or not formatted_history.strip():
        return query.strip()

    llm = get_rewriter_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", REWRITER_SYSTEM_PROMPT),
        ("human", "Conversation History:\n{history}\n\nLatest Customer Question: {query}\n\nRewritten Standalone Question:"),
    ])

    chain = prompt | llm

    response = None
    for attempt in range(5):
        try:
            response = chain.invoke({
                "history": formatted_history,
                "query": query,
            })
            break
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "RateLimitError" in type(e).__name__:
                time.sleep(2.0 * (attempt + 1))
            else:
                raise e

    if response is None:
        response = chain.invoke({
            "history": formatted_history,
            "query": query,
        })

    content = response.content
    if isinstance(content, list):
        raw_text = "".join([p.get("text", "") if isinstance(p, dict) else str(p) for p in content])
    else:
        raw_text = str(content)

    rewritten = raw_text.strip().strip('"').strip("'")
    return rewritten if rewritten else query
