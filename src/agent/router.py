"""
Intent Classification & Routing Module (The Agent's Front Brain).

Uses Pydantic and Google Gemini to reliably categorize customer queries
into discrete routes:
  1. greeting  -> Quick polite greeting / conversational chit-chat (no RAG needed)
  2. policy    -> Questions on returns, refunds, shipping, warranty terms
  3. manual    -> Questions on product specifications, setup, troubleshooting
  4. escalate  -> Angry customers, billing disputes, requests for human agent
"""

import os
import time
from typing import Literal, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


# =====================================================================
# TASK 1: PYDANTIC SCHEMA DEFINITION
# =====================================================================
# TODO 1 [YOUR TURN]:
# Define the fields for RouteQuery inheriting from Pydantic's BaseModel.
# It enforces that the LLM MUST return a valid structured object with:
#   - category: Literal["greeting", "policy", "manual", "escalate"]
#   - confidence: float (between 0.0 and 1.0)
#   - reasoning: str (brief 1-sentence reason)
# =====================================================================

class RouteQuery(BaseModel):
    category: Literal["greeting", "policy", "manual", "escalate"] = Field(
        description="The predicted category for the customer's query."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 (uncertain) and 1.0 (certain)."
    )
    reasoning: str = Field(
        description="A brief explanation of why this category was selected."
    )


# =====================================================================
# TASK 2: ROUTER PROMPT ENGINEERING
# =====================================================================
# TODO 2 [YOUR TURN]:
# Write the classification system instructions.
# Define what qualifies as greeting, policy, manual, and escalate.
# =====================================================================

ROUTER_SYSTEM_PROMPT = """You are an expert customer intent classifier for an e-commerce support system.
Analyze the customer's input and classify it into EXACTLY ONE of the following 4 categories:
1. "greeting":
   - The user is saying hello, goodbye, thank you, or general pleasantries.
   - Examples: "Hi", "Good morning", "Thanks for your help!", "Have a great day."
2. "policy":
   - Inquiries about store rules and customer policies, including:
     * Return windows and conditions (e.g. open-box rules, restocking fees).
     * Shipping costs, delivery timelines, and carrier tracking.
     * Warranty coverage periods and claim procedures.
     * Refund timelines and payment methods.
   - Examples: "Can I return opened headphones?", "How much is express shipping?", "Is battery wear covered under warranty?"
3. "manual":
   - Technical product inquiries and troubleshooting from user manuals, including:
     * Product specifications, dimensions, wattage, and battery life.
     * Bluetooth pairing, multipoint connections, and app setup.
     * Hardware controls, touch sliders, and button combinations.
     * Troubleshooting steps and factory reset procedures.
   - Examples: "How do I factory reset my headphones?", "Does the lamp support 5GHz Wi-Fi?", "How many lumens is the desk lamp?"
4. "escalate":
   - Situations requiring immediate human support:
     * The customer expresses severe frustration, anger, or accusations of fraud.
     * Financial disputes (e.g. double charged, stolen funds).
     * The customer explicitly asks to speak to a real human, manager, or supervisor.
   - Examples: "You charged my card twice!", "Let me speak to a real person right now!", "This is ridiculous, get me a manager."
Be objective. Output the selected category, a calibrated confidence score between 0.0 and 1.0, and a concise 1-sentence reasoning.
"""


def get_router_llm() -> ChatGoogleGenerativeAI:
    """Initializes the Gemini LLM for classification with temperature 0.0."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing from environment or .env file.")

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.0,  # 0.0 ensures deterministic, consistent classification
        google_api_key=api_key,
        max_retries=0,
    )


# =====================================================================
# TASK 3: STRUCTURED CLASSIFIER INVOCATION
# =====================================================================
# TODO 3 [YOUR TURN]:
# 1. Bind the Pydantic schema using: llm.with_structured_output(RouteQuery)
# 2. Build the ChatPromptTemplate with ROUTER_SYSTEM_PROMPT and human query.
# 3. Create the chain (prompt | structured_llm) and invoke it.
# 4. Return the resulting RouteQuery object.
# =====================================================================

def classify_intent(query: str, chat_history: Optional[str] = None) -> RouteQuery:
    """
    Classifies a customer query into one of the 4 defined categories.

    Args:
        query: The customer's message.
        chat_history: Optional summary of prior messages for multi-turn context.

    Returns:
        RouteQuery: Pydantic object with category, confidence, and reasoning.
    """
    llm = get_router_llm()

    structured_llm = llm.with_structured_output(RouteQuery)
    prompt = ChatPromptTemplate.from_messages([
        ("system", ROUTER_SYSTEM_PROMPT),
        ("human", "Customer Message: {query}\n\nRecent History Context: {history}"),
    ])
    chain = prompt | structured_llm

    try:
        return chain.invoke({
            "query": query,
            "history": chat_history or "No previous history."
        })
    except Exception as e:
        err_msg = str(e)
        if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "RateLimitError" in type(e).__name__:
            # Re-raise cleanly so the application informs the user immediately
            raise e
        raise e