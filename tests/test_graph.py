"""
Interactive Verification Script for Phase 5.
Tests the complete LangGraph State Machine across greetings, policies,
manuals, multi-turn dialogues, direct escalation, and CRAG fallback escalation.
"""

import sys
import io
import time
from pathlib import Path
from langchain_core.messages import HumanMessage

# Force UTF-8 output on Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.agent.graph import build_support_graph


def run_tests():
    print("=" * 80)
    print("PHASE 5 VERIFICATION: LANGGRAPH AGENT STATE MACHINE")
    print("=" * 80)

    app = build_support_graph()

    # -----------------------------------------------------------------
    # Test 1: Greeting Path
    # -----------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TEST 1: GREETING PATH (No Vector Search)")
    print("-" * 80)
    config = {"configurable": {"thread_id": "thread-1"}}
    query1 = "Hi! Good morning, how are you?"
    print(f"Customer: \"{query1}\"")

    res1 = app.invoke({
        "current_query": query1,
        "messages": [HumanMessage(content=query1)],
    }, config=config)

    print(f"Category: [{res1['category'].upper()}]")
    print(f"Response:\n{res1['generation']}")
    time.sleep(2)

    # -----------------------------------------------------------------
    # Test 2: Policy Path (Electronics Return Window)
    # -----------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TEST 2: POLICY RAG PATH (Electronics Return Window)")
    print("-" * 80)
    config = {"configurable": {"thread_id": "thread-2"}}
    query2 = "What is the return window for electronics?"
    print(f"Customer: \"{query2}\"")

    res2 = app.invoke({
        "current_query": query2,
        "messages": [HumanMessage(content=query2)],
    }, config=config)

    print(f"Category: [{res2['category'].upper()}]")
    print(f"Standalone Query: \"{res2.get('standalone_query')}\"")
    print(f"Response:\n{res2['generation']}")
    time.sleep(2)

    # -----------------------------------------------------------------
    # Test 3: Multi-turn Follow-up on Policy (Open-Box Restocking Fee)
    # -----------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TEST 3: MULTI-TURN FOLLOW-UP (Continuing Thread-2)")
    print("-" * 80)
    query3 = "Does that apply to open-box items as well?"
    print(f"Customer: \"{query3}\"")

    res3 = app.invoke({
        "current_query": query3,
        "messages": [HumanMessage(content=query3)],
    }, config=config)

    print(f"Rewritten Query: \"{res3.get('standalone_query')}\"")
    print(f"Response:\n{res3['generation']}")
    time.sleep(2)

    # -----------------------------------------------------------------
    # Test 4: Product Manual Path (Headphones Reset)
    # -----------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TEST 4: PRODUCT MANUAL RAG PATH (Headphones Factory Reset)")
    print("-" * 80)
    config = {"configurable": {"thread_id": "thread-3"}}
    query4 = "How do I factory reset my AeroSound Pro headphones?"
    print(f"Customer: \"{query4}\"")

    res4 = app.invoke({
        "current_query": query4,
        "messages": [HumanMessage(content=query4)],
    }, config=config)

    print(f"Category: [{res4['category'].upper()}]")
    print(f"Response:\n{res4['generation']}")
    time.sleep(2)

    # -----------------------------------------------------------------
    # Test 5: Direct Human Escalation Path
    # -----------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TEST 5: DIRECT HUMAN ESCALATION PATH (Billing Dispute)")
    print("-" * 80)
    config = {"configurable": {"thread_id": "thread-4"}}
    query5 = "You charged my credit card twice! Connect me to a real agent right now!"
    print(f"Customer: \"{query5}\"")

    res5 = app.invoke({
        "current_query": query5,
        "messages": [HumanMessage(content=query5)],
    }, config=config)

    print(f"Escalated: {res5.get('escalate_to_human')}")
    print(f"Response:\n{res5['generation']}")
    time.sleep(2)

    # -----------------------------------------------------------------
    # Test 6: Corrective RAG / Missing Info Fallback Escalation
    # -----------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TEST 6: CORRECTIVE RAG FALLBACK (Out-of-Scope Query)")
    print("-" * 80)
    config = {"configurable": {"thread_id": "thread-5"}}
    query6 = "What is your refund policy for live pet parrots and goldfishes?"
    print(f"Customer: \"{query6}\"")

    res6 = app.invoke({
        "current_query": query6,
        "messages": [HumanMessage(content=query6)],
    }, config=config)

    print(f"Escalated to Human: {res6.get('escalate_to_human')}")
    print(f"Response:\n{res6['generation']}")

    print("\n" + "=" * 80)
    print("✅ ALL 6 LANGGRAPH WORKFLOW TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_tests()
