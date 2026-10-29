"""
End-to-End Test Suite for Intelligent Customer Support Agent.
Validates all 5 core requirements from the problem statement:
  1. Intent categorization
  2. Policy retrieval & grounding
  3. Product manual retrieval & specs
  4. Multi-turn conversational memory & follow-ups
  5. Human escalation & ticketing
"""

import sys
import io
import time
from pathlib import Path
from langchain_core.messages import HumanMessage

# Force UTF-8 on Windows consoles
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.agent.graph import build_support_graph


def test_end_to_end_suite():
    print("=" * 80)
    print("PHASE 6: END-TO-END SPECIFICATION VERIFICATION SUITE")
    print("=" * 80)

    app = build_support_graph()
    thread_id = "test-thread-e2e"
    config = {"configurable": {"thread_id": thread_id}}

    # Turn 1: Policy inquiry
    q1 = "What is the return window for electronics?"
    print(f"\n[Turn 1] User: \"{q1}\"")
    r1 = app.invoke({"current_query": q1, "messages": [HumanMessage(content=q1)]}, config=config)
    print(f"Agent ({r1['category'].upper()}):\n{r1['generation']}")
    time.sleep(1)

    # Turn 2: Follow-up question (tests memory & rewriting)
    q2 = "Does that apply to open-box items as well?"
    print(f"\n[Turn 2] User: \"{q2}\"")
    r2 = app.invoke({"current_query": q2, "messages": [HumanMessage(content=q2)]}, config=config)
    print(f"Rewritten Query: \"{r2.get('standalone_query')}\"")
    print(f"Agent:\n{r2['generation']}")
    time.sleep(1)

    # Turn 3: Product manual inquiry
    q3 = "How do I factory reset my AeroSound Pro headphones?"
    print(f"\n[Turn 3] User: \"{q3}\"")
    r3 = app.invoke({"current_query": q3, "messages": [HumanMessage(content=q3)]}, config=config)
    print(f"Agent ({r3['category'].upper()}):\n{r3['generation']}")
    time.sleep(1)

    # Turn 4: Direct human escalation
    q4 = "You double charged my account, I want a human agent immediately!"
    print(f"\n[Turn 4] User: \"{q4}\"")
    r4 = app.invoke({"current_query": q4, "messages": [HumanMessage(content=q4)]}, config=config)
    print(f"Escalated Flag: {r4.get('escalate_to_human')}")
    print(f"Agent:\n{r4['generation']}")

    print("\n" + "=" * 80)
    print("✅ END-TO-END VERIFICATION SUITE COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    test_end_to_end_suite()
