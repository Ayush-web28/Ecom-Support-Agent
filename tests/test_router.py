"""
Interactive Verification Script for Phase 3.
Tests the Pydantic-powered Intent Router across diverse user messages.
"""

import sys
import io
from pathlib import Path

# Force UTF-8 output on Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.agent.router import classify_intent


def run_tests():
    print("=" * 75)
    print("PHASE 3 VERIFICATION: INTENT CLASSIFICATION & ROUTING")
    print("=" * 75)

    test_cases = [
        {
            "id": 1,
            "message": "Hello! Good morning, hope you're doing well.",
            "expected": "greeting",
        },
        {
            "id": 2,
            "message": "Is there a restocking fee if I return an opened box laptop?",
            "expected": "policy",
        },
        {
            "id": 3,
            "message": "How do I factory reset my AeroSound headphones if buttons freeze?",
            "expected": "manual",
        },
        {
            "id": 4,
            "message": "You charged my credit card twice! Let me speak to a real manager right now!",
            "expected": "escalate",
        },
        {
            "id": 5,
            "message": "Does the smart desk lamp support 5GHz Wi-Fi networks?",
            "expected": "manual",
        },
        {
            "id": 6,
            "message": "How long does standard delivery take and is it free?",
            "expected": "policy",
        },
    ]

    passed = 0

    for case in test_cases:
        print("\n" + "-" * 75)
        print(f"Test Case #{case['id']}")
        print(f"Customer Input : \"{case['message']}\"")
        print(f"Expected Route : [{case['expected'].upper()}]")
        print("-" * 75)

        decision = classify_intent(case["message"])

        status = "PASSED" if decision.category == case["expected"] else "FAILED"
        if status == "PASSED":
            passed += 1

        print(f"Predicted Route: [{decision.category.upper()}]  <-- {status}")
        print(f"Confidence     : {decision.confidence:.2f}")
        print(f"Reasoning      : {decision.reasoning}")

    print("\n" + "=" * 75)
    print(f"ROUTER ACCURACY: {passed}/{len(test_cases)} tests passed ({passed/len(test_cases)*100:.0f}%)")
    print("=" * 75)


if __name__ == "__main__":
    run_tests()
