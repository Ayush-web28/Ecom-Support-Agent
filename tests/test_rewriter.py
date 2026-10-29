"""
Interactive Verification Script for Phase 4.
Tests Conversational Query Rewriting on multi-turn dialogue follow-ups.
"""

import sys
import io
from pathlib import Path

# Force UTF-8 output on Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.agent.rewriter import rewrite_query


def run_tests():
    print("=" * 75)
    print("PHASE 4 VERIFICATION: CONVERSATIONAL QUERY REWRITING")
    print("=" * 75)

    test_scenarios = [
        {
            "id": 1,
            "title": "Policy Follow-up (Electronics -> Open-box)",
            "history": [
                {"role": "user", "content": "What is your return policy for electronics?"},
                {"role": "assistant", "content": "Electronics can be returned within 14 calendar days from delivery in original packaging."},
            ],
            "follow_up": "Does that apply to open-box items too?",
            "expected_keywords": ["open-box", "electronic", "return"],
        },
        {
            "id": 2,
            "title": "Hardware Follow-up (AeroSound Headphones -> Recharge time)",
            "history": [
                {"role": "user", "content": "What is the battery life on the AeroSound Pro headphones?"},
                {"role": "assistant", "content": "The AeroSound Pro AP-900 lasts up to 30 hours with ANC and 40 hours without ANC."},
            ],
            "follow_up": "How long does it take to fully recharge it?",
            "expected_keywords": ["recharge", "aerosound", "headphone"],
        },
        {
            "id": 3,
            "title": "Desk Lamp Follow-up (App setup -> 5GHz Wi-Fi)",
            "history": [
                {"role": "user", "content": "I am trying to connect my LumiSmart Desk Lamp to Wi-Fi."},
                {"role": "assistant", "content": "Download the LumiSmart Home app and tap Add Device when the base blinks blue."},
            ],
            "follow_up": "Can I connect it to my 5GHz router?",
            "expected_keywords": ["lumismart", "lamp", "5ghz", "wi-fi"],
        },
        {
            "id": 4,
            "title": "Standalone Query without Context",
            "history": [],
            "follow_up": "What is the warranty period for all electronics?",
            "expected_keywords": ["warranty", "electronics"],
        },
    ]

    passed = 0

    for sc in test_scenarios:
        print("\n" + "-" * 75)
        print(f"Scenario #{sc['id']}: {sc['title']}")
        if sc["history"]:
            print("Chat History:")
            for turn in sc["history"]:
                print(f"  {turn['role'].capitalize()}: {turn['content']}")
        else:
            print("Chat History: [None]")

        print(f"\nUser Follow-up : \"{sc['follow_up']}\"")

        rewritten = rewrite_query(sc["follow_up"], sc["history"])
        print(f"Rewritten Query: \"{rewritten}\"")

        # Check if expected keywords are present
        rewritten_lower = rewritten.lower()
        matched = sum(1 for kw in sc["expected_keywords"] if kw in rewritten_lower)
        status = "PASSED" if matched >= 2 else "FAILED"

        if status == "PASSED":
            passed += 1

        print(f"Evaluation     : {status} (Found {matched}/{len(sc['expected_keywords'])} key concepts)")

    print("\n" + "=" * 75)
    print(f"REWRITER ACCURACY: {passed}/{len(test_scenarios)} scenarios passed ({passed/len(test_scenarios)*100:.0f}%)")
    print("=" * 75)


if __name__ == "__main__":
    run_tests()
