"""
Interactive CLI for Intelligent Customer Support Agent (LangGraph + RAG).

Run with:
    python main.py
"""

import sys
import io
import uuid
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Force UTF-8 on Windows consoles
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from src.agent.graph import build_support_graph

load_dotenv()


def print_banner():
    print("\n" + "=" * 78)
    print("      🛒 INTELLIGENT E-COMMERCE CUSTOMER SUPPORT AGENT (LangGraph + RAG)")
    print("=" * 78)
    print("Type your questions below to chat with the assistant.")
    print("Special Commands:")
    print("  /reset  - Clear conversation history and start a fresh session")
    print("  /help   - Show example questions for policies, manuals, and escalations")
    print("  /exit   - Quit the program")
    print("=" * 78 + "\n")


def print_help():
    print("\n💡 EXAMPLE QUESTIONS YOU CAN ASK:")
    print("  Policies:")
    print("    • 'What is your return policy for electronics?'")
    print("    • 'Does that apply to open-box items too?' (Test multi-turn context!)")
    print("    • 'How much does express shipping cost?'")
    print("    • 'Is battery degradation covered under the standard warranty?'")
    print("\n  Product Manuals:")
    print("    • 'How do I factory reset my AeroSound Pro headphones?'")
    print("    • 'Can the LumiSmart desk lamp connect to a 5GHz Wi-Fi network?'")
    print("    • 'How many hours does the headphone battery last with ANC on?'")
    print("\n  Escalation:")
    print("    • 'You double charged my card, I want to speak to a real manager!'")
    print("    • 'What is your refund policy for live pet lizards?' (Test CRAG fallback)\n")


def main():
    print_banner()

    # Compile the LangGraph agent
    try:
        app = build_support_graph()
    except Exception as e:
        print(f"[Error] Failed to initialize agent graph: {e}")
        return

    # Generate a unique thread ID for conversational memory
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        try:
            user_input = input("\nYou > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["/exit", "exit", "quit", ":q"]:
            print("\nThank you for using Customer Support Copilot. Goodbye!")
            break

        if user_input.lower() == "/reset":
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}
            print("\n🔄 Conversation reset. Started a new session!")
            continue

        if user_input.lower() == "/help":
            print_help()
            continue

        print("\nThinking...", end="\r", flush=True)

        try:
            result = app.invoke(
                {
                    "current_query": user_input,
                    "messages": [HumanMessage(content=user_input)],
                },
                config=config,
            )

            # Print debug metadata badge
            category = result.get("category", "unknown").upper()
            standalone = result.get("standalone_query")

            print(" " * 20, end="\r")  # Clear "Thinking..."
            print(f"🤖 [Intent: {category}]", end="")
            if standalone and standalone != user_input:
                print(f" [Rewritten: \"{standalone}\"]")
            else:
                print()

            # Print agent response
            print("-" * 78)
            print(result.get("generation", "No response generated."))
            print("-" * 78)

        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                print("\n⚠️ [Rate Limit Notice]: Gemini Free Tier daily/minute quota reached.")
                print("   To continue, generate a fresh free key at: https://aistudio.google.com/apikey")
                print("   and update GEMINI_API_KEY in your .env file.")
            else:
                print(f"\n[Error executing query]: {e}")


if __name__ == "__main__":
    main()
