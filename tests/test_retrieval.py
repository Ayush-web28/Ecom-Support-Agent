"""
Interactive Verification Script for Phase 2.
Tests chunking, vector indexing in ChromaDB, and metadata-filtered similarity search.
"""

import sys
import io
from pathlib import Path

# Force UTF-8 output on Windows consoles
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.rag.vector_store import (
    build_or_load_vector_store,
    search_documents,
    load_documents,
    chunk_documents,
)


def run_tests():
    print("=" * 70)
    print("PHASE 2 VERIFICATION: CHROMA VECTOR STORE & RETRIEVAL")
    print("=" * 70)

    # 1. Test Document Loading & Metadata
    print("\n[Step 1] Loading and tagging documents...")
    raw_docs = load_documents()
    print(f"Loaded {len(raw_docs)} documents.")
    for doc in raw_docs:
        print(f"  - File: {doc.metadata['source']:<32} | Type: {doc.metadata['doc_type']:<7} | Category: {doc.metadata['category']}")

    # 2. Test Chunking
    print("\n[Step 2] Chunking documents...")
    chunks = chunk_documents(raw_docs)
    print(f"Total chunks generated: {len(chunks)}")
    avg_len = sum(len(c.page_content) for c in chunks) / len(chunks)
    print(f"Average chunk length: {avg_len:.1f} characters")

    # 3. Test Vector Store Indexing
    print("\n[Step 3] Initializing ChromaDB with Gemini Embeddings...")
    vector_store = build_or_load_vector_store()

    # 4. Test Semantic Retrieval Queries
    test_queries = [
        {
            "title": "Query A: Policy Question (Open-box items)",
            "query": "What is the return fee or policy for opened box electronics?",
            "doc_type": "policy",
        },
        {
            "title": "Query B: Product Manual Question (Headphones Factory Reset)",
            "query": "How do I factory reset my headphones if buttons are unresponsive?",
            "doc_type": "manual",
        },
        {
            "title": "Query C: Product Manual Question (Lamp Wi-Fi Compatibility)",
            "query": "Can the smart desk lamp connect to a 5GHz Wi-Fi network?",
            "doc_type": "manual",
        },
        {
            "title": "Query D: Policy Question (Battery Warranty Coverage)",
            "query": "What is the warranty coverage on the battery?",
            "doc_type": "policy",
        },
        {            "title": "Query E: Product Manual Question (Headphones Multipoint Dual Pairing)",
            "query": "How do I pair the headphones to two devices?",
            "doc_type": "manual",
        },
    ]

    print("\n[Step 4] Running Retrieval Tests...")
    for item in test_queries:
        print("\n" + "-" * 70)
        print(f"🔍 {item['title']}")
        print(f"Question: \"{item['query']}\"")
        print(f"Filter: doc_type = '{item['doc_type']}'")
        print("-" * 70)

        results = search_documents(
            query=item["query"],
            doc_type=item["doc_type"],
            top_k=2,
            vector_store=vector_store,
        )

        for i, doc in enumerate(results, start=1):
            source = doc.metadata.get("source", "unknown")
            category = doc.metadata.get("category", "unknown")
            snippet = doc.page_content.strip().replace("\n", " ")
            print(f"\n[Result #{i}] Source: {source} (category: {category})")
            print(f"Content: {snippet[:220]}...")

    print("\n" + "=" * 70)
    print("✅ PHASE 2 TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
