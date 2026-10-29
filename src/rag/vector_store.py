"""
Vector Store & RAG Retrieval Module.
Handles document loading, metadata tagging, markdown chunking,
ChromaDB indexing, and filtered similarity search.
"""

import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
CHROMA_DIR = Path(__file__).resolve().parent.parent.parent / "chroma_db"


# =====================================================================
# TASK 1: METADATA ENRICHMENT
# =====================================================================
def extract_metadata(file_path: Path) -> dict:
    """
    Extracts rich metadata from the file path.

    Why it matters:
    Without metadata, a customer asking "How do I return my lamp?" might retrieve
    lamp specs instead of return policies. Tagging every chunk with 'doc_type'
    and 'category' enables targeted, high-precision retrieval.

    TODO 1 [YOUR TURN]:
    Inspect file_path.parent.name and file_path.stem to return a dictionary with:
      - "source": file_path.name (e.g. "return_and_refund.md")
      - "doc_type": "policy" if the file is in 'policies' folder, else "manual"
      - "category": a clean string derived from file_path.stem (e.g. "return_and_refund" or "aero_sound_pro_headphones")
    """
    # --- YOUR CODE STARTS HERE ---
    source = file_path.name
    doc_type = "policy" if "policies" in str(file_path) else "manual"
    category = file_path.stem
    # --- YOUR CODE ENDS HERE ---

    return {
        "source": source,
        "doc_type": doc_type,
        "category": category,
    }


def load_documents(data_dir: Path = DATA_DIR) -> List[Document]:
    """Loads all Markdown files from data/policies and data/manuals."""
    documents = []
    for md_file in data_dir.glob("**/*.md"):
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()

        meta = extract_metadata(md_file)
        documents.append(Document(page_content=content, metadata=meta))

    return documents


# =====================================================================
# TASK 2: CHUNKING CONFIGURATION
# =====================================================================
def create_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Configures the text splitter for Markdown documents.

    Why it matters:
    - Chunk size: If too big (e.g. 2000 chars), specific rules get diluted.
                  If too small (e.g. 100 chars), sentences get sliced.
    - Chunk overlap: Ensures context is not lost across split boundaries.
    - Separators: Splitting at markdown headers ('\\n## ', '\\n### ') keeps
                  sections semantically coherent.

    TODO 2 [YOUR TURN]:
    Choose suitable parameters:
      - chunk_size: recommended between 400 and 600 characters for concise policy rules.
      - chunk_overlap: recommended between 50 and 100 characters.
    """
    # --- YOUR CODE STARTS HERE ---
    chunk_size = 500
    chunk_overlap = 60
    separators = ["\n## ", "\n### ", "\n\n", "\n", " "]
    # --- YOUR CODE ENDS HERE ---

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )


def chunk_documents(docs: List[Document]) -> List[Document]:
    """Splits raw documents into smaller semantic chunks."""
    splitter = create_text_splitter()
    return splitter.split_documents(docs)


# =====================================================================
# EMBEDDINGS & VECTOR DATABASE
# =====================================================================
def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """Returns Gemini's embedding model."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment or .env file.")
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )


def build_or_load_vector_store(persist_dir: Path = CHROMA_DIR) -> Chroma:
    """
    Builds the vector store if it does not exist, or loads the existing one.
    """
    embeddings = get_embedding_model()

    if persist_dir.exists() and any(persist_dir.iterdir()):
        print(f"[VectorStore] Loading existing ChromaDB from: {persist_dir}")
        return Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
        )

    print(f"[VectorStore] Building new ChromaDB vector store...")
    raw_docs = load_documents()
    chunked_docs = chunk_documents(raw_docs)
    print(f"[VectorStore] Loaded {len(raw_docs)} files -> Split into {len(chunked_docs)} chunks.")

    vector_store = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )
    print("[VectorStore] Successfully indexed all chunks into ChromaDB!")
    return vector_store


# =====================================================================
# TASK 3: FILTERED VECTOR SEARCH
# =====================================================================
def search_documents(
    query: str,
    doc_type: Optional[str] = None,
    top_k: int = 3,
    vector_store: Optional[Chroma] = None,
) -> List[Document]:
    """
    Searches the vector store for documents relevant to `query`.

    Why it matters:
    Our LangGraph router classifies queries as 'policy' or 'manual'.
    Passing `doc_type` allows filtering out irrelevant collections.

    TODO 3 [YOUR TURN]:
    Construct the Chroma search call:
    - If doc_type is provided (e.g. 'policy' or 'manual'), use:
        filter_dict = {"doc_type": doc_type}
      Otherwise, filter_dict = None.
    - Call vector_store.similarity_search(query=query, k=top_k, filter=filter_dict)
    """
    if vector_store is None:
        vector_store = build_or_load_vector_store()

    # --- YOUR CODE STARTS HERE ---
    filter_dict = {"doc_type": doc_type} if doc_type else None
    results = vector_store.similarity_search(query=query, k=top_k, filter=filter_dict)
    # --- YOUR CODE ENDS HERE ---

    return results
