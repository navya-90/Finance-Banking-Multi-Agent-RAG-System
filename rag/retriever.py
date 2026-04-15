import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import CHROMA_PERSIST_DIR, HF_EMBEDDING_MODEL, HF_DEVICE, RAG_TOP_K

# Single shared embedding instance — same model used in ingest.py
embeddings = HuggingFaceEmbeddings(
    model_name=HF_EMBEDDING_MODEL,
    model_kwargs={"device": HF_DEVICE},
    encode_kwargs={"normalize_embeddings": True},
)

COLLECTIONS = {
    "fraud":      "fraud_policies",
    "loans":      "loan_products",
    "accounts":   "account_faqs",
    "markets":    "market_reports",
    "compliance": "regulatory_docs",
    "general":    "general_faqs",
}


def _get_vectorstore(domain: str) -> Chroma:
    return Chroma(
        collection_name=COLLECTIONS[domain],
        embedding_function=embeddings,
        persist_directory=os.path.join(CHROMA_PERSIST_DIR, domain),
    )


def retrieve(domain: str, query: str, n_results: int = RAG_TOP_K) -> list[str]:
    """Return top-k relevant chunks for a query. Called by @tool functions."""
    vs = _get_vectorstore(domain)
    docs = vs.similarity_search(query, k=n_results)
    return [doc.page_content for doc in docs]
