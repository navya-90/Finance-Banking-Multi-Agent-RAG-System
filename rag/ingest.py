import os
import sys
import argparse
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Import shared embedding instance and settings from config — not redefined here
from rag.retriever import embeddings, COLLECTIONS
from config import CHROMA_PERSIST_DIR, CHUNK_SIZE, CHUNK_OVERLAP

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def load_pdf(pdf_path: str) -> list:
    """Load PDF using PyPDFLoader. Returns one Document per page."""
    print(f"  Loading: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"  - {len(pages)} pages loaded")
    return pages


def split_documents(pages: list) -> list:
    """Split pages into overlapping chunks and tag each with a unique chunk_id."""
    chunks = _splitter.split_documents(pages)
    for chunk in chunks:
        chunk.metadata["chunk_id"] = str(uuid.uuid4())
    print(f"  - {len(chunks)} chunks after splitting")
    return chunks


def ingest_pdf(domain: str, pdf_path: str):
    """Full pipeline: load PDF → split → embed (shared model) → store in ChromaDB."""
    if domain not in COLLECTIONS:
        raise ValueError(f"Unknown domain '{domain}'. Choose from: {list(COLLECTIONS)}")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"\n[{domain.upper()}] {os.path.basename(pdf_path)}")
    pages  = load_pdf(pdf_path)
    chunks = split_documents(pages)

    persist_dir = os.path.join(CHROMA_PERSIST_DIR, domain)
    vectorstore = Chroma(
        collection_name=COLLECTIONS[domain],
        embedding_function=embeddings,       # same instance as retriever.py
        persist_directory=persist_dir,
    )

    ids = [c.metadata["chunk_id"] for c in chunks]
    vectorstore.add_documents(documents=chunks, ids=ids)
    print(f"  - {len(chunks)} chunks stored in '{COLLECTIONS[domain]}'")


def ingest_folder(domain: str, folder_path: str):
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Folder not found: {folder_path}")
    pdfs = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    if not pdfs:
        print(f"  No PDFs found in {folder_path}")
        return
    print(f"\n[{domain.upper()}] {len(pdfs)} PDFs in {folder_path}")
    for pdf_file in pdfs:
        ingest_pdf(domain, os.path.join(folder_path, pdf_file))


def ingest_all(base_folder: str = "data/docs"):
    for domain in COLLECTIONS:
        folder = os.path.join(base_folder, domain)
        if os.path.isdir(folder):
            ingest_folder(domain, folder)
        else:
            print(f"[{domain.upper()}] Skipping — folder not found: {folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into ChromaDB")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdf",    help="Path to a single PDF")
    group.add_argument("--folder", help="Path to a folder of PDFs")
    group.add_argument("--all",    action="store_true", help="Ingest all domains from data/docs/")
    parser.add_argument("--domain", help="Target domain (required with --pdf / --folder)")
    args = parser.parse_args()

    if args.all:
        ingest_all()
    elif args.pdf:
        if not args.domain:
            parser.error("--domain is required with --pdf")
        ingest_pdf(args.domain, args.pdf)
    elif args.folder:
        if not args.domain:
            parser.error("--domain is required with --folder")
        ingest_folder(args.domain, args.folder)

    print("\nIngestion complete.")
