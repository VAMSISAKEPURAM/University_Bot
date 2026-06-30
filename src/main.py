"""
Main entry point for the Local RAG Chatbot.

Usage:
  Ingest a PDF:      python src/main.py --ingest path/to/file.pdf
  Start the server:  python src/main.py --serve
  Ingest + serve:    python src/main.py --ingest path/to/file.pdf --serve
"""

import argparse
import logging
import sys
import os

# ── Make sure all src sub-packages are importable ──────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

import uvicorn

from utils.helpers import load_config, setup_logging
from ingestion.loader import load_pdf
from chunking.chunkers import make_chunks
from embeddings.embedder import make_embeddings
from vector_db.vectorstores import vector_stores
from retrieval.retriever import Retriever, load_retriever
from llm.llm_clients import GroqLLMClient
from api.routes import app, initialize_rag


def ingest(pdf_path: str, config: dict) -> Retriever:
    """
    Full ingestion pipeline:
      PDF → load → chunk → embed → FAISS → Retriever

    Args:
        pdf_path: Path to the PDF file to ingest.
        config:   Parsed config.yaml as a dict.

    Returns:
        A Retriever object ready to answer queries.
    """
    logger = logging.getLogger(__name__)

    logger.info(f"Loading PDF: {pdf_path}")
    documents = load_pdf(pdf_path)  # loading pdf files

    chunk_cfg = config.get("chunking", {})
    logger.info("Chunking documents...")
    chunks = make_chunks(
        documents,
        chunk_size=chunk_cfg.get("chunk_size", 500),
        chunk_overlap=chunk_cfg.get("chunk_overlap", 50),
    )                                                               # chunks are stored in the faiss_index directory
    logger.info(f"Created {len(chunks)} chunks.")

    embed_cfg = config.get("embeddings", {})
    logger.info("Loading embedding model...")
    embedding_model = make_embeddings(
        model_name=embed_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    )                                       # Loading Embedding Model

    vectordb_cfg = config.get("vectordb", {})
    index_path = vectordb_cfg.get("persist_directory", "faiss_index") # Path to save FAISS index

    logger.info("Building FAISS index...")
    vector_store = vector_stores(chunks, embedding_model, save_path=index_path)    # Building FAISS index
    logger.info(f"FAISS index saved to '{index_path}'.")

    return Retriever(vector_store)


def main():
    setup_logging()
    config = load_config()

    parser = argparse.ArgumentParser(description="Local RAG Chatbot")
    parser.add_argument("--ingest", metavar="PDF_PATH", help="Path to a PDF to ingest")
    parser.add_argument("--serve", action="store_true", help="Start the FastAPI server")
    args = parser.parse_args()

    retriever = None

    if args.ingest:
        retriever = ingest(args.ingest, config)

    if args.serve:
        # If not freshly ingested, try loading an existing index from disk
        if retriever is None:
            vectordb_cfg = config.get("vectordb", {})
            index_path = vectordb_cfg.get("persist_directory", "faiss_index")
            logging.getLogger(__name__).info(
                f"Loading existing FAISS index from '{index_path}'..."
            )
            retriever = load_retriever(index_path)

        llm_cfg = config.get("llm", {})
        llm_client = GroqLLMClient(
            model_name=llm_cfg.get("model_name", "llama-3.1-8b-instant")
        )

        initialize_rag(retriever, llm_client)

        logging.getLogger(__name__).info("Starting FastAPI server on http://localhost:8000")
        uvicorn.run(app, host="0.0.0.0", port=8000)

    if not args.ingest and not args.serve:
        parser.print_help()


if __name__ == "__main__":
    main()
