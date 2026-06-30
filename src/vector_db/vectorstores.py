"""
Vector store creation and persistence using FAISS.
"""

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def vector_stores(splits, embeddings: HuggingFaceEmbeddings, save_path: str = "faiss_index") -> FAISS:
    """
    Build a FAISS vector store from document chunks, save it to disk, and return it.

    Args:
        splits: List of LangChain Document objects (the chunked text).
        embeddings: An initialised HuggingFaceEmbeddings model object.
        save_path: Directory path where the FAISS index will be saved.

    Returns:
        The FAISS vector store object (needed to create a retriever).
    """
    vector_store = FAISS.from_documents(
        documents=splits,
        embedding=embeddings,
    )
    vector_store.save_local(save_path)
    return vector_store  # ← was missing; caller received None without this