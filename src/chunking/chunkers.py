"""
Chunking module — splits documents into overlapping text chunks.
Reads chunk_size and chunk_overlap from config.yaml when available.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def make_chunks(documents, chunk_size: int = 500, chunk_overlap: int = 50):
    """
    Split a list of LangChain Documents into smaller overlapping chunks.

    Args:
        documents: List of Document objects from the loader.
        chunk_size: Maximum number of characters per chunk (default matches config.yaml).
        chunk_overlap: Number of characters shared between consecutive chunks.

    Returns:
        List of chunked Document objects.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_documents(documents)