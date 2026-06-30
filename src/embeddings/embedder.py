"""
Embedding model factory.

IMPORTANT: This function returns the embedding MODEL OBJECT, not pre-computed vectors.
FAISS.from_documents() needs the model itself so it can embed both documents and queries.
"""

from langchain_huggingface import HuggingFaceEmbeddings


def make_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """
    Create and return a HuggingFaceEmbeddings model object.

    Args:
        model_name: HuggingFace model identifier for sentence embeddings.

    Returns:
        HuggingFaceEmbeddings instance — pass this directly to FAISS.from_documents().
    """
    return HuggingFaceEmbeddings(model_name=model_name)