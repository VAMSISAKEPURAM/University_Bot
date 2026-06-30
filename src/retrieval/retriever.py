"""
Retriever module — wraps a FAISS vector store to answer user queries.
"""

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


class Retriever:
    """
    Wraps a FAISS vector store and exposes a .retrieve(query) method
    so the API layer can call it without knowing the underlying store type.
    """

    def __init__(self, vector_store: FAISS, k: int = 3):
        """
        Args:
            vector_store: An initialised FAISS vector store object.
            k: Number of most-similar chunks to return per query.
        """
        self._retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    def retrieve(self, query: str) -> str:
        """
        Retrieve the top-k most relevant document chunks for a query
        and return them as a single joined string (ready to inject into a prompt).

        Args:
            query: The user's question.

        Returns:
            A newline-separated string of the retrieved chunk contents.
        """
        docs = self._retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in docs)


def load_retriever(index_path: str = "faiss_index", k: int = 3) -> Retriever:
    """
    Load a persisted FAISS index from disk and return a Retriever instance.

    Args:
        index_path: Folder where index.faiss and index.pkl were saved.
        k: Number of chunks to retrieve.

    Returns:
        A ready-to-use Retriever object.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return Retriever(vector_store, k=k)