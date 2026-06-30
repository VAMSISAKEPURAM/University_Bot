"""
RAG prompt templates.
"""

RAG_PROMPT = """You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer cannot be found in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""
