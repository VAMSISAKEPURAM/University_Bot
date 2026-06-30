"""
API routes — FastAPI endpoints for the RAG chatbot.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="RAG Chatbot API")

# Global variables to hold our core components
_retriever = None
_llm_client = None


def initialize_rag(retriever, llm_client):
    """Called from main.py after ingestion to wire up the retriever and LLM."""
    global _retriever, _llm_client
    _retriever = retriever
    _llm_client = llm_client


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask(request: AskRequest):
    if _retriever is None or _llm_client is None:
        raise HTTPException(
            status_code=503,
            detail="RAG not initialized. Please ingest a PDF first.",
        )

    # 1. Retrieve context
    context = _retriever.retrieve(request.question)
    if not context:
        return {
            "question": request.question,
            "answer": "No relevant information found in the ingested document.",
            "context": "",
        }

    # 2. Build prompt — import from the actual filename (prompt_templets, not prompt_templates)
    from prompts.prompt_templets import RAG_PROMPT          # ← fixed import path
    prompt = RAG_PROMPT.format(context=context, question=request.question)

    # 3. Generate answer
    answer = _llm_client.generate(prompt)

    return {
        "question": request.question,
        "answer": answer,
        "context": context,
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "rag_initialized": _retriever is not None}