"""
LLM client using Groq Inference API.
Reads the API key from Streamlit secrets (preferred) or GROQ_API_KEY env variable.

Model: llama-3.1-8b-instant (fast, free-tier friendly on Groq)
Embeddings: sentence-transformers/all-MiniLM-L6-v2 (HuggingFace, unchanged)
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


def _get_groq_api_key() -> str:
    """
    Resolve the Groq API key with the following priority:
    1. Streamlit secrets  (st.secrets["GROQ_API_KEY"])  — used on Streamlit Cloud
    2. Environment variable / .env file                  — used in local development
    """
    # Try Streamlit secrets first (available when running via `streamlit run`)
    try:
        import streamlit as st
        key = st.secrets.get("GROQ_API_KEY", "")
        if key:
            return key
    except Exception:
        pass  # Not running inside Streamlit, fall through

    # Fall back to environment variable / .env
    key = os.getenv("GROQ_API_KEY", "")
    return key


class GroqLLMClient:
    """
    Wraps the Groq Inference API for chat-based text generation.
    Reads GROQ_API_KEY from Streamlit secrets first, then from .env / environment.
    """

    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        api_key = _get_groq_api_key()
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. "
                "Add it to Streamlit secrets (GROQ_API_KEY = '...') "
                "or to your .env file (GROQ_API_KEY=gsk_...)."
            )
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        """
        Send a prompt to the Groq-hosted LLaMA model and return the response.

        Uses Hugging Face embeddings (sentence-transformers/all-MiniLM-L6-v2)
        to find the most relevant context, and Groq LLM (llama-3.1-8b-instant)
        to generate the response.

        Args:
            prompt: The full RAG prompt (context + question already formatted).
            max_new_tokens: Maximum tokens to generate in the response.

        Returns:
            The model's text response as a string.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()


# ── Backward-compatible alias so any existing import still works ───────────────
HuggingFaceLLMClient = GroqLLMClient
