"""
LLM client using Groq Inference API.
Reads the API key from the GROQ_API_KEY environment variable in .env.

Model: llama-3.1-8b-instant (fast, free-tier friendly on Groq)
Embeddings: sentence-transformers/all-MiniLM-L6-v2 (HuggingFace, unchanged)
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class GroqLLMClient:
    """
    Wraps the Groq Inference API for chat-based text generation.
    Uses the key stored in .env under GROQ_API_KEY.
    """

    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found in environment. "
                "Please add it to your .env file:\n  GROQ_API_KEY=gsk_..."
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
