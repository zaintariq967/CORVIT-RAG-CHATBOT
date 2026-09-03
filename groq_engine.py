"""
groq_engine.py
------------------------------------------------------------------
Every call this project makes to Groq goes through this one class.

Groq is used for BOTH stages of the RAG pipeline:
  - Embeddings  (model: nomic-embed-text-v1_5)  -> turns text into
    768-dim vectors, for both indexing the knowledge base and
    embedding each user question.
  - Chat generation (model: openai/gpt-oss-120b by default) -> writes
    the final answer, grounded in the chunks retrieved from FAISS.

Using Groq for both means the whole app needs exactly one API key,
entered at runtime -- nothing is hardcoded, and no separate embedding
provider or heavyweight local model is required.

Note on model choice: Groq deprecated free/standard access to its
older Llama chat models (llama-3.3-70b-versatile, llama-3.1-8b-instant)
in favor of the open-weight GPT-OSS family, so those are what this
project targets by default. See AVAILABLE_MODELS below.
"""

from typing import Dict, List

import numpy as np
from groq import Groq

# Prefixes recommended by Nomic for retrieval-quality embeddings: the
# model was contrastively trained to distinguish "this is a document"
# from "this is a search query", which measurably improves retrieval.
_DOC_PREFIX = "search_document: "
_QUERY_PREFIX = "search_query: "

EMBEDDING_MODEL = "nomic-embed-text-v1_5"

# Shown in the Streamlit model picker. gpt-oss-120b is the stronger
# default; gpt-oss-20b trades a little quality for noticeably lower
# latency and cost.
AVAILABLE_MODELS = {
    "openai/gpt-oss-120b": "GPT-OSS 120B — best quality (default)",
    "openai/gpt-oss-20b": "GPT-OSS 20B — fastest, cheapest",
}
DEFAULT_MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You are CorvitSage, the official AI knowledge assistant for Corvit Systems, \
an IT and professional-training organization in Pakistan with campuses in Lahore, Islamabad, \
Rawalpindi, Peshawar and Muzaffarabad.

You answer questions using ONLY the CONTEXT passages supplied with each question. The passages \
are excerpts from Corvit's own knowledge-base document.

Follow these rules strictly:
1. Ground every statement in the supplied context. Never invent course eligibility rules, batch \
dates, instructor names, discounts or guarantees that are not explicitly present in the context.
2. If the context does not contain the answer, say so plainly and suggest the person contact \
Corvit directly or check the official website -- do not guess.
3. Prices, phone numbers, and timings can change. When you state one, briefly note it should be \
verified with Corvit directly, since the source material itself flags these as subject to change.
4. Be concise, warm and professional -- like a knowledgeable admissions counselor, not a search engine.
5. Where it helps, mention which section of the knowledge base an answer is drawn from.
"""


class GroqEngine:
    """Thin, purpose-built wrapper around the Groq Python SDK."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        if not api_key:
            raise ValueError("A Groq API key is required.")
        self.client = Groq(api_key=api_key)
        self.model = model

    # ---------------------------------------------------------- embeddings

    def _embed(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Embed a list of strings, batching to stay well under API limits."""
        all_vectors: List[List[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=EMBEDDING_MODEL,
            )
            # Groq returns results in the same order as the input batch,
            # each tagged with its index -- sort defensively just in case.
            ordered = sorted(response.data, key=lambda e: e.index)
            all_vectors.extend(item.embedding for item in ordered)

        vectors = np.array(all_vectors, dtype="float32")
        # L2-normalize so a FAISS inner-product index behaves as cosine similarity.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Embed knowledge-base chunks for indexing."""
        return self._embed([_DOC_PREFIX + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single user question for retrieval."""
        return self._embed([_QUERY_PREFIX + text])

    # ----------------------------------------------------------- generation

    def generate_answer(
        self,
        question: str,
        context: str,
        history: List[Dict[str, str]],
        model: str = None,
        temperature: float = 0.3,
        max_tokens: int = 800,
    ) -> str:
        """Generate a grounded answer from retrieved context + chat history.

        `model` can override the engine's default per call -- this keeps a
        single cached GroqEngine/RAGPipeline safe to reuse across a
        Streamlit session even as the user switches models in the sidebar,
        with no shared mutable state to race on.
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append(
            {
                "role": "user",
                "content": (
                    f"CONTEXT from Corvit's knowledge base:\n{context}\n\n"
                    f"QUESTION: {question}"
                ),
            }
        )

        response = self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content