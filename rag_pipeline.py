"""
rag_pipeline.py
------------------------------------------------------------------
Ties the two RAG phases together behind one small interface that
app.py calls:

  Offline / indexing phase (runs once, cached to disk after that):
    PDF  --extract-->  pages  --chunk-->  Chunks
         --embed (Groq)-->  vectors  --build-->  FAISS index

  Online / query phase (runs on every user question):
    question  --embed (Groq)-->  query vector
              --search (FAISS)-->  top-k Chunks
              --build prompt-->  context + question + history
              --generate (Groq)-->  answer
"""

from pathlib import Path
from typing import Dict, List, Tuple

from .groq_engine import GroqEngine
from .pdf_processor import Chunk, chunk_document
from .vector_store import VectorStore, file_hash

PDF_PATH = "assets/data/corvit.pdf"
INDEX_DIR = "assets/index"


class RAGPipeline:
    def __init__(self, engine: GroqEngine, pdf_path: str = PDF_PATH, index_dir: str = INDEX_DIR):
        self.engine = engine
        self.pdf_path = pdf_path
        self.index_dir = index_dir
        self.store = self._load_or_build_index()

    # ------------------------------------------------------------ indexing

    def _load_or_build_index(self) -> VectorStore:
        current_hash = file_hash(self.pdf_path)

        if VectorStore.exists(self.index_dir):
            cached = VectorStore.load(self.index_dir)
            if cached.source_hash == current_hash:
                return cached
            # The PDF in assets/data has changed since the index was built --
            # fall through and rebuild rather than silently serving stale data.

        return self._build_index(current_hash)

    def _build_index(self, source_hash: str) -> VectorStore:
        chunks: List[Chunk] = chunk_document(self.pdf_path)
        if not chunks:
            raise RuntimeError(
                f"No text could be extracted from {self.pdf_path}. "
                "Confirm the PDF exists and is not a scanned image."
            )
        texts = [c.text for c in chunks]
        embeddings = self.engine.embed_documents(texts)

        store = VectorStore(dim=embeddings.shape[1])
        store.build(chunks, embeddings, source_hash)
        store.save(self.index_dir)
        return store

    def rebuild(self) -> None:
        """Force a fresh index, e.g. after the source PDF is replaced."""
        self.store = self._build_index(file_hash(self.pdf_path))

    @property
    def chunk_count(self) -> int:
        return len(self.store.chunks)

    # -------------------------------------------------------------- query

    def retrieve(self, question: str, top_k: int = 4) -> List[Tuple[Chunk, float]]:
        query_vector = self.engine.embed_query(question)
        return self.store.search(query_vector, top_k=top_k)

    def answer(
        self,
        question: str,
        history: List[Dict[str, str]],
        top_k: int = 4,
        model: str = None,
    ) -> Tuple[str, List[Tuple[Chunk, float]]]:
        results = self.retrieve(question, top_k=top_k)

        if not results:
            context = "No relevant passages were found in the knowledge base."
        else:
            context = "\n\n".join(f"[Section {c.section_title}]\n{c.text}" for c, _ in results)

        answer_text = self.engine.generate_answer(question, context, history, model=model)
        return answer_text, results