"""
vector_store.py
------------------------------------------------------------------
A small FAISS wrapper: build an index from chunk embeddings, persist
it to disk so it survives restarts, and search it at query time.

Embeddings coming out of GroqEngine are already L2-normalized, so a
plain inner-product index (IndexFlatIP) gives exact cosine-similarity
search with no extra math -- and for a knowledge base this size (a
few dozen chunks), an exhaustive flat index is both simplest and fast
enough that no approximate-search structure is needed.
"""

import hashlib
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from .pdf_processor import Chunk

INDEX_FILE = "index.faiss"
META_FILE = "meta.pkl"


def file_hash(path: str) -> str:
    """Content hash of the source PDF, used to detect a swapped-in file
    so a stale cached index doesn't silently keep being served."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: List[Chunk] = []
        self.source_hash: str = ""

    def build(self, chunks: List[Chunk], embeddings: np.ndarray, source_hash: str) -> None:
        self.chunks = chunks
        self.source_hash = source_hash
        self.index.add(embeddings.astype("float32"))

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> List[Tuple[Chunk, float]]:
        if self.index.ntotal == 0:
            return []
        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vector.astype("float32"), top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def save(self, folder: str) -> None:
        folder_path = Path(folder)
        folder_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(folder_path / INDEX_FILE))
        with open(folder_path / META_FILE, "wb") as f:
            pickle.dump(
                {"chunks": self.chunks, "source_hash": self.source_hash, "dim": self.dim}, f
            )

    @classmethod
    def load(cls, folder: str) -> "VectorStore":
        folder_path = Path(folder)
        with open(folder_path / META_FILE, "rb") as f:
            meta = pickle.load(f)
        store = cls(meta["dim"])
        store.index = faiss.read_index(str(folder_path / INDEX_FILE))
        store.chunks = meta["chunks"]
        store.source_hash = meta["source_hash"]
        return store

    @staticmethod
    def exists(folder: str) -> bool:
        folder_path = Path(folder)
        return (folder_path / INDEX_FILE).exists() and (folder_path / META_FILE).exists()