from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from .schemas import Chunk, RetrievalResult


class VisualEmbedder(Protocol):
    def embed_chunks(self, chunks: Sequence[Chunk]) -> np.ndarray: ...
    def embed_text(self, texts: Sequence[str]) -> np.ndarray: ...


class VectorIndex(Protocol):
    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None: ...
    def search(self, query_vector: np.ndarray, top_k: int) -> list[RetrievalResult]: ...
    def save(self, path: Path) -> None: ...


class AnswerGenerator(Protocol):
    def answer(self, question: str, evidence: Sequence[RetrievalResult]) -> str: ...
