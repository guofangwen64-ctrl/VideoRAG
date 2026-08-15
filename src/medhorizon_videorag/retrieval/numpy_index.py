from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from medhorizon_videorag.core.schemas import Chunk, RetrievalResult


class NumpyVectorIndex:
    def __init__(self, chunks: Sequence[Chunk] | None = None, vectors: np.ndarray | None = None) -> None:
        self.chunks = list(chunks or [])
        self.vectors = vectors if vectors is not None else np.empty((0, 0), dtype=np.float32)

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Each chunk must have exactly one vector")
        normalized = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        self.chunks.extend(chunks)
        self.vectors = normalized.astype(np.float32) if self.vectors.size == 0 else np.vstack((self.vectors, normalized))

    def search(self, query_vector: np.ndarray, top_k: int, video_id: str | None = None) -> list[RetrievalResult]:
        """Return nearest chunks, optionally restricted to one source video."""
        if not self.chunks:
            return []
        query = query_vector / max(float(np.linalg.norm(query_vector)), 1e-12)
        scores = self.vectors @ query
        if video_id is None:
            ids = np.argsort(-scores)[:top_k]
        else:
            candidate_ids = np.fromiter((index for index, chunk in enumerate(self.chunks) if chunk.video_id == video_id), dtype=np.int64)
            if not len(candidate_ids):
                return []
            ranked = candidate_ids[np.argsort(-scores[candidate_ids])[:top_k]]
            ids = ranked
        return [RetrievalResult(self.chunks[index], float(scores[index])) for index in ids]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.vectors)
        (path / "chunks.json").write_text(json.dumps([c.to_dict() for c in self.chunks], ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "NumpyVectorIndex":
        vectors = np.load(path / "vectors.npy")
        chunks = [Chunk(**row) for row in json.loads((path / "chunks.json").read_text(encoding="utf-8"))]
        return cls(chunks, vectors)
