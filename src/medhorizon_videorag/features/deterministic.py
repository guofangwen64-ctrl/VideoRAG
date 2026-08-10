from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from medhorizon_videorag.core.schemas import Chunk


@dataclass
class DeterministicVisualEmbedder:
    """Dependency-free baseline encoder for testing the complete pipeline.

    Replace this with a VideoCLIP/medical VLM adapter for real experiments.
    """
    embedding_dim: int = 256

    def embed_chunks(self, chunks: Sequence[Chunk]) -> np.ndarray:
        vectors = []
        for chunk in chunks:
            payload = f"{chunk.video_id}|{chunk.start_seconds}|{chunk.end_seconds}".encode()
            seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
            vector = np.random.default_rng(seed).normal(size=self.embedding_dim).astype(np.float32)
            vectors.append(vector / np.linalg.norm(vector))
        return np.vstack(vectors) if vectors else np.empty((0, self.embedding_dim), dtype=np.float32)

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
            vector = np.random.default_rng(seed).normal(size=self.embedding_dim).astype(np.float32)
            vectors.append(vector / np.linalg.norm(vector))
        return np.vstack(vectors) if vectors else np.empty((0, self.embedding_dim), dtype=np.float32)
