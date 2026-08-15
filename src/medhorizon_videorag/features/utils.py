from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from medhorizon_videorag.core.schemas import Chunk


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


def embed_and_pool_chunks(
    chunks: Sequence[Chunk], embed_images: Callable[[list[str]], np.ndarray], embedding_dim: int
) -> np.ndarray:
    """Encode frames independently, then normalized mean-pool per temporal chunk."""
    embeddings: list[np.ndarray] = []
    for chunk in chunks:
        if not chunk.frame_paths:
            raise ValueError(f"Chunk {chunk.id} has no decodable frames")
        frame_vectors = normalize_rows(embed_images(chunk.frame_paths))
        pooled = frame_vectors.mean(axis=0, keepdims=True)
        embeddings.append(normalize_rows(pooled)[0])
    return np.vstack(embeddings).astype(np.float32) if embeddings else np.empty((0, embedding_dim), dtype=np.float32)


def open_rgb_images(paths: Sequence[str]):
    from PIL import Image

    images = []
    for path in paths:
        with Image.open(Path(path)) as image:
            images.append(image.convert("RGB"))
    return images
