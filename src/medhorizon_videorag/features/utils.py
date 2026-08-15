from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from typing import Callable, Sequence

import numpy as np

from medhorizon_videorag.core.schemas import Chunk


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


def chunks_with_existing_frames(chunks: Sequence[Chunk]) -> tuple[list[Chunk], list[dict[str, str]]]:
    """Remove zero-frame chunks before indexing and retain an audit trail."""
    valid: list[Chunk] = []
    skipped: list[dict[str, str]] = []
    for chunk in chunks:
        paths = [path for path in chunk.frame_paths if Path(path).is_file()]
        if not paths:
            skipped.append({"chunk_id": chunk.id, "video_id": chunk.video_id, "reason": "no_existing_decodable_frames"})
        else:
            valid.append(replace(chunk, frame_paths=paths))
    return valid, skipped


def embed_and_pool_chunks(
    chunks: Sequence[Chunk], embed_images: Callable[[list[str]], np.ndarray], embedding_dim: int, batch_size: int, progress_label: str,
) -> np.ndarray:
    """Encode frames in cross-chunk batches, then normalized mean-pool per chunk.

    Cross-chunk batching keeps the GPU full: a 30-second chunk only has eight
    frames, so batching per chunk would otherwise make ``batch_size`` useless.
    """
    if not chunks:
        return np.empty((0, embedding_dim), dtype=np.float32)
    total_frames = sum(len(chunk.frame_paths) for chunk in chunks)
    sums = np.zeros((len(chunks), embedding_dim), dtype=np.float32)
    counts = np.zeros(len(chunks), dtype=np.int32)
    try:
        from tqdm.auto import tqdm
        progress = tqdm(total=total_frames, desc=progress_label, unit="frame", dynamic_ncols=True)
    except ImportError:
        progress = None

    paths: list[str] = []
    owners: list[int] = []

    def flush() -> None:
        if not paths:
            return
        vectors = normalize_rows(embed_images(paths))
        for owner, vector in zip(owners, vectors, strict=True):
            sums[owner] += vector
            counts[owner] += 1
        if progress:
            progress.update(len(paths))
        paths.clear()
        owners.clear()

    for chunk_index, chunk in enumerate(chunks):
        for path in chunk.frame_paths:
            paths.append(path)
            owners.append(chunk_index)
            if len(paths) >= batch_size:
                flush()
    flush()
    if progress:
        progress.close()
    if not np.all(counts):
        raise RuntimeError("Frame filtering yielded an empty chunk during embedding")
    return normalize_rows(sums / counts[:, None]).astype(np.float32)


def open_rgb_images(paths: Sequence[str]):
    from PIL import Image

    images = []
    for path in paths:
        with Image.open(Path(path)) as image:
            images.append(image.convert("RGB"))
    return images
