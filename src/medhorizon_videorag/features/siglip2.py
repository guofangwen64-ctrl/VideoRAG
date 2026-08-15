from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from medhorizon_videorag.core.schemas import Chunk
from .utils import embed_and_pool_chunks, normalize_rows, open_rgb_images


@dataclass
class SigLIP2VisualEmbedder:
    """SigLIP2 image-text encoder; supports NaFlex checkpoints and native aspect ratios."""
    model_name: str = "google/siglip2-base-patch16-naflex"
    device: str = "cuda"
    batch_size: int = 16

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as error:
            raise RuntimeError("Install model dependencies: pip install -e '.[models]'") from error
        self._torch = torch
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        self.embedding_dim = self.model.config.projection_dim

    def _embed_images(self, paths: list[str]) -> np.ndarray:
        images = open_rgb_images(paths)
        outputs = []
        with self._torch.no_grad():
            for offset in range(0, len(images), self.batch_size):
                inputs = self.processor(images=images[offset:offset + self.batch_size], return_tensors="pt")
                inputs = {name: value.to(self.device) for name, value in inputs.items()}
                outputs.append(self.model.get_image_features(**inputs).float().cpu().numpy())
        return np.vstack(outputs)

    def embed_chunks(self, chunks: Sequence[Chunk]) -> np.ndarray:
        return embed_and_pool_chunks(chunks, self._embed_images, self.embedding_dim, self.batch_size, "Encoding SigLIP2")

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        outputs = []
        with self._torch.no_grad():
            for offset in range(0, len(texts), self.batch_size):
                inputs = self.processor(
                    text=list(texts[offset:offset + self.batch_size]), padding="max_length", max_length=64, return_tensors="pt",
                )
                inputs = {name: value.to(self.device) for name, value in inputs.items()}
                outputs.append(self.model.get_text_features(**inputs).float().cpu().numpy())
        return normalize_rows(np.vstack(outputs)).astype(np.float32) if outputs else np.empty((0, self.embedding_dim), dtype=np.float32)
