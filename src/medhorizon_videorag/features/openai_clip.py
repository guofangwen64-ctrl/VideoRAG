from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from medhorizon_videorag.core.schemas import Chunk
from .utils import embed_and_pool_chunks, normalize_rows, open_rgb_images


@dataclass
class OpenAIClipVisualEmbedder:
    """OpenAI CLIP ViT-B/32 encoder for the primary MedHorizon baseline."""
    model_name: str = "ViT-B-32"
    pretrained: str = "openai"
    device: str = "cuda"
    batch_size: int = 32

    def __post_init__(self) -> None:
        try:
            import open_clip
            import torch
        except ImportError as error:
            raise RuntimeError("Install model dependencies: pip install -e '.[models]'") from error
        self._torch = torch
        self._open_clip = open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(self.model_name, pretrained=self.pretrained)
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.embedding_dim = self.model.visual.output_dim

    def _embed_images(self, paths: list[str]) -> np.ndarray:
        images = open_rgb_images(paths)
        outputs = []
        with self._torch.no_grad():
            for offset in range(0, len(images), self.batch_size):
                batch = self._torch.stack([self.preprocess(image) for image in images[offset:offset + self.batch_size]]).to(self.device)
                outputs.append(self.model.encode_image(batch).float().cpu().numpy())
        return np.vstack(outputs)

    def embed_chunks(self, chunks: Sequence[Chunk]) -> np.ndarray:
        return embed_and_pool_chunks(chunks, self._embed_images, self.embedding_dim, self.batch_size, "Encoding OpenAI CLIP")

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        outputs = []
        with self._torch.no_grad():
            for offset in range(0, len(texts), self.batch_size):
                tokens = self.tokenizer(list(texts[offset:offset + self.batch_size])).to(self.device)
                outputs.append(self.model.encode_text(tokens).float().cpu().numpy())
        return normalize_rows(np.vstack(outputs)).astype(np.float32) if outputs else np.empty((0, self.embedding_dim), dtype=np.float32)
