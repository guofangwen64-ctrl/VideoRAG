from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from medhorizon_videorag.core.schemas import Chunk


@dataclass
class HFCLIPVisualEmbedder:
    """CLIP adapter: encodes sampled frames then mean-pools them into a video chunk."""
    model_name: str
    device: str = "cpu"

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as error:
            raise RuntimeError("Install model dependencies: pip install -e '.[models]'") from error
        self._torch = torch
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device).eval()
        self.embedding_dim = self.model.config.projection_dim

    def embed_chunks(self, chunks: Sequence[Chunk]) -> np.ndarray:
        from PIL import Image

        embeddings: list[np.ndarray] = []
        for chunk in chunks:
            if not chunk.frame_paths:
                raise ValueError(f"Chunk {chunk.id} has no sampled frames")
            images = [Image.open(frame).convert("RGB") for frame in chunk.frame_paths]
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {name: value.to(self.device) for name, value in inputs.items()}
            with self._torch.no_grad():
                vector = self.model.get_image_features(**inputs).mean(dim=0)
            embeddings.append(self._normalize(vector.cpu().numpy()))
        return np.vstack(embeddings).astype(np.float32)

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        inputs = self.processor(text=list(texts), padding=True, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with self._torch.no_grad():
            vectors = self.model.get_text_features(**inputs).cpu().numpy()
        return np.vstack([self._normalize(vector) for vector in vectors]).astype(np.float32)

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        return vector / max(float(np.linalg.norm(vector)), 1e-12)
