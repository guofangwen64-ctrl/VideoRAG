from __future__ import annotations

from typing import Any

from .deterministic import DeterministicVisualEmbedder
from .hf_clip import HFCLIPVisualEmbedder
from .openai_clip import OpenAIClipVisualEmbedder
from .biomed_clip import BiomedCLIPVisualEmbedder
from .siglip2 import SigLIP2VisualEmbedder


def build_visual_embedder(config: dict[str, Any]):
    provider = config.get("provider", "deterministic")
    if provider == "deterministic":
        return DeterministicVisualEmbedder(int(config.get("embedding_dim", 256)))
    if provider == "hf_clip":
        return HFCLIPVisualEmbedder(config["model_name"], config.get("device", "cpu"))
    if provider == "openai_clip":
        return OpenAIClipVisualEmbedder(
            model_name=config.get("model_name", "ViT-B-32"), pretrained=config.get("pretrained", "openai"),
            device=config.get("device", "cuda"), batch_size=int(config.get("batch_size", 32)),
        )
    if provider == "biomed_clip":
        return BiomedCLIPVisualEmbedder(
            model_name=config.get("model_name", BiomedCLIPVisualEmbedder.model_name), device=config.get("device", "cuda"),
            batch_size=int(config.get("batch_size", 32)),
        )
    if provider == "siglip2":
        return SigLIP2VisualEmbedder(
            model_name=config.get("model_name", "google/siglip2-base-patch16-naflex"), device=config.get("device", "cuda"),
            batch_size=int(config.get("batch_size", 16)),
        )
    raise NotImplementedError(f"Vision provider '{provider}' is not registered")
