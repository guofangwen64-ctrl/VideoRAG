from __future__ import annotations

from typing import Any

from .deterministic import DeterministicVisualEmbedder
from .hf_clip import HFCLIPVisualEmbedder


def build_visual_embedder(config: dict[str, Any]):
    provider = config.get("provider", "deterministic")
    if provider == "deterministic":
        return DeterministicVisualEmbedder(int(config.get("embedding_dim", 256)))
    if provider == "hf_clip":
        return HFCLIPVisualEmbedder(config["model_name"], config.get("device", "cpu"))
    raise NotImplementedError(f"Vision provider '{provider}' is not registered")
