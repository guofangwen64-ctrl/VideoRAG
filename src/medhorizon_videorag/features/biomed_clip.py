from __future__ import annotations

from dataclasses import dataclass

from .openai_clip import OpenAIClipVisualEmbedder


@dataclass
class BiomedCLIPVisualEmbedder(OpenAIClipVisualEmbedder):
    """PMC-15M pretrained biomedical image-text encoder exposed through OpenCLIP."""
    model_name: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    pretrained: str = ""

    def __post_init__(self) -> None:
        try:
            import open_clip
            import torch
        except ImportError as error:
            raise RuntimeError("Install model dependencies: pip install -e '.[models]'") from error
        self._torch = torch
        self._open_clip = open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(self.model_name)
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.embedding_dim = self.model.visual.output_dim
