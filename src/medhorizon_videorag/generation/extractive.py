from __future__ import annotations

from medhorizon_videorag.core.schemas import RetrievalResult


class ExtractiveGenerator:
    """Transparent offline generator; replace with an LLM adapter for experiments."""

    def answer(self, question: str, evidence: list[RetrievalResult]) -> str:
        if not evidence:
            return "未检索到与问题相关的视频证据。"
        best = evidence[0].chunk
        return (
            f"根据最相关视频片段（{best.start_seconds:.1f}s–{best.end_seconds:.1f}s），"
            "需要由医学语言模型结合该片段内容给出最终回答。"
        )
