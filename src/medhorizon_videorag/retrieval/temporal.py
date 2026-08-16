from __future__ import annotations

from dataclasses import dataclass

from medhorizon_videorag.core.schemas import RetrievalResult
from medhorizon_videorag.datasets import TemporalQuery, parse_temporal_query

from .numpy_index import NumpyVectorIndex


def _iou(first: tuple[float, float], second: tuple[float, float]) -> float:
    overlap = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    union = max(first[1], second[1]) - min(first[0], second[0])
    return overlap / union if union else float(first == second)


@dataclass
class TemporalRetriever:
    """Deterministically retrieve chunks that overlap an explicit time expression."""
    index: NumpyVectorIndex

    def retrieve(self, question: str, video_id: str, top_k: int) -> tuple[TemporalQuery | None, list[RetrievalResult]]:
        temporal = parse_temporal_query(question)
        if temporal is None:
            return None, []
        candidates = [chunk for chunk in self.index.chunks if chunk.video_id == video_id]
        if temporal.kind == "range":
            target = (temporal.start_seconds, temporal.end_seconds)
            results = [
                RetrievalResult(chunk, _iou(target, (chunk.start_seconds, chunk.end_seconds)), source="temporal")
                for chunk in candidates
                if max(target[0], chunk.start_seconds) < min(target[1], chunk.end_seconds)
            ]
        else:
            point = temporal.start_seconds
            results = [
                RetrievalResult(
                    chunk,
                    1.0 if chunk.start_seconds <= point <= chunk.end_seconds else 1.0 / (1.0 + min(abs(point - chunk.start_seconds), abs(point - chunk.end_seconds))),
                    source="temporal",
                )
                for chunk in candidates
            ]
        return temporal, sorted(results, key=lambda item: item.score, reverse=True)[:top_k]
