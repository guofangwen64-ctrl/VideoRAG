import numpy as np

from medhorizon_videorag.core.schemas import Chunk, RetrievalResult
from medhorizon_videorag.retrieval import HybridRetriever, NumpyVectorIndex, TemporalRetriever


class _StubVisualRetriever:
    def retrieve(self, question, top_k, video_id=None):
        return [RetrievalResult(Chunk("visual", video_id, "v.mp4", 0, 30), 0.9, "visual")]


def test_hybrid_routes_explicit_range_without_loading_visual_encoder() -> None:
    index = NumpyVectorIndex()
    index.add([Chunk("c1", "video", "v.mp4", 60, 90), Chunk("c2", "video", "v.mp4", 90, 120)], np.eye(2, dtype=np.float32))
    loaded = False

    def build_visual():
        nonlocal loaded
        loaded = True
        return _StubVisualRetriever()

    retriever = HybridRetriever(TemporalRetriever(index), build_visual)
    response = retriever.retrieve("What happened from 1:05 to 1:35?", "video", 2)
    assert response.route == "temporal"
    assert [item.chunk.id for item in response.results] == ["c1", "c2"]
    assert not loaded


def test_hybrid_routes_non_temporal_question_to_visual() -> None:
    index = NumpyVectorIndex()
    retriever = HybridRetriever(TemporalRetriever(index), _StubVisualRetriever)
    response = retriever.retrieve("Which instrument is visible?", "video", 1)
    assert response.route == "visual"
    assert response.results[0].source == "visual"
