import numpy as np

from medhorizon_videorag.core.schemas import Chunk
from medhorizon_videorag.retrieval import NumpyVectorIndex


def test_search_can_be_restricted_to_a_video() -> None:
    first = Chunk("a", "video-a", "a.mp4", 0, 30)
    second = Chunk("b", "video-b", "b.mp4", 0, 30)
    index = NumpyVectorIndex()
    index.add([first, second], np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))

    result = index.search(np.array([0.0, 1.0], dtype=np.float32), top_k=1, video_id="video-a")
    assert [hit.chunk.id for hit in result] == ["a"]
