from medhorizon_videorag.ingestion import VideoChunker


def test_baseline_uses_consecutive_30_second_windows() -> None:
    windows = list(VideoChunker().time_windows(95.0))
    assert windows == [(0.0, 30.0), (30.0, 60.0), (60.0, 90.0), (90.0, 95.0)]


def test_chunker_rejects_invalid_stride() -> None:
    try:
        list(VideoChunker(stride_seconds=0).time_windows(30.0))
    except ValueError as error:
        assert "stride_seconds" in str(error)
    else:
        raise AssertionError("Expected invalid stride to be rejected")
