from medhorizon_videorag.core.schemas import Chunk, RetrievalResult
from medhorizon_videorag.datasets.temporal_ground_truth import TemporalEvidence
from medhorizon_videorag.evaluation import evaluate_retrieval, temporal_iou


def test_temporal_iou_and_retrieval_metrics() -> None:
    assert temporal_iou((0, 60), (0, 30)) == 0.5
    evidence = [
        TemporalEvidence(1, "video", "video.mp4", "direct_range", "high", [(0, 60)]),
        TemporalEvidence(2, "video", "video.mp4", "direct_point", "high", [(75, 75)]),
    ]
    retrieved = [[RetrievalResult(Chunk("c1", "video", "video.mp4", 0, 30), 1.0)], [
        RetrievalResult(Chunk("c2", "video", "video.mp4", 60, 90), 1.0),
    ]]
    report, _ = evaluate_retrieval(evidence, retrieved, [1], 0.3)
    assert report["range_metrics"]["at_1"]["recall_at_1"] == 1.0
    assert report["point_metrics"]["at_1"]["point_hit_at_1"] == 1.0
