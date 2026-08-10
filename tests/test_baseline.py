from pathlib import Path

from medhorizon_videorag.core.config import ExperimentConfig
from medhorizon_videorag.core.schemas import Chunk, QAExample
from medhorizon_videorag.evaluation import evaluate_predictions
from medhorizon_videorag.pipelines import build_index, run_qa


def test_index_and_qa(tmp_path: Path) -> None:
    config = ExperimentConfig(
        vision={"provider": "deterministic", "embedding_dim": 16},
        retrieval={"index_path": str(tmp_path / "index"), "top_k": 1},
        llm={"provider": "extractive"},
    )
    build_index([Chunk("c1", "v1", "v1.mp4", 0, 8), Chunk("c2", "v1", "v1.mp4", 8, 16)], config)
    predictions = run_qa([QAExample("q1", "v1", "v1.mp4", "何时开始操作？", "8秒")], config)
    assert len(predictions) == 1
    assert predictions[0].evidence
    assert evaluate_predictions(predictions)["count"] == 1
