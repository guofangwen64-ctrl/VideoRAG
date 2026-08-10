import json
from pathlib import Path

from medhorizon_videorag.datasets import MedHorizonDataset


def test_load_and_report_medhorizon_jsonl(tmp_path: Path) -> None:
    annotation = tmp_path / "split.jsonl"
    annotation.write_text(json.dumps({
        "key": "case-1", "dataset": "demo", "video_path": "demo/case-1.mp4",
        "num_frames": 120, "fps": 2, "duration_seconds": 60, "qa": [
            {"uid": 1, "question": "What happened?", "answer": "A", "options": ["A", "B"],
             "task_id": "C1", "task_name": "Action Recognition", "task_class": "control",
             "category": "temporal_localization", "question_type": ["action recognition"]},
        ],
    }) + "\n", encoding="utf-8")
    dataset = MedHorizonDataset(annotation)

    assert len(dataset.videos) == 1
    assert len(dataset.questions) == 1
    assert dataset.questions[0].video_key == "case-1"
    assert dataset.report()["task_categories"]["task_name"] == {"Action Recognition": 1}
