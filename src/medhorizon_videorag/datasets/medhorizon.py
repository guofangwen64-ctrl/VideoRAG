"""Typed reader for MedHorizon JSONL annotations.

Each JSONL row is a video record with its associated QA list.  The reader keeps
the original metadata available while exposing flattened QA records for training,
evaluation, and dataset analysis.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class MedHorizonVideo:
    key: str
    dataset: str
    video_path: str
    num_frames: int | None
    fps: float | None
    duration_seconds: float | None
    organ: str | None = None
    scene_type: str | None = None
    duration_tier: str | None = None
    split: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MedHorizonQA:
    uid: int | str
    video_key: str
    video_path: str
    question: str
    answer: str | None
    options: list[str] = field(default_factory=list)
    task_id: str | None = None
    task_name: str | None = None
    task_class: str | None = None
    category: str | None = None
    question_type: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class MedHorizonDataset:
    """Load MedHorizon annotation JSONL and expose videos plus flattened QA data."""

    def __init__(self, annotation_path: str | Path) -> None:
        self.annotation_path = Path(annotation_path)
        self.videos: list[MedHorizonVideo] = []
        self.questions: list[MedHorizonQA] = []
        self._load()

    def _load(self) -> None:
        with self.annotation_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON on line {line_number} of {self.annotation_path}") from error
                self._add_record(record, line_number)

    def _add_record(self, record: dict[str, Any], line_number: int) -> None:
        key = self._required(record, "key", line_number)
        video_path = self._required(record, "video_path", line_number)
        video_fields = {
            "key", "dataset", "video_path", "num_frames", "fps", "duration_seconds",
            "organ", "scene_type", "duration_tier", "split", "qa",
        }
        self.videos.append(MedHorizonVideo(
            key=str(key), dataset=str(record.get("dataset", "unknown")), video_path=str(video_path),
            num_frames=self._as_int(record.get("num_frames")), fps=self._as_float(record.get("fps")),
            duration_seconds=self._as_float(record.get("duration_seconds")),
            organ=record.get("organ"), scene_type=record.get("scene_type"),
            duration_tier=record.get("duration_tier"), split=record.get("split"),
            metadata={name: value for name, value in record.items() if name not in video_fields},
        ))
        for qa in record.get("qa", []):
            if "uid" not in qa or "question" not in qa:
                raise ValueError(f"Video {key} on line {line_number} contains a QA without uid or question")
            qa_fields = {"uid", "question", "answer", "options", "task_id", "task_name", "task_class", "category", "question_type"}
            raw_types = qa.get("question_type", [])
            types = [str(item) for item in (raw_types if isinstance(raw_types, list) else [raw_types])]
            self.questions.append(MedHorizonQA(
                uid=qa["uid"], video_key=str(key), video_path=str(video_path), question=str(qa["question"]),
                answer=qa.get("answer"), options=list(qa.get("options", [])), task_id=qa.get("task_id"),
                task_name=qa.get("task_name"), task_class=qa.get("task_class"), category=qa.get("category"),
                question_type=types, metadata={name: value for name, value in qa.items() if name not in qa_fields},
            ))

    @staticmethod
    def _required(record: dict[str, Any], field_name: str, line_number: int) -> Any:
        if field_name not in record:
            raise ValueError(f"Missing '{field_name}' on line {line_number}")
        return record[field_name]

    @staticmethod
    def _as_float(value: Any) -> float | None:
        return None if value is None else float(value)

    @staticmethod
    def _as_int(value: Any) -> int | None:
        return None if value is None else int(value)

    def iter_videos(self) -> Iterator[MedHorizonVideo]:
        return iter(self.videos)

    def iter_questions(self) -> Iterator[MedHorizonQA]:
        return iter(self.questions)

    def report(self) -> dict[str, Any]:
        durations = [video.duration_seconds for video in self.videos if video.duration_seconds is not None]
        task_names = Counter(question.task_name or "unknown" for question in self.questions)
        task_classes = Counter(question.task_class or "unknown" for question in self.questions)
        categories = Counter(question.category or "unknown" for question in self.questions)
        question_types = Counter(item for question in self.questions for item in question.question_type) or Counter({"unknown": len(self.questions)})
        return {
            "source": str(self.annotation_path),
            "videos": {
                "count": len(self.videos),
                "with_duration": len(durations),
                "total_duration_seconds": round(sum(durations), 3),
                "average_duration_seconds": round(sum(durations) / len(durations), 3) if durations else None,
            },
            "questions": {"count": len(self.questions), "per_video": round(len(self.questions) / len(self.videos), 3) if self.videos else 0.0},
            "task_categories": {"task_name": dict(sorted(task_names.items())), "task_class": dict(sorted(task_classes.items())), "category": dict(sorted(categories.items()))},
            "question_types": dict(sorted(question_types.items())),
        }
