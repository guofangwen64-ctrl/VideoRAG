from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class QAExample:
    id: str
    video_id: str
    video_path: str
    question: str
    answer: str | None = None
    choices: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Chunk:
    id: str
    video_id: str
    video_path: str
    start_seconds: float
    end_seconds: float
    frame_paths: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalResult:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Prediction:
    id: str
    question: str
    prediction: str
    evidence: list[dict[str, Any]]
    reference: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
