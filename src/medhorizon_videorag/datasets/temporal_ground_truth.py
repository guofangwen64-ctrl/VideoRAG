"""Recover temporal evidence stated in MedHorizon QA annotations.

The released QA JSONL has no explicit evidence-span fields. This module only
calls a span ``direct`` when a range or timestamp is stated in the question.
Phase windows inferred from another QA in the same video are kept separate as
weak candidate anchors, not gold temporal evidence.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from .medhorizon import MedHorizonDataset, MedHorizonQA

_TIME_CODE = r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?)"
_RANGE_TIME_CODE = re.compile(
    rf"(?:from|between|spanning|during|interval|at)\s+{_TIME_CODE}\s*(?:to|and|-)\s*(?P<end>\d{{1,2}}:\d{{2}}(?::\d{{2}})?)",
    re.IGNORECASE,
)
_RANGE_SECONDS = re.compile(
    r"(?:from|between|spanning|during|interval|at)\s+(?:second(?:s)?\s*)?(?P<time>\d+(?:\.\d+)?)\s*"
    r"(?:to|and|-)\s*(?:second(?:s)?\s*)?(?P<end>\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_POINT_TIME_CODE = re.compile(rf"(?:around|near|about|at)\s+{_TIME_CODE}", re.IGNORECASE)
_POINT_SECONDS = re.compile(r"(?:around|near|about|at)\s+(?:second(?:s)?\s*)?(?P<time>\d+(?:\.\d+)?)", re.IGNORECASE)


@dataclass(frozen=True)
class TemporalEvidence:
    qa_uid: int | str
    video_key: str
    video_path: str
    method: str
    confidence: str
    windows: list[tuple[float, float]] = field(default_factory=list)
    source_field: str | None = None
    phase: str | None = None
    candidate_count: int = 0
    task_name: str | None = None
    question: str | None = None
    answer: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["windows"] = [{"start_seconds": start, "end_seconds": end} for start, end in self.windows]
        return result


@dataclass(frozen=True)
class TemporalQuery:
    start_seconds: float
    end_seconds: float
    kind: str


def parse_timecode(value: str) -> float:
    units = [float(unit) for unit in value.split(":")]
    if len(units) == 2:
        return units[0] * 60 + units[1]
    if len(units) == 3:
        return units[0] * 3600 + units[1] * 60 + units[2]
    raise ValueError(f"Invalid time code: {value}")


def parse_temporal_query(text: str) -> TemporalQuery | None:
    """Extract an explicit range or point from a user question at inference time."""
    for pattern, parse in ((_RANGE_TIME_CODE, lambda item: parse_timecode(item)), (_RANGE_SECONDS, float)):
        match = pattern.search(text)
        if match:
            start, end = parse(match.group("time")), parse(match.group("end"))
            if end >= start:
                return TemporalQuery(start, end, "range")
    for pattern, parse in ((_POINT_TIME_CODE, lambda item: parse_timecode(item)), (_POINT_SECONDS, float)):
        match = pattern.search(text)
        if match:
            point = parse(match.group("time"))
            return TemporalQuery(point, point, "point")
    return None


def _question_texts(qa: MedHorizonQA) -> list[tuple[str, str]]:
    fields = [("question", qa.question)]
    for name in ("question_original", "question_pre_rewrite_03_22", "question_before_natural_rewrite"):
        value = qa.metadata.get(name)
        if isinstance(value, str) and value and value != qa.question:
            fields.append((name, value))
    return fields


def direct_evidence(qa: MedHorizonQA) -> TemporalEvidence | None:
    for field, text in _question_texts(qa):
        temporal = parse_temporal_query(text)
        if temporal:
            return TemporalEvidence(
                qa.uid, qa.video_key, qa.video_path, f"direct_{temporal.kind}", "high",
                [(temporal.start_seconds, temporal.end_seconds)], field,
                task_name=qa.task_name, question=qa.question, answer=qa.answer,
            )
    return None


def _answer_label(qa: MedHorizonQA) -> str | None:
    if not qa.answer:
        return None
    answer = str(qa.answer).strip()
    for option in qa.options:
        match = re.match(r"\s*([A-Za-z0-9]+)[\.)：:]\s*(.+)", option)
        if match and match.group(1).lower() == answer.lower():
            return match.group(2).strip()
    return None


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def recover_evidence(dataset: MedHorizonDataset) -> list[TemporalEvidence]:
    direct: dict[int | str, TemporalEvidence] = {}
    phase_windows: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for qa in dataset.questions:
        evidence = direct_evidence(qa)
        if evidence:
            direct[qa.uid] = evidence
            if qa.task_name == "Action Recognition":
                phase = _answer_label(qa)
                if phase and evidence.method == "direct_range":
                    phase_windows[qa.video_key][_normalized(phase)].extend(evidence.windows)

    recovered: list[TemporalEvidence] = []
    for qa in dataset.questions:
        if qa.uid in direct:
            recovered.append(direct[qa.uid])
            continue
        text = " ".join(text for _, text in _question_texts(qa))
        candidates = phase_windows.get(qa.video_key, {})
        matched = [key for key in candidates if len(key) >= 4 and key in _normalized(text)]
        if matched:
            windows = [window for key in matched for window in candidates[key]]
            recovered.append(TemporalEvidence(
                qa.uid, qa.video_key, qa.video_path, "phase_anchor", "weak", windows,
                phase=", ".join(sorted(matched)), candidate_count=len(windows), task_name=qa.task_name,
                question=qa.question, answer=qa.answer,
            ))
        else:
            recovered.append(TemporalEvidence(
                qa.uid, qa.video_key, qa.video_path, "unresolved", "none", task_name=qa.task_name,
                question=qa.question, answer=qa.answer,
            ))
    return recovered


def recovery_report(evidence: Iterable[TemporalEvidence]) -> dict[str, Any]:
    rows = list(evidence)
    by_method = Counter(item.method for item in rows)
    by_confidence = Counter(item.confidence for item in rows)
    by_task: dict[str, Counter[str]] = defaultdict(Counter)
    for item in rows:
        by_task[item.task_name or "unknown"][item.method] += 1
    return {
        "questions": len(rows),
        "recovered_any": sum(item.method != "unresolved" for item in rows),
        "recovered_high_confidence": sum(item.confidence == "high" for item in rows),
        "recovered_weak_anchor": sum(item.confidence == "weak" for item in rows),
        "unresolved": sum(item.method == "unresolved" for item in rows),
        "by_method": dict(sorted(by_method.items())),
        "by_confidence": dict(sorted(by_confidence.items())),
        "by_task_and_method": {task: dict(sorted(methods.items())) for task, methods in sorted(by_task.items())},
    }
