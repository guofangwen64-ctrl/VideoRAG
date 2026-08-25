from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .schemas import GraphNode, VideoEvidenceGraph

SEQUENCE_PHASE_VERSION = "observation-sequence-phase-v1"
_CONFIDENCE = {"high": 0.9, "medium": 0.65, "low": 0.35}


def load_observation_sequence(path: str | Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("Observation description JSONL is empty")
    rows.sort(key=lambda row: (int(row["clip_index"]), float(row["start_seconds"])))
    clip_ids = [str(row.get("clip_id", "")) for row in rows]
    if any(not clip_id for clip_id in clip_ids):
        raise ValueError("Every observation row requires clip_id")
    if len(clip_ids) != len(set(clip_ids)):
        raise ValueError("Observation rows contain duplicate clip IDs")
    video_ids = {str(row.get("video_id", "")) for row in rows}
    if len(video_ids) != 1 or "" in video_ids:
        raise ValueError("Observation rows must belong to exactly one video")
    indices = [int(row["clip_index"]) for row in rows]
    if indices != sorted(indices) or len(indices) != len(set(indices)):
        raise ValueError("Observation clip indices must be unique and ordered")
    return rows


def compact_observation_sequence(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    compact = []
    for row in rows:
        description = row.get("description", {})
        if not isinstance(description, dict):
            raise TypeError("description must be a JSON object")
        facts = description.get("observed_facts", {})
        if not isinstance(facts, dict):
            facts = {}
        compact.append(
            {
                "clip_id": str(row["clip_id"]),
                "start_seconds": float(row["start_seconds"]),
                "end_seconds": float(row["end_seconds"]),
                "summary": str(description.get("summary", "")).strip(),
                "visible_instruments": _string_list(facts.get("visible_instruments")),
                "visible_objects": _string_list(facts.get("visible_objects")),
                "actions": _compact_actions(facts.get("actions")),
                "state_changes": _string_list(facts.get("state_changes")),
            }
        )
    return compact


def build_sequence_phase_prompt(
    observations: Sequence[dict[str, Any]], phase_labels: Sequence[str]
) -> str:
    if not observations:
        raise ValueError("Sequence phase inference requires observations")
    labels = [str(item).strip() for item in phase_labels if str(item).strip()]
    if not labels:
        raise ValueError("Sequence phase inference requires candidate phase labels")
    return (
        "You are inferring an explicitly uncertain sequence-level phase segmentation "
        "for one long medical procedure video. You receive the complete ordered "
        "sequence of observation-first clip descriptions. These descriptions contain "
        "only visible evidence. Do not use a video title, dataset metadata, expected "
        "procedure order, question answers, or outside medical context. Use temporal "
        "continuity across multiple adjacent clips, but do not invent a phase when the "
        "observations are insufficient.\n\n"
        "Return ONLY one valid JSON object with this schema:\n"
        '{"phase_segments":[{"label":"exact candidate or unknown",'
        '"start_clip_id":"...","end_clip_id":"...",'
        '"confidence":"high|medium|low","basis_clip_ids":["..."],'
        '"basis":"brief observation-grounded explanation"}]}\n\n'
        "Rules:\n"
        "1. Cover every supplied clip exactly once, in order, with contiguous and "
        "non-overlapping segments.\n"
        "2. Boundaries may occur only between supplied clips.\n"
        "3. Use only an exact candidate label below or 'unknown'.\n"
        "4. Prefer a stable phase across adjacent clips when observations remain "
        "compatible; create a boundary only when the visible activity changes.\n"
        "5. A phase label is a medical hypothesis, not an observed fact. Use 'unknown' "
        "when distinctive evidence is absent.\n"
        "6. basis_clip_ids must contain 1-5 supplied clip IDs inside that segment.\n"
        "7. Do not identify instruments or anatomy beyond the supplied "
        "observations.\n\n"
        "Candidate phases:\n- "
        + "\n- ".join(labels)
        + "\n\nOrdered observations:\n"
        + json.dumps(list(observations), ensure_ascii=False, separators=(",", ":"))
    )


def normalize_sequence_phase_response(
    payload: dict[str, Any],
    observations: Sequence[dict[str, Any]],
    phase_labels: Sequence[str],
) -> list[dict[str, Any]]:
    raw_segments = payload.get("phase_segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError("Response requires a non-empty phase_segments list")
    ids = [str(item["clip_id"]) for item in observations]
    position = {clip_id: index for index, clip_id in enumerate(ids)}
    label_map = {_canonical(label): str(label) for label in phase_labels}
    label_map["unknown"] = "unknown"
    normalized = []
    next_position = 0
    for number, item in enumerate(raw_segments):
        if not isinstance(item, dict):
            raise TypeError("Every phase segment must be a JSON object")
        start_id = str(item.get("start_clip_id", ""))
        end_id = str(item.get("end_clip_id", ""))
        if start_id not in position or end_id not in position:
            raise ValueError(f"Phase segment {number} references an unknown clip")
        start = position[start_id]
        end = position[end_id]
        if start != next_position or end < start:
            raise ValueError("Phase segments must be ordered, contiguous, and disjoint")
        label = label_map.get(_canonical(item.get("label", "")))
        if label is None:
            raise ValueError(f"Phase segment {number} uses a non-candidate label")
        confidence = _confidence_label(item.get("confidence"))
        support = item.get("basis_clip_ids", [])
        if not isinstance(support, list):
            raise TypeError("basis_clip_ids must be a list")
        support = list(dict.fromkeys(str(value) for value in support))
        valid_support = set(ids[start : end + 1])
        if not 1 <= len(support) <= 5 or any(
            clip_id not in valid_support for clip_id in support
        ):
            raise ValueError("basis_clip_ids must contain 1-5 IDs inside the segment")
        normalized.append(
            {
                "segment_id": f"sequence_phase:{number:05d}",
                "label": label,
                "start_clip_id": start_id,
                "end_clip_id": end_id,
                "start_seconds": float(observations[start]["start_seconds"]),
                "end_seconds": float(observations[end]["end_seconds"]),
                "confidence": confidence,
                "confidence_score": _CONFIDENCE[confidence],
                "supporting_clip_ids": ids[start : end + 1],
                "basis_clip_ids": support,
                "basis": str(item.get("basis", "")).strip(),
                "fact_status": "medical_hypothesis",
            }
        )
        next_position = end + 1
    if next_position != len(ids):
        raise ValueError(
            "Phase segments do not cover the complete observation sequence"
        )
    return normalized


def project_sequence_phases_to_events(
    graph: VideoEvidenceGraph,
    segments: Sequence[dict[str, Any]],
    *,
    source: str,
) -> list[dict[str, Any]]:
    segment_by_clip = {
        clip_id: segment
        for segment in segments
        for clip_id in segment["supporting_clip_ids"]
    }
    events = sorted(
        (node for node in graph.nodes if node.node_type == "temporal_event"),
        key=_node_start,
    )
    rows = []
    for event in events:
        clip_ids = [
            str(value) for value in event.metadata.get("supporting_clip_ids", [])
        ]
        matched = [
            segment_by_clip[item] for item in clip_ids if item in segment_by_clip
        ]
        if not matched:
            label = "unknown"
            confidence = "low"
            basis = "No supporting clip was covered by the sequence segmentation."
            segment_ids: list[str] = []
        else:
            votes = Counter(str(item["label"]) for item in matched)
            label = min(votes, key=lambda value: (-votes[value], value))
            winning = [item for item in matched if item["label"] == label]
            confidence = min(
                (str(item["confidence"]) for item in winning),
                key=lambda value: _CONFIDENCE[value],
            )
            segment_ids = list(
                dict.fromkeys(str(item["segment_id"]) for item in winning)
            )
            basis = (
                f"Sequence-level phase segment vote covers {votes[label]}/"
                f"{len(matched)} supporting clips; sources: {', '.join(segment_ids)}."
            )
        rows.append(
            {
                "event_id": event.id,
                "source": source,
                "phase_hypothesis": {
                    "label": label,
                    "confidence": confidence,
                    "basis": basis,
                },
                "instrument_hypotheses": [],
                "sequence_phase_segment_ids": segment_ids,
                "candidate_aware_diagnostic": True,
                "fact_status": "medical_hypothesis",
            }
        )
    return rows


def _compact_actions(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if not isinstance(item, dict):
            continue
        result.append(
            {
                key: str(item.get(key, "")).strip()
                for key in ("subject", "action", "target")
            }
        )
    return result


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _confidence_label(value: Any) -> str:
    lowered = str(value or "low").strip().lower()
    if lowered not in _CONFIDENCE:
        raise ValueError(f"Unsupported confidence label: {value}")
    return lowered


def _canonical(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).lower()))


def _node_start(node: GraphNode) -> float:
    return min(item.start_seconds for item in node.evidence)
