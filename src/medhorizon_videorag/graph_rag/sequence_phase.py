from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .schemas import GraphNode, VideoEvidenceGraph

SEQUENCE_PHASE_VERSION = "observation-sequence-phase-v1"
TWO_STAGE_SEQUENCE_PHASE_VERSION = "observation-sequence-phase-two-stage-v1"
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


def build_open_activity_segmentation_prompt(
    observations: Sequence[dict[str, Any]],
) -> str:
    if not observations:
        raise ValueError("Open activity segmentation requires observations")
    return (
        "Segment the complete ordered observation sequence of one long medical "
        "procedure video into contiguous visible-activity states. This is NOT a "
        "medical phase classification task. No phase candidate list is available. "
        "Describe only recurring visible actions, instruments, objects, and state "
        "changes from the supplied observations. Do not use external context, an "
        "expected procedure order, or inferred anatomy.\n\n"
        "Return ONLY one valid JSON object with this schema:\n"
        '{"activity_segments":[{"activity_label":"short literal activity",'
        '"start_clip_id":"...","end_clip_id":"...",'
        '"confidence":"high|medium|low","basis_clip_ids":["..."],'
        '"observed_pattern":"visible evidence recurring in the segment",'
        '"boundary_reason":"visible change from the previous segment or video start"}]}'
        "\n\nRules:\n"
        "1. activity_segments MUST NOT be empty. Cover every supplied clip exactly "
        "once, in order, with contiguous and "
        "non-overlapping segments.\n"
        "2. Boundaries may occur only between supplied clips and require a visible "
        "change in activity, tool/object pattern, or scene state.\n"
        "3. Do not use named surgical phases, diagnoses, or inferred anatomy in "
        "activity_label or observed_pattern.\n"
        "4. Do not split merely because a clip boundary exists; preserve long-running "
        "activity when the visible pattern remains compatible.\n"
        "5. basis_clip_ids must contain 1-5 supplied clip IDs inside that segment.\n\n"
        "If no reliable internal boundary is visible, return one segment from the "
        "first clip through the last clip. A literal label such as 'repeated tool and "
        "thread-like material manipulation' is valid; an empty list is invalid.\n\n"
        "Ordered observations:\n"
        + json.dumps(list(observations), ensure_ascii=False, separators=(",", ":"))
    )


def normalize_open_activity_response(
    payload: dict[str, Any], observations: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    raw_segments = payload.get("activity_segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError("Response requires a non-empty activity_segments list")
    ids = [str(item["clip_id"]) for item in observations]
    position = {clip_id: index for index, clip_id in enumerate(ids)}
    normalized = []
    next_position = 0
    for number, item in enumerate(raw_segments):
        if not isinstance(item, dict):
            raise TypeError("Every activity segment must be a JSON object")
        start_id = str(item.get("start_clip_id", ""))
        end_id = str(item.get("end_clip_id", ""))
        if start_id not in position or end_id not in position:
            raise ValueError(f"Activity segment {number} references an unknown clip")
        start = position[start_id]
        end = position[end_id]
        if start != next_position or end < start:
            raise ValueError(
                "Activity segments must be ordered, contiguous, and disjoint"
            )
        support = _validate_basis_clip_ids(item, ids[start : end + 1])
        label = str(item.get("activity_label", "")).strip()
        pattern = str(item.get("observed_pattern", "")).strip()
        if not label or not pattern:
            raise ValueError(
                "Activity segments require activity_label and observed_pattern"
            )
        normalized.append(
            {
                "segment_id": f"open_activity:{number:05d}",
                "activity_label": label,
                "start_clip_id": start_id,
                "end_clip_id": end_id,
                "start_seconds": float(observations[start]["start_seconds"]),
                "end_seconds": float(observations[end]["end_seconds"]),
                "confidence": _confidence_label(item.get("confidence")),
                "supporting_clip_ids": ids[start : end + 1],
                "basis_clip_ids": support,
                "observed_pattern": pattern,
                "boundary_reason": str(item.get("boundary_reason", "")).strip(),
                "fact_status": "derived_observation_summary",
            }
        )
        next_position = end + 1
    if next_position != len(ids):
        raise ValueError(
            "Activity segments do not cover the complete observation sequence"
        )
    return normalized


def build_strict_phase_mapping_prompt(
    activity_segments: Sequence[dict[str, Any]], phase_labels: Sequence[str]
) -> str:
    if not activity_segments:
        raise ValueError("Strict phase mapping requires activity segments")
    labels = [str(item).strip() for item in phase_labels if str(item).strip()]
    if not labels:
        raise ValueError("Strict phase mapping requires candidate phase labels")
    return (
        "Map open-vocabulary visible-activity segments to an explicitly uncertain "
        "medical phase ontology. The activity segmentation was produced without phase "
        "labels. Evaluate each segment independently from its visible evidence, while "
        "using adjacent segments only to understand continuity. Do not use question "
        "answers, a video title, dataset metadata, or an expected phase order.\n\n"
        "Return ONLY one valid JSON object with this schema:\n"
        '{"phase_mappings":[{"segment_id":"open_activity:...",'
        '"label":"exact candidate or unknown","decision":"supported|insufficient",'
        '"confidence":"high|medium|low","distinctive_cues":["..."],'
        '"missing_evidence":["..."],"basis":"brief explanation"}]}\n\n'
        "Rules:\n"
        "1. Return exactly one mapping for every supplied segment_id, in order.\n"
        "2. Use decision='supported' only when visible cues distinguish the selected "
        "phase from other plausible candidates.\n"
        "3. Generic suturing cues are insufficient by themselves: thread passing, "
        "pulling, tightening, red fluid, or an instrument near tissue do not identify "
        "which named suturing phase is occurring.\n"
        "4. If multiple candidates remain plausible, use label='unknown', "
        "decision='insufficient', and list the missing distinctive evidence.\n"
        "5. Do not force a complete named-phase timeline. Unknown is preferred over a "
        "weak label.\n"
        "6. distinctive_cues may contain only evidence already stated in the activity "
        "segment.\n\n"
        "Candidate phases:\n- "
        + "\n- ".join(labels)
        + "\n\nOpen activity segments:\n"
        + json.dumps(list(activity_segments), ensure_ascii=False, separators=(",", ":"))
    )


def normalize_strict_phase_mapping_response(
    payload: dict[str, Any],
    activity_segments: Sequence[dict[str, Any]],
    phase_labels: Sequence[str],
) -> list[dict[str, Any]]:
    mappings = payload.get("phase_mappings")
    if not isinstance(mappings, list):
        raise TypeError("Response requires a phase_mappings list")
    expected_ids = [str(item["segment_id"]) for item in activity_segments]
    mapping_ids = [
        str(item.get("segment_id", "")) for item in mappings if isinstance(item, dict)
    ]
    if mapping_ids != expected_ids or len(mappings) != len(expected_ids):
        raise ValueError(
            "Phase mappings must cover every activity segment once in order"
        )
    label_map = {_canonical(label): str(label) for label in phase_labels}
    label_map["unknown"] = "unknown"
    normalized = []
    for number, (activity, item) in enumerate(zip(activity_segments, mappings)):
        if not isinstance(item, dict):
            raise TypeError("Every phase mapping must be a JSON object")
        requested_label = label_map.get(_canonical(item.get("label", "")))
        if requested_label is None:
            raise ValueError(f"Phase mapping {number} uses a non-candidate label")
        decision = str(item.get("decision", "insufficient")).strip().lower()
        if decision not in {"supported", "insufficient"}:
            raise ValueError("Phase mapping decision must be supported or insufficient")
        confidence = _confidence_label(item.get("confidence"))
        distinctive_cues = _string_list(item.get("distinctive_cues"))
        accepted = (
            requested_label != "unknown"
            and decision == "supported"
            and confidence in {"high", "medium"}
            and bool(distinctive_cues)
        )
        label = requested_label if accepted else "unknown"
        final_confidence = confidence if accepted else "low"
        normalized.append(
            {
                "segment_id": f"sequence_phase:{number:05d}",
                "activity_segment_id": activity["segment_id"],
                "label": label,
                "requested_label": requested_label,
                "mapping_accepted": accepted,
                "decision": decision,
                "start_clip_id": activity["start_clip_id"],
                "end_clip_id": activity["end_clip_id"],
                "start_seconds": activity["start_seconds"],
                "end_seconds": activity["end_seconds"],
                "confidence": final_confidence,
                "confidence_score": _CONFIDENCE[final_confidence],
                "supporting_clip_ids": activity["supporting_clip_ids"],
                "basis_clip_ids": activity["basis_clip_ids"],
                "activity_label": activity["activity_label"],
                "observed_pattern": activity["observed_pattern"],
                "distinctive_cues": distinctive_cues,
                "missing_evidence": _string_list(item.get("missing_evidence")),
                "basis": str(item.get("basis", "")).strip(),
                "fact_status": "medical_hypothesis",
                "mapping_protocol": TWO_STAGE_SEQUENCE_PHASE_VERSION,
            }
        )
    return normalized


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


def _validate_basis_clip_ids(
    item: dict[str, Any], valid_clip_ids: Sequence[str]
) -> list[str]:
    support = item.get("basis_clip_ids", [])
    if not isinstance(support, list):
        raise TypeError("basis_clip_ids must be a list")
    support = list(dict.fromkeys(str(value) for value in support))
    valid = set(valid_clip_ids)
    if not support or any(clip_id not in valid for clip_id in support):
        raise ValueError("basis_clip_ids must contain IDs inside the segment")
    if len(support) > 5:
        positions = [round(index * (len(support) - 1) / 4) for index in range(5)]
        support = [support[position] for position in positions]
    return support


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
