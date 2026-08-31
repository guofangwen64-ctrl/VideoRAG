"""Lossless sequence intervals and conservative event/clip intersections."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from copy import deepcopy
from typing import Any

from .schemas import EvidenceInterval, GraphEdge, GraphNode, VideoEvidenceGraph

PHASE_PROJECTION_VERSION = "sequence-event-intersection-v3"
_CONFIDENCE = {"high": 0.9, "medium": 0.65, "low": 0.35}


def clip_intervals(
    intervals: Sequence[EvidenceInterval], start: float, end: float
) -> list[EvidenceInterval]:
    result = []
    for item in intervals:
        left, right = max(start, item.start_seconds), min(end, item.end_seconds)
        if right <= left:  # Half-open intervals: a shared endpoint is not support.
            continue
        partial = left != item.start_seconds or right != item.end_seconds
        metadata = dict(item.metadata)
        if partial:
            metadata.update(
                source_interval_seconds=[item.start_seconds, item.end_seconds],
                frame_policy="withheld_without_per_frame_timestamps",
            )
        result.append(
            EvidenceInterval(
                item.video_id,
                left,
                right,
                [] if partial else list(item.frame_paths),
                item.confidence,
                metadata,
            )
        )
    return result


def intersect_evidence(
    evidence: Sequence[EvidenceInterval], windows: Sequence[EvidenceInterval]
) -> list[EvidenceInterval]:
    result = []
    seen = set()
    for item in evidence:
        for window in windows:
            if item.video_id != window.video_id:
                continue
            a, b = item.metadata.get("clip_id"), window.metadata.get("clip_id")
            if a and b and a != b:
                continue
            for overlap in clip_intervals(
                [item], window.start_seconds, window.end_seconds
            ):
                key = (overlap.start_seconds, overlap.end_seconds, a)
                if key not in seen:
                    seen.add(key)
                    result.append(overlap)
    return result


def interval_duration(intervals: Sequence[EvidenceInterval]) -> float:
    """Union duration avoids duplicate evidence and does not fill temporal gaps."""
    total, end = 0.0, -1.0
    for item in sorted(intervals, key=lambda x: (x.start_seconds, x.end_seconds)):
        total += max(0.0, item.end_seconds - max(end, item.start_seconds))
        end = max(end, item.end_seconds)
    return total


def project_intersections(
    graph: VideoEvidenceGraph, segments: Sequence[dict[str, Any]], *, source: str
) -> list[dict[str, Any]]:
    events = sorted(
        (n for n in graph.nodes if n.node_type == "temporal_event"),
        key=lambda n: (min(e.start_seconds for e in n.evidence), n.id),
    )
    if not events:
        raise ValueError("Sequence projection requires temporal events")
    catalog = deepcopy(list(segments))
    if not catalog:
        raise ValueError("Sequence segments must not be empty")
    seen = set()
    for segment in catalog:
        sid = segment["segment_id"]
        start, end = float(segment["start_seconds"]), float(segment["end_seconds"])
        if sid in seen or not sid:
            raise ValueError("Sequence segment IDs must be unique and non-empty")
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError("Invalid sequence segment interval")
        seen.add(sid)
    rows = []
    for event in events:
        overlaps = []
        for segment in catalog:
            start, end = float(segment["start_seconds"]), float(segment["end_seconds"])
            allowed = set(segment["supporting_clip_ids"])
            evidence = clip_intervals(event.evidence, start, end)
            evidence = [
                e
                for e in evidence
                if not e.metadata.get("clip_id") or e.metadata["clip_id"] in allowed
            ]
            seconds = interval_duration(evidence)
            if not seconds:
                continue
            overlaps.append(
                {
                    "sequence_phase_segment_id": segment["segment_id"],
                    "label": segment["label"],
                    "confidence": segment["confidence"],
                    "source_start_seconds": start,
                    "source_end_seconds": end,
                    "overlap_seconds": seconds,
                    "event_overlap_ratio": seconds / interval_duration(event.evidence),
                    "phase_overlap_ratio": seconds / (end - start),
                    "supporting_clip_ids": list(
                        dict.fromkeys(
                            e.metadata["clip_id"]
                            for e in evidence
                            if e.metadata.get("clip_id")
                        )
                    ),
                    "evidence": [e.to_dict() for e in evidence],
                }
            )
        # Retain a scalar field for old callers, but never assign a mixed event
        # to a winning label. Only the interval records carry phase support.
        full = len(overlaps) == 1 and math.isclose(
            overlaps[0]["event_overlap_ratio"], 1.0
        )
        rows.append(
            {
                "event_id": event.id,
                "source": source,
                "phase_projection_version": PHASE_PROJECTION_VERSION,
                "phase_overlaps": overlaps,
                "phase_hypothesis": {
                    "label": overlaps[0]["label"] if full else "unknown",
                    "confidence": overlaps[0]["confidence"] if full else "low",
                    "basis": "Single source segment covers the event."
                    if full
                    else "Mixed or incomplete coverage; use phase_overlaps, not a whole-event label.",
                },
                "sequence_phase_segment_ids": [
                    o["sequence_phase_segment_id"] for o in overlaps
                ],
                "instrument_hypotheses": [],
                "candidate_aware_diagnostic": True,
                "fact_status": "medical_hypothesis",
            }
        )
    # Keep even source segments with no event overlap, without repeating a large
    # source catalog in every JSONL row. Semantic augmentation validates it.
    rows[0]["sequence_phase_segments"] = catalog
    return rows


def append_sequence_phases(
    graph: VideoEvidenceGraph,
    rows: Sequence[dict[str, Any]],
    nodes: list[GraphNode],
    edges: list[GraphEdge],
) -> tuple[dict[str, dict[str, list[EvidenceInterval]]], int]:
    catalogs = [
        row["sequence_phase_segments"]
        for row in rows
        if "sequence_phase_segments" in row
    ]
    if len(catalogs) != 1:
        raise ValueError(
            "Interval projection requires exactly one source segment catalog"
        )
    catalog = catalogs[0]
    expected = project_intersections(graph, catalog, source="validation")
    actual = {row["event_id"]: row for row in rows}
    if set(actual) != {row["event_id"] for row in expected}:
        raise ValueError("Interval projection must cover the current graph event set")
    for row in expected:
        supplied = actual[row["event_id"]]
        if (
            supplied.get("phase_projection_version") != PHASE_PROJECTION_VERSION
            or supplied.get("phase_overlaps") != row["phase_overlaps"]
        ):
            raise ValueError(
                "Stale or invalid interval projection; reproject onto the current graph"
            )
    membership = {}
    for index, segment in enumerate(catalog):
        sid = segment["segment_id"]
        pid = f"phase_hypothesis:{graph.video_id}:{index:05d}"
        start, end = float(segment["start_seconds"]), float(segment["end_seconds"])
        confidence = _CONFIDENCE[str(segment["confidence"])]
        support = [
            (row, overlap)
            for row in rows
            for overlap in row["phase_overlaps"]
            if overlap["sequence_phase_segment_id"] == sid
        ]
        evidence = [
            EvidenceInterval(**e) for _, overlap in support for e in overlap["evidence"]
        ]
        source_interval = EvidenceInterval(
            graph.video_id,
            start,
            end,
            metadata={
                "grounding_source": "sequence_description_interval",
                "event_support_available": bool(support),
            },
        )
        clip_durations = sorted(
            {
                e.end_seconds - e.start_seconds
                for n in graph.nodes
                if n.node_type == "segment"
                for e in n.evidence
                if e.metadata.get("clip_id", n.metadata.get("clip_id"))
                in segment["supporting_clip_ids"]
            }
        )
        common = {
            "derived": True,
            "fact_status": "medical_hypothesis",
            "phase_projection_version": PHASE_PROJECTION_VERSION,
            "sequence_phase_segment_id": sid,
            "source_start_seconds": start,
            "source_end_seconds": end,
            "boundary_accuracy_seconds": None,
            "source_clip_duration_seconds": clip_durations,
        }
        nodes.append(
            GraphNode(
                pid,
                graph.video_id,
                "phase_hypothesis",
                segment["label"],
                evidence or [source_interval],
                confidence=confidence,
                metadata={
                    **common,
                    "canonical_label": " ".join(
                        re.findall(r"[a-z0-9]+", str(segment["label"]).lower())
                    ),
                    "supporting_event_ids": [row["event_id"] for row, _ in support],
                    "source_supporting_clip_ids": list(segment["supporting_clip_ids"]),
                    "supporting_clip_ids": list(
                        dict.fromkeys(
                            e.metadata["clip_id"]
                            for e in evidence
                            if e.metadata.get("clip_id")
                        )
                    ),
                    "source_segment": deepcopy(segment),
                    "phase_candidates": deepcopy(segment.get("phase_candidates", [])),
                    "hypothesis_source": list(
                        dict.fromkeys(row.get("source", "") for row in rows)
                    ),
                    "bases": [segment.get("basis", "")],
                },
            )
        )
        for row, overlap in support:
            intervals = [EvidenceInterval(**e) for e in overlap["evidence"]]
            membership.setdefault(row["event_id"], {})[pid] = intervals
            edges.append(
                GraphEdge(
                    pid,
                    row["event_id"],
                    "derived_from",
                    intervals,
                    confidence,
                    {**common, **{k: v for k, v in overlap.items() if k != "evidence"}},
                )
            )
        for kind, timestamp in [("onset", start), ("offset", end)]:
            bid = f"phase_boundary:{graph.video_id}:{index:05d}:{kind}"
            anchors = [
                (
                    row["event_id"],
                    [
                        EvidenceInterval(**e)
                        for e in overlap["evidence"]
                        if (
                            e["start_seconds"] <= timestamp < e["end_seconds"]
                            if kind == "onset"
                            else e["start_seconds"] < timestamp <= e["end_seconds"]
                        )
                    ],
                )
                for row, overlap in support
            ]
            anchors = [(eid, intervals) for eid, intervals in anchors if intervals]
            boundary_evidence = [e for _, intervals in anchors for e in intervals] or [
                source_interval
            ]
            nodes.append(
                GraphNode(
                    bid,
                    graph.video_id,
                    "phase_boundary",
                    f"{kind} of {segment['label']}",
                    boundary_evidence,
                    confidence=confidence,
                    metadata={
                        **common,
                        "boundary_kind": kind,
                        "timestamp_seconds": timestamp,
                        "phase_hypothesis_id": pid,
                        "grounding_event_ids": [eid for eid, _ in anchors],
                        "grounding_event_id": anchors[0][0] if anchors else None,
                    },
                )
            )
            edges.append(
                GraphEdge(
                    pid,
                    bid,
                    "has_boundary",
                    boundary_evidence,
                    confidence,
                    {**common, "boundary_kind": kind},
                )
            )
            for eid, intervals in anchors:
                edges.append(
                    GraphEdge(bid, eid, "grounded_by", intervals, confidence, common)
                )
    return membership, len(catalog)
