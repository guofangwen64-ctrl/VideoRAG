from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .schemas import GraphNode, VideoEvidenceGraph
from .semantic_layer import retrieve_phase_boundary_instruments

PHASE_INSTRUMENT_READER_VERSION = "phase-instrument-track-reader-v2-query-fallback"

_GENERIC_FAMILIES = {"generic_instrument", "generic_object"}
_CONFIDENCE = {"high": 0.9, "medium": 0.65, "low": 0.35}


def load_open_activity_segments(
    path: str | Path, *, video_id: str | None = None
) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("segments"), list):
        raise TypeError("Open activity artifact must contain a segments list")
    artifact_video_id = str(payload.get("video_id", ""))
    if video_id is not None and artifact_video_id != str(video_id):
        raise ValueError(
            f"Open activity video {artifact_video_id} does not match {video_id}"
        )
    rows = []
    seen: set[str] = set()
    previous_end = -1.0
    for raw in payload["segments"]:
        if not isinstance(raw, dict):
            raise TypeError("Every open activity segment must be an object")
        row = dict(raw)
        segment_id = str(row.get("segment_id", ""))
        supporting = row.get("supporting_clip_ids")
        if not segment_id or segment_id in seen:
            raise ValueError("Open activity segment IDs must be non-empty and unique")
        if not isinstance(supporting, list) or not supporting:
            raise ValueError(f"Open activity segment {segment_id} has no clips")
        start = float(row.get("start_seconds", -1))
        end = float(row.get("end_seconds", -1))
        if start < 0 or end <= start or start < previous_end:
            raise ValueError(
                "Open activity segments must be ordered and non-overlapping"
            )
        seen.add(segment_id)
        previous_end = end
        rows.append(row)
    if not rows:
        raise ValueError("Open activity artifact contains no segments")
    return rows


def build_open_activity_catalog(
    segments: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return observation-derived activity summaries with local sequence context."""
    catalog = []
    for index, segment in enumerate(segments):
        catalog.append(
            {
                "segment_id": str(segment["segment_id"]),
                "sequence_index": index,
                "start_seconds": float(segment["start_seconds"]),
                "end_seconds": float(segment["end_seconds"]),
                "activity_label": str(segment.get("activity_label", "")),
                "observed_pattern": str(segment.get("observed_pattern", "")),
                "previous_activity": str(segments[index - 1].get("activity_label", ""))
                if index
                else None,
                "next_activity": str(segments[index + 1].get("activity_label", ""))
                if index + 1 < len(segments)
                else None,
            }
        )
    return catalog


def select_activity_candidate_frame_groups(
    graph: VideoEvidenceGraph,
    candidates: Sequence[dict[str, Any]],
    *,
    max_clips_per_segment: int = 2,
    frames_per_clip: int = 4,
) -> list[dict[str, Any]]:
    if max_clips_per_segment < 1:
        raise ValueError("max_clips_per_segment must be at least 1")
    if frames_per_clip < 1:
        raise ValueError("frames_per_clip must be at least 1")
    clips = {
        _segment_clip_id(node): node
        for node in graph.nodes
        if node.node_type == "segment"
    }
    groups = []
    for candidate in candidates:
        segment_id = str(candidate["segment_id"])
        clip_ids = [
            str(item)
            for item in candidate.get("basis_clip_ids", [])
            or candidate.get("supporting_clip_ids", [])
            if str(item) in clips
        ]
        selected_clip_ids = _uniform_sample(
            clip_ids, min(max_clips_per_segment, len(clip_ids))
        )
        for clip_id in selected_clip_ids:
            clip = clips[clip_id]
            source_frames = [
                path for path in clip.evidence[0].frame_paths if Path(path).is_file()
            ]
            if not source_frames:
                continue
            groups.append(
                {
                    "segment_id": segment_id,
                    "clip_id": clip_id,
                    "start_seconds": clip.evidence[0].start_seconds,
                    "end_seconds": clip.evidence[0].end_seconds,
                    "reader_frame_paths": _uniform_sample(
                        source_frames, min(frames_per_clip, len(source_frames))
                    ),
                    "selection": "query_conditioned_phase_verification",
                }
            )
    if not groups:
        raise ValueError("Activity candidates have no readable frame evidence")
    return groups


def build_phase_instrument_reader_input(
    graph: VideoEvidenceGraph,
    phase_label: str,
    *,
    boundary_kind: str = "onset",
    context_events: int = 1,
    max_tracks: int = 6,
    max_evidence_clips: int = 4,
    frames_per_clip: int = 8,
) -> dict[str, Any]:
    """Build an answer-free visual packet from a persistent phase hypothesis."""
    retrieval = retrieve_phase_boundary_instruments(
        graph,
        phase_label,
        boundary_kind=boundary_kind,
        context_events=context_events,
    )
    retrieval["phase_route"] = "persistent_phase_hypothesis"
    retrieval["query_conditioned_phase_candidate"] = None
    return _build_track_reader_input(
        graph,
        retrieval,
        max_tracks=max_tracks,
        max_evidence_clips=max_evidence_clips,
        frames_per_clip=frames_per_clip,
    )


def build_query_conditioned_phase_reader_input(
    graph: VideoEvidenceGraph,
    phase_label: str,
    activity_segment: dict[str, Any],
    *,
    verification_confidence: str,
    verification_rationale: str,
    context_events: int = 1,
    max_tracks: int = 6,
    max_evidence_clips: int = 4,
    frames_per_clip: int = 8,
) -> dict[str, Any]:
    """Build a temporary phase route without mutating the persistent graph."""
    if context_events < 0:
        raise ValueError("context_events must be non-negative")
    supporting_clips = {
        str(item) for item in activity_segment.get("supporting_clip_ids", [])
    }
    events = sorted(
        (node for node in graph.nodes if node.node_type == "temporal_event"),
        key=lambda node: (node.evidence[0].start_seconds, node.id),
    )
    overlapping = [
        index
        for index, event in enumerate(events)
        if supporting_clips
        & {str(item) for item in event.metadata.get("supporting_clip_ids", [])}
    ]
    if not overlapping:
        raise ValueError(
            f"Activity segment {activity_segment.get('segment_id')} maps to no events"
        )
    onset_index = overlapping[0]
    selected_events = events[onset_index : onset_index + context_events + 1]
    event_ids = [event.id for event in selected_events]
    retrieval = _retrieve_tracks_for_events(graph, event_ids)
    segment_id = str(activity_segment["segment_id"])
    retrieval.update(
        {
            "phase_hypothesis_id": f"query_conditioned_phase:{graph.video_id}:{segment_id}",
            "phase_label": phase_label,
            "phase_match_score": _CONFIDENCE.get(verification_confidence, 0.35),
            "phase_confidence": _CONFIDENCE.get(verification_confidence, 0.35),
            "boundary_id": f"query_conditioned_boundary:{graph.video_id}:{segment_id}:onset",
            "boundary_kind": "onset",
            "boundary_seconds": float(activity_segment["start_seconds"]),
            "event_ids": event_ids,
            "evidence": [
                interval.to_dict()
                for event in selected_events
                for interval in event.evidence
            ],
            "reasoning_path": [
                f"query_phase:{phase_label}",
                "query_conditioned_match",
                segment_id,
                "maps_to_onset_event",
                *event_ids,
                "inverse:visible_during",
                *[item["track_id"] for item in retrieval["instruments"]],
            ],
            "phase_route": "query_conditioned_activity_fallback",
            "query_conditioned_phase_candidate": {
                "segment_id": segment_id,
                "activity_label": activity_segment.get("activity_label"),
                "observed_pattern": activity_segment.get("observed_pattern"),
                "verification_confidence": verification_confidence,
                "verification_rationale": verification_rationale,
                "persistent_graph_mutated": False,
            },
        }
    )
    return _build_track_reader_input(
        graph,
        retrieval,
        max_tracks=max_tracks,
        max_evidence_clips=max_evidence_clips,
        frames_per_clip=frames_per_clip,
    )


def _retrieve_tracks_for_events(
    graph: VideoEvidenceGraph, event_ids: Sequence[str]
) -> dict[str, Any]:
    selected = set(event_ids)
    node_by_id = {node.id: node for node in graph.nodes}
    tracks = {
        edge.source: node_by_id[edge.source]
        for edge in graph.edges
        if edge.relation == "visible_during" and edge.target in selected
    }
    instruments = []
    for track in tracks.values():
        instruments.append(
            {
                "track_id": track.id,
                "label": track.label,
                "canonical_label": track.metadata.get("canonical_label"),
                "canonical_instrument": track.metadata.get(
                    "canonical_instrument", "unknown"
                ),
                "appearance_signature": track.metadata.get("appearance_signature"),
                "surface_forms": track.metadata.get("surface_forms", []),
                "tracking_scope": track.metadata.get("tracking_scope"),
                "physical_identity_confirmed": track.metadata.get(
                    "physical_identity_confirmed"
                ),
                "fact_status": track.metadata.get("fact_status"),
                "confidence": track.confidence,
                "supporting_event_ids": [
                    event_id
                    for event_id in track.metadata.get("supporting_event_ids", [])
                    if event_id in selected
                ],
            }
        )
    instruments.sort(
        key=lambda item: (-float(item["confidence"]), str(item["canonical_label"]))
    )
    return {"instruments": instruments}


def _build_track_reader_input(
    graph: VideoEvidenceGraph,
    retrieval: dict[str, Any],
    *,
    max_tracks: int,
    max_evidence_clips: int,
    frames_per_clip: int,
) -> dict[str, Any]:
    if max_tracks < 1:
        raise ValueError("max_tracks must be at least 1")
    if max_evidence_clips < 1:
        raise ValueError("max_evidence_clips must be at least 1")
    if frames_per_clip < 1:
        raise ValueError("frames_per_clip must be at least 1")

    node_by_id = {node.id: node for node in graph.nodes}
    clips = {
        _segment_clip_id(node): node
        for node in graph.nodes
        if node.node_type == "segment"
    }
    event_order = {
        event_id: index for index, event_id in enumerate(retrieval["event_ids"])
    }
    candidates = []
    for item in retrieval["instruments"]:
        track = node_by_id[str(item["track_id"])]
        detections = [
            detection
            for detection in track.metadata.get("detections", [])
            if str(detection.get("event_id")) in event_order
        ]
        action_roles = sorted(
            {
                str(role)
                for detection in detections
                for role in detection.get("action_roles", [])
            }
        )
        evidence_clip_ids = _track_evidence_clips(
            track, detections, node_by_id, clips, event_order
        )
        if not evidence_clip_ids:
            continue
        signature = item.get("appearance_signature") or {}
        family = str(item.get("canonical_label") or "")
        onset_support = any(
            event_order[str(detection["event_id"])] == 0 for detection in detections
        )
        descriptor_count = sum(
            len(signature.get(name, []))
            for name in ("colors", "shapes", "markings", "appearance", "sizes")
        )
        selected_support = len(item.get("supporting_event_ids", []))
        score = (
            0.4 * float(item.get("confidence", 0.0))
            + 0.25 * float(onset_support)
            + 0.12 * float(family not in _GENERIC_FAMILIES)
            + 0.08 * float(bool(action_roles))
            + 0.05 * min(descriptor_count, 2)
            + 0.05 * min(selected_support, 2)
        )
        candidates.append(
            {
                "track_id": track.id,
                "label": track.label,
                "canonical_instrument": track.metadata.get(
                    "canonical_instrument", "unknown"
                ),
                "appearance_family": family,
                "appearance_signature": signature,
                "surface_forms": list(item.get("surface_forms", [])),
                "action_roles": action_roles,
                "supporting_event_ids": list(item.get("supporting_event_ids", [])),
                "evidence_clip_ids": evidence_clip_ids,
                "graph_score": round(score, 4),
                "score_components": {
                    "track_confidence": item.get("confidence"),
                    "onset_support": onset_support,
                    "specific_appearance_family": family not in _GENERIC_FAMILIES,
                    "has_action_role": bool(action_roles),
                    "descriptor_count": descriptor_count,
                    "selected_event_support": selected_support,
                },
                "fact_status": item.get("fact_status"),
                "physical_identity_confirmed": item.get("physical_identity_confirmed"),
            }
        )

    candidates.sort(
        key=lambda item: (
            -float(item["graph_score"]),
            str(item["appearance_family"]),
            str(item["track_id"]),
        )
    )
    candidates = candidates[:max_tracks]
    groups_by_clip: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        for clip_id in candidate["evidence_clip_ids"]:
            if clip_id in groups_by_clip:
                groups_by_clip[clip_id]["track_ids"].append(candidate["track_id"])
                continue
            if len(groups_by_clip) >= max_evidence_clips:
                continue
            clip = clips[clip_id]
            source_frames = [
                path for path in clip.evidence[0].frame_paths if Path(path).is_file()
            ]
            if not source_frames:
                continue
            groups_by_clip[clip_id] = {
                "clip_id": clip_id,
                "start_seconds": clip.evidence[0].start_seconds,
                "end_seconds": clip.evidence[0].end_seconds,
                "track_ids": [candidate["track_id"]],
                "reader_frame_paths": _uniform_sample(
                    source_frames, min(frames_per_clip, len(source_frames))
                ),
                "selection": "appearance_track_grounded_clip",
            }

    evidence_groups = list(groups_by_clip.values())
    included_tracks = {
        track_id for group in evidence_groups for track_id in group["track_ids"]
    }
    candidates = [
        candidate
        for candidate in candidates
        if candidate["track_id"] in included_tracks
    ]
    displayed_clip_ids = set(groups_by_clip)
    for rank, candidate in enumerate(candidates, start=1):
        candidate["graph_rank"] = rank
        candidate["reader_clip_ids"] = [
            clip_id
            for clip_id in candidate["evidence_clip_ids"]
            if clip_id in displayed_clip_ids
        ]
    if not candidates or not evidence_groups:
        raise ValueError(
            f"Phase {retrieval['phase_label']!r} has no tracks with readable frames"
        )
    return {
        "reader_version": PHASE_INSTRUMENT_READER_VERSION,
        "phase_label": retrieval["phase_label"],
        "phase_hypothesis_id": retrieval["phase_hypothesis_id"],
        "phase_confidence": retrieval["phase_confidence"],
        "phase_route": retrieval["phase_route"],
        "query_conditioned_phase_candidate": retrieval[
            "query_conditioned_phase_candidate"
        ],
        "boundary_id": retrieval["boundary_id"],
        "boundary_kind": retrieval["boundary_kind"],
        "boundary_seconds": retrieval["boundary_seconds"],
        "event_ids": retrieval["event_ids"],
        "candidate_tracks": candidates,
        "evidence_groups": evidence_groups,
        "reasoning_path": retrieval["reasoning_path"],
        "answers_used_for_retrieval": False,
        "qa_options_used_for_retrieval": False,
        "persistent_graph_mutated": False,
    }


def _track_evidence_clips(
    track: GraphNode,
    detections: Sequence[dict[str, Any]],
    node_by_id: dict[str, GraphNode],
    clips: dict[str, GraphNode],
    event_order: dict[str, int],
) -> list[str]:
    ranked: list[tuple[int, str]] = []
    seen: set[str] = set()
    for detection in sorted(
        detections, key=lambda item: event_order[str(item["event_id"])]
    ):
        for mention_id in detection.get("mention_ids", []):
            mention = node_by_id.get(str(mention_id))
            if mention is None:
                continue
            clip_id = str(mention.metadata.get("clip_id", ""))
            if clip_id in clips and clip_id not in seen:
                seen.add(clip_id)
                ranked.append((event_order[str(detection["event_id"])], clip_id))
    if ranked:
        return [item[1] for item in sorted(ranked)]

    for mention_id in track.metadata.get("source_mention_ids", []):
        mention = node_by_id.get(str(mention_id))
        if mention is None:
            continue
        clip_id = str(mention.metadata.get("clip_id", ""))
        if clip_id in clips and clip_id not in seen:
            seen.add(clip_id)
            ranked.append((len(event_order), clip_id))
    return [item[1] for item in sorted(ranked)]


def _uniform_sample(items: Sequence[str], count: int) -> list[str]:
    if count < 1 or not items:
        raise ValueError("Cannot sample an empty frame sequence")
    if count == 1:
        return [str(items[len(items) // 2])]
    positions = [
        round(index * (len(items) - 1) / (count - 1)) for index in range(count)
    ]
    return [str(items[position]) for position in positions]


def _segment_clip_id(node: GraphNode) -> str:
    clip_id = str(node.metadata.get("clip_id", ""))
    if not clip_id and node.evidence:
        clip_id = str(node.evidence[0].metadata.get("clip_id", ""))
    if not clip_id and node.id.startswith("clip:"):
        clip_id = node.id.removeprefix("clip:")
    return clip_id
