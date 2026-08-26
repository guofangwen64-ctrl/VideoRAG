from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .schemas import GraphNode, VideoEvidenceGraph
from .semantic_layer import retrieve_phase_boundary_instruments

PHASE_INSTRUMENT_READER_VERSION = "phase-instrument-track-reader-v1"

_GENERIC_FAMILIES = {"generic_instrument", "generic_object"}


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
    """Build an answer-free visual packet from phase-boundary appearance tracks."""
    if max_tracks < 1:
        raise ValueError("max_tracks must be at least 1")
    if max_evidence_clips < 1:
        raise ValueError("max_evidence_clips must be at least 1")
    if frames_per_clip < 1:
        raise ValueError("frames_per_clip must be at least 1")

    retrieval = retrieve_phase_boundary_instruments(
        graph,
        phase_label,
        boundary_kind=boundary_kind,
        context_events=context_events,
    )
    node_by_id = {node.id: node for node in graph.nodes}
    clips = {
        str(node.metadata.get("clip_id")): node
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
    for rank, candidate in enumerate(candidates, start=1):
        candidate["graph_rank"] = rank

    groups_by_clip: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        for clip_id in candidate["evidence_clip_ids"]:
            if clip_id in groups_by_clip:
                groups_by_clip[clip_id]["track_ids"].append(candidate["track_id"])
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
            if len(groups_by_clip) >= max_evidence_clips:
                break
        if len(groups_by_clip) >= max_evidence_clips:
            break

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
            f"Phase {phase_label!r} has no candidate tracks with readable frames"
        )
    return {
        "reader_version": PHASE_INSTRUMENT_READER_VERSION,
        "phase_label": retrieval["phase_label"],
        "phase_hypothesis_id": retrieval["phase_hypothesis_id"],
        "phase_confidence": retrieval["phase_confidence"],
        "boundary_id": retrieval["boundary_id"],
        "boundary_kind": retrieval["boundary_kind"],
        "boundary_seconds": retrieval["boundary_seconds"],
        "event_ids": retrieval["event_ids"],
        "candidate_tracks": candidates,
        "evidence_groups": evidence_groups,
        "reasoning_path": retrieval["reasoning_path"],
        "answers_used_for_retrieval": False,
        "qa_options_used_for_retrieval": False,
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
