"""Opt-in, answer-free candidate transport with auditable lifecycle states."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from .phase_instrument_reader import (
    _build_track_reader_input,
    _retrieve_tracks_for_events,
    _uniform_sample,
)
from .phase_projection import clip_intervals, intersect_evidence

PHASE_CANDIDATE_FLOW_VERSION = "phase-candidate-flow-v3"


def phase_key(label):
    return " ".join(re.findall(r"[a-z0-9]+", str(label).lower()))


def candidate_id(video_id, segment_id, label, start, end):
    identity = json.dumps(
        [str(video_id), str(segment_id), phase_key(label), float(start), float(end)],
        separators=(",", ":"),
    )
    return (
        f"phase_candidate:{video_id}:"
        + hashlib.sha256(identity.encode()).hexdigest()[:20]
    )


def load_phase_candidates(path, graph):
    """Accept exported JSONL or source sequence JSON; never infer medical labels."""
    path = Path(path)
    if path.suffix == ".jsonl":
        raw = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    else:
        payload = json.loads(path.read_text())
        if payload["video_id"] != graph.video_id:
            raise ValueError("Candidate file video mismatch")
        raw = []
        for segment in payload["segments"]:
            for item in segment.get("phase_candidates", []):
                raw.append(
                    {
                        "video_id": payload["video_id"],
                        "sequence_phase_segment_id": segment["segment_id"],
                        "start_seconds": segment["start_seconds"],
                        "end_seconds": segment["end_seconds"],
                        **item,
                    }
                )
    sources = {
        n.metadata["sequence_phase_segment_id"]: n
        for n in graph.nodes
        if n.node_type == "phase_hypothesis" and n.metadata.get("source_segment")
    }
    results, seen = [], set()
    for item in raw:
        row = deepcopy(item)
        if str(row.get("video_id", graph.video_id)) != graph.video_id:
            raise ValueError("Candidate file video mismatch")
        sid = str(row["sequence_phase_segment_id"])
        start, end = float(row["start_seconds"]), float(row["end_seconds"])
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError("Invalid phase candidate interval")
        decision = str(row.get("decision", "insufficient"))
        if decision not in {"supported", "tentative", "insufficient", "contradicted"}:
            raise ValueError("Unsupported candidate decision")
        if not str(row.get("label", "")).strip():
            raise ValueError("Candidate label is required")
        if (
            not math.isfinite(float(row.get("score", 0)))
            or int(row.get("rank", 999)) < 1
        ):
            raise ValueError("Candidate score/rank is invalid")
        cid = candidate_id(graph.video_id, sid, row["label"], start, end)
        if row.get("candidate_id", cid) != cid or cid in seen:
            raise ValueError("Duplicate or inconsistent stable candidate ID")
        seen.add(cid)
        source_node = sources.get(sid)
        if source_node is None:
            raise ValueError(
                f"Candidate source segment {sid} is absent; reproject sequence phases first"
            )
        source = source_node.metadata["source_segment"]
        if (start, end) != (source["start_seconds"], source["end_seconds"]):
            raise ValueError("Candidate/source interval mismatch")
        persisted = any(
            phase_key(c.get("label")) == phase_key(row["label"])
            and c.get("decision", "insufficient") == decision
            for c in source.get("phase_candidates", [])
        )
        evidence_role = (
            "counter_evidence"
            if decision == "contradicted"
            else (
                "positive"
                if decision in {"supported", "tentative"} and row.get("positive_cues")
                else "uncertain"
            )
        )
        support = []
        for event in graph.nodes:
            if event.node_type != "temporal_event":
                continue
            intervals = [
                e
                for e in clip_intervals(event.evidence, start, end)
                if e.metadata.get("clip_id") in source["supporting_clip_ids"]
            ]
            if intervals:
                support.append(
                    {"event_id": event.id, "evidence": [e.to_dict() for e in intervals]}
                )
        support.sort(
            key=lambda r: (
                min(e["start_seconds"] for e in r["evidence"]),
                r["event_id"],
            )
        )
        results.append(
            {
                **row,
                "candidate_id": cid,
                "video_id": graph.video_id,
                "decision": decision,
                "evidence_role": evidence_role,
                "source_phase_node_id": source_node.id,
                "persistent_candidate_record": persisted,
                "persistent_primary_label_match": phase_key(source_node.label)
                == phase_key(row["label"]),
                "supporting_clip_ids": list(source["supporting_clip_ids"]),
                "observed_pattern": source.get("observed_pattern", ""),
                "event_support": support,
            }
        )
    return results


def candidate_frames(graph, candidate, *, max_clips=2, frames_per_clip=4):
    groups = []
    clips = []
    for node in graph.nodes:
        if (
            node.node_type != "segment"
            or node.metadata.get("clip_id") not in candidate["supporting_clip_ids"]
        ):
            continue
        interval = node.evidence[0]
        if (
            candidate["start_seconds"]
            <= interval.start_seconds
            < interval.end_seconds
            <= candidate["end_seconds"]
        ):
            paths = [p for p in interval.frame_paths if Path(p).is_file()]
            if paths:
                clips.append((node, paths))
    indexes = (
        _uniform_sample([str(i) for i in range(len(clips))], min(max_clips, len(clips)))
        if clips
        else []
    )
    for index in indexes:
        node, paths = clips[int(index)]
        groups.append(
            {
                "candidate_id": candidate["candidate_id"],
                "evidence_role": candidate["evidence_role"],
                "clip_id": node.metadata["clip_id"],
                "start_seconds": node.evidence[0].start_seconds,
                "end_seconds": node.evidence[0].end_seconds,
                "reader_frame_paths": _uniform_sample(
                    paths, min(frames_per_clip, len(paths))
                ),
            }
        )
    return groups


def candidate_reader_packet(
    graph,
    candidate,
    *,
    qa_options=None,
    context_events=1,
    max_tracks=6,
    max_evidence_clips=4,
    frames_per_clip=8,
):
    if candidate["evidence_role"] == "counter_evidence":
        raise ValueError("Counter-evidence cannot be a positive Reader route")
    from .schemas import EvidenceInterval

    support = candidate["event_support"][: context_events + 1]
    windows = {
        r["event_id"]: [EvidenceInterval(**e) for e in r["evidence"]] for r in support
    }
    retrieval = _retrieve_tracks_for_events(graph, list(windows))
    visible = defaultdict(set)
    for edge in graph.edges:
        if (
            edge.relation == "visible_during"
            and edge.target in windows
            and intersect_evidence(edge.evidence, windows[edge.target])
        ):
            visible[edge.source].add(edge.target)
    retrieval["instruments"] = [
        {**t, "supporting_event_ids": sorted(visible[t["track_id"]])}
        for t in retrieval["instruments"]
        if t["track_id"] in visible
    ]
    cid = candidate["candidate_id"]
    retrieval.update(
        phase_hypothesis_id=cid,
        phase_label=candidate["label"],
        phase_confidence=float(candidate.get("confidence_score", 0.35)),
        boundary_id=cid + ":onset",
        boundary_kind="onset",
        boundary_seconds=candidate["start_seconds"],
        event_ids=list(windows),
        evidence=[e.to_dict() for intervals in windows.values() for e in intervals],
        phase_scoped_evidence=True,
        phase_route="phase_candidate_file_v3",
        query_conditioned_phase_candidate=None,
        reasoning_path=[cid, "interval_support", *windows, "inverse:visible_during"],
    )
    packet = _build_track_reader_input(
        graph,
        retrieval,
        qa_options=qa_options,
        max_tracks=max_tracks,
        max_evidence_clips=max_evidence_clips,
        frames_per_clip=frames_per_clip,
    )
    packet["phase_candidate"] = {
        k: deepcopy(candidate.get(k))
        for k in [
            "candidate_id",
            "sequence_phase_segment_id",
            "label",
            "decision",
            "accepted",
            "evidence_role",
            "start_seconds",
            "end_seconds",
            "positive_cues",
            "negative_cues",
            "missing_evidence",
        ]
    }
    return packet


def run_candidate_question(
    graph,
    candidates,
    *,
    question_id,
    question,
    options,
    phase,
    reader=None,
    top_k=3,
    option_verifier=False,
    option_aware=False,
    context_events=1,
    max_tracks=6,
    max_evidence_clips=4,
    frames_per_clip=8,
    verification_min_confidence="medium",
):
    """The interface intentionally accepts neither a reference answer nor GT time."""
    if top_k < 1 or context_events < 0:
        raise ValueError("Invalid candidate/graph limits")
    trace = {
        "version": PHASE_CANDIDATE_FLOW_VERSION,
        "id": str(question_id),
        "video_key": graph.video_id,
        "question": question,
        "qa_options": list(options),
        "phase_query": phase,
        "inference_requested": reader is not None,
        "status": "prepared" if reader is None else "unresolved",
        "retrieval_top1_candidate_id": None,
        "verified_candidate_id": None,
        "reader_candidate_id": None,
        "prediction": None,
        "selected_track_ids": [],
        "candidates": [],
    }
    for candidate in candidates:
        row = deepcopy(candidate)
        row["lifecycle"] = {
            "load": {"status": "kept"},
            "graph": {
                "status": "grounded" if row["event_support"] else "unmapped",
                "persistent_candidate_record": row["persistent_candidate_record"],
            },
            "retrieval": {"status": "not_evaluated"},
            "verification": {"status": "not_run", "reason": "not_reached"},
            "reader": {"status": "not_run", "reason": "not_reached"},
        }
        trace["candidates"].append(row)
    trace["graph_primary_phase_matches"] = [
        n.id
        for n in graph.nodes
        if n.node_type == "phase_hypothesis" and phase_key(n.label) == phase_key(phase)
    ]
    matching = []
    counter = []
    for row in trace["candidates"]:
        stage = row["lifecycle"]["retrieval"]
        if phase_key(row["label"]) != phase_key(phase):
            stage.update(status="discarded", reason="phase_name_mismatch")
        elif not row["event_support"]:
            stage.update(status="discarded", reason="no_graph_interval_support")
        elif row["evidence_role"] == "counter_evidence":
            stage.update(status="counter_only", reason="source_contradicted")
            counter.append(row)
        else:
            matching.append(row)
    matching.sort(
        key=lambda r: (
            -float(r.get("score", 0)),
            int(r.get("rank", 999)),
            r["start_seconds"],
            r["candidate_id"],
        )
    )
    for rank, row in enumerate(matching, 1):
        row["lifecycle"]["retrieval"].update(
            status="kept" if rank <= top_k else "discarded",
            reason="within_top_k" if rank <= top_k else "top_k_limit",
            rank=rank,
        )
    kept = matching[:top_k]
    if kept:
        trace["retrieval_top1_candidate_id"] = kept[0]["candidate_id"]
    groups = []
    for row in [*kept, *counter]:
        frames = candidate_frames(graph, row)
        row["verification_frame_groups"] = frames
        groups.extend(frames)
        row["lifecycle"]["verification"].update(
            status="prepared" if frames else "not_run",
            reason="offline_no_model" if frames else "no_readable_frames",
        )
    trace["verification_input"] = {
        "target_phase": phase,
        "candidates": [
            {
                k: r.get(k)
                for k in [
                    "candidate_id",
                    "label",
                    "decision",
                    "accepted",
                    "evidence_role",
                    "start_seconds",
                    "end_seconds",
                    "score",
                    "positive_cues",
                    "negative_cues",
                    "missing_evidence",
                    "observed_pattern",
                ]
            }
            for r in [*kept, *counter]
        ],
        "evidence_groups": groups,
        "answers_or_options_used": False,
    }
    ready = [r for r in kept if r["verification_frame_groups"]]
    if reader is None:
        for row in kept:
            try:
                row["prepared_reader_input"] = candidate_reader_packet(
                    graph,
                    row,
                    qa_options=options if option_aware else None,
                    context_events=context_events,
                    max_tracks=max_tracks,
                    max_evidence_clips=max_evidence_clips,
                    frames_per_clip=frames_per_clip,
                )
                row["lifecycle"]["reader"].update(
                    status="prepared_not_sent", reason="verification_not_run"
                )
            except ValueError as error:
                row["lifecycle"]["reader"].update(status="not_run", reason=str(error))
        return trace
    if not ready:
        trace["unresolved_reason"] = "no_non_counter_candidate_with_frames"
        return trace
    failure_stage = "verification"
    try:
        for row in [*kept, *counter]:
            row["lifecycle"]["verification"]["status"] = "sent"
        verification = reader.verify_phase_candidates(trace["verification_input"])
        trace["verification_result"] = verification
        assessments = {
            r["candidate_id"]: r for r in verification.get("assessments", [])
        }
        selected = verification.get("selected_candidate_id")
        confidence = {"low": 0, "medium": 1, "high": 2}
        for row in [*kept, *counter]:
            assessment = assessments.get(row["candidate_id"])
            row["lifecycle"]["verification"] = {
                "status": "assessed" if assessment else "unassessed",
                "assessment": assessment,
            }
        chosen = next((r for r in ready if r["candidate_id"] == selected), None)
        assessment = assessments.get(selected, {})
        if (
            chosen is None
            or assessment.get("decision") != "supported"
            or not assessment.get("positive_evidence")
            or confidence.get(assessment.get("confidence"), -1)
            < confidence[verification_min_confidence]
        ):
            trace["unresolved_reason"] = (
                "verification_did_not_accept_an_eligible_candidate"
            )
            for row in kept:
                row["lifecycle"]["reader"] = {
                    "status": "discarded",
                    "reason": "verification_not_accepted",
                }
            return trace
        trace["verified_candidate_id"] = selected
        for row in kept:
            row["lifecycle"]["reader"] = {
                "status": "selected" if row is chosen else "discarded",
                "reason": "verifier_selection",
            }
        failure_stage = "reader"
        packet = candidate_reader_packet(
            graph,
            chosen,
            qa_options=options if option_aware else None,
            context_events=context_events,
            max_tracks=max_tracks,
            max_evidence_clips=max_evidence_clips,
            frames_per_clip=frames_per_clip,
        )
        packet["phase_candidate"]["verification"] = assessment
        packet["phase_counter_evidence"] = [
            {
                "candidate_id": r["candidate_id"],
                "decision": r["decision"],
                "negative_cues": r.get("negative_cues", []),
                "verification_assessment": assessments.get(r["candidate_id"]),
                "start_seconds": r["start_seconds"],
                "end_seconds": r["end_seconds"],
            }
            for r in [*kept, *counter]
            if r["evidence_role"] == "counter_evidence"
            or assessments.get(r["candidate_id"], {}).get("decision") == "contradicted"
        ]
        trace.update(reader_input=packet, reader_candidate_id=selected)
        chosen["lifecycle"]["reader"]["status"] = "sent"
        if option_verifier:
            prediction, rationale, track_ids, options_assessed = (
                reader.answer_phase_instrument_with_option_verifier(
                    question, options, packet
                )
            )
            trace["option_assessments"] = options_assessed
        else:
            prediction, rationale, track_ids = reader.answer_phase_instrument(
                question, options, packet
            )
        chosen["lifecycle"]["reader"]["status"] = "completed"
        trace.update(
            status="completed",
            prediction=prediction,
            rationale=rationale,
            selected_track_ids=track_ids,
        )
    except Exception as error:  # noqa: BLE001
        # Do not serialize provider exceptions: they can contain credentials/URLs.
        trace.update(
            status="failed",
            error_type=type(error).__name__,
            failure_stage=failure_stage,
        )
        for row in [*kept, *counter]:
            stage = row["lifecycle"][failure_stage]
            if stage["status"] in {"sent", "selected"}:
                stage.update(status="failed", reason=type(error).__name__)
    return trace


def full_event_catalog(graph):
    """Do not use truncated event labels as the diagnostic event content."""
    for event in graph.nodes:
        if event.node_type != "temporal_event":
            continue
        clips = set(event.metadata.get("supporting_clip_ids", []))
        yield {
            "event_id": event.id,
            "display_mode": "full",
            "event": event.to_dict(),
            "clips": [
                n.to_dict()
                for n in graph.nodes
                if n.node_type == "segment" and n.metadata.get("clip_id") in clips
            ],
            "atomic_actions": [
                n.to_dict()
                for n in graph.nodes
                if n.node_type == "action_event" and n.metadata.get("clip_id") in clips
            ],
            "entity_mentions": [
                n.to_dict()
                for n in graph.nodes
                if n.node_type == "entity_mention"
                and n.metadata.get("clip_id") in clips
            ],
        }


def render_event_catalog(graph):
    lines = [
        "# 完整事件观察内容",
        "",
        "不以截断的 event label 替代事件内容；下列 observations、actions 和 mentions 均不截取前 N 项。",
        "",
    ]
    for record in full_event_catalog(graph):
        event = record["event"]
        lines.extend(
            [
                f"## {event['id']}",
                "",
                "Predicates: "
                + json.dumps(
                    event["metadata"].get("predicates", []), ensure_ascii=False
                ),
                "",
                "Concepts: "
                + json.dumps(event["metadata"].get("concepts", []), ensure_ascii=False),
                "",
            ]
        )
        for clip in record["clips"]:
            lines.extend(
                [
                    f"### {clip['id']}",
                    "",
                    "```json",
                    json.dumps(
                        clip["metadata"].get("observation", {}),
                        ensure_ascii=False,
                        indent=2,
                    ),
                    "```",
                    "",
                ]
            )
        for name in ["atomic_actions", "entity_mentions"]:
            lines.extend(
                [
                    name + ":",
                    "",
                    "```json",
                    json.dumps(
                        [
                            {k: n[k] for k in ["id", "label", "metadata"]}
                            for n in record[name]
                        ],
                        ensure_ascii=False,
                        indent=2,
                    ),
                    "```",
                    "",
                ]
            )
    return "\n".join(lines)
