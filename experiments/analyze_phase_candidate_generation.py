"""Trace phase-candidate recall and top-k ranking failures.

This diagnostic is intentionally offline: it reads existing second-layer phase
outputs, evidence graphs, and observation-first descriptions without calling a
reader model or an API.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medhorizon_videorag.core.io import write_jsonl
from medhorizon_videorag.datasets import MedHorizonDataset, recover_evidence
from medhorizon_videorag.graph_rag import extract_phase_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="medhorizon_test.jsonl")
    parser.add_argument(
        "--run-root",
        action="append",
        required=True,
        metavar="VIDEO=DIR",
        help="Repeat for each video, e.g. 047=artifacts/graph_rag/047/<run>",
    )
    parser.add_argument(
        "--graph",
        action="append",
        required=True,
        metavar="VIDEO=PATH",
        help="Repeat for each video evidence_graph.json.",
    )
    parser.add_argument(
        "--descriptions",
        action="append",
        required=True,
        metavar="VIDEO=PATH",
        help="Repeat for each video descriptions.jsonl.",
    )
    parser.add_argument(
        "--qa-uids",
        action="append",
        required=True,
        metavar="VIDEO=UID,UID",
        help="Restrict diagnostics to the listed QA ids for each video.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--details", required=True)
    args = parser.parse_args()

    run_roots = _parse_named_paths(args.run_root)
    graphs = _parse_named_paths(args.graph)
    descriptions = _parse_named_paths(args.descriptions)
    qa_uids = {
        video: {item.strip() for item in value.split(",") if item.strip()}
        for video, value in _parse_named_paths(args.qa_uids).items()
    }

    dataset = MedHorizonDataset(args.annotations)
    evidence_by_uid = {
        str(item.qa_uid): item.to_dict() for item in recover_evidence(dataset)
    }
    questions = [
        item
        for item in dataset.questions
        if item.video_key in qa_uids and str(item.uid) in qa_uids[item.video_key]
    ]

    details: list[dict[str, Any]] = []
    for video_id in sorted(qa_uids):
        index = _load_video_index(
            video_id,
            Path(run_roots[video_id]),
            Path(graphs[video_id]),
            Path(descriptions[video_id]),
        )
        video_questions = [item for item in questions if item.video_key == video_id]
        details.extend(
            _diagnose_question(index, item, evidence_by_uid.get(str(item.uid), {}))
            for item in video_questions
        )

    report = {
        "annotations": args.annotations,
        "overall": _summarize(details),
        "videos": {
            video_id: _summarize(
                [row for row in details if row["video_id"] == video_id]
            )
            for video_id in sorted(qa_uids)
        },
        "missing_phase_traces": [
            row for row in details if not row["candidate_generation"]["topk_hit"]
        ],
        "topk_not_top1": [
            row
            for row in details
            if row["candidate_generation"]["topk_hit"]
            and not row["candidate_generation"]["top1_hit"]
        ],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_jsonl(args.details, details)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {output}\nDetails: {args.details}")


def _load_video_index(
    video_id: str, run_root: Path, graph_path: Path, description_path: Path
) -> dict[str, Any]:
    candidates = _read_jsonl(run_root / "phase_candidate_hypotheses.jsonl")
    segments = _read_json(run_root / "sequence_phase_segments.json")["segments"]
    graph = _read_json(graph_path)
    descriptions = _read_jsonl(description_path)

    nodes = graph["nodes"]
    by_type: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        by_type.setdefault(str(node.get("node_type")), []).append(node)
    descriptions_by_clip = {str(row["clip_id"]): row for row in descriptions}
    segment_nodes_by_clip = {
        str(_node_clip_id(node)): node
        for node in by_type.get("segment", [])
        if _node_clip_id(node) is not None
    }
    return {
        "video_id": video_id,
        "run_root": run_root,
        "candidates": candidates,
        "segments": segments,
        "temporal_events": by_type.get("temporal_event", []),
        "action_events": by_type.get("action_event", []),
        "entity_mentions": by_type.get("entity_mention", []),
        "descriptions_by_clip": descriptions_by_clip,
        "segment_nodes_by_clip": segment_nodes_by_clip,
    }


def _diagnose_question(
    index: dict[str, Any], qa: Any, recovered_evidence: dict[str, Any]
) -> dict[str, Any]:
    phase = extract_phase_name(qa.question) or ""
    phase_key = _canonical(phase)
    matching_candidates = [
        item
        for item in index["candidates"]
        if _canonical(str(item.get("label") or "")) == phase_key
    ]
    matching_candidates = sorted(
        matching_candidates,
        key=lambda item: (
            int(item.get("rank") or 999),
            -float(item.get("score") or 0.0),
            float(item.get("start_seconds") or 0.0),
        ),
    )
    top1 = [item for item in matching_candidates if int(item.get("rank") or 999) == 1]
    windows = recovered_evidence.get("windows") or []
    trace_source = "temporal_anchor" if windows else "nearest_lexical_events"
    trace_events = (
        _events_overlapping_windows(index, windows)
        if windows
        else _nearest_events_for_phase(index, phase)
    )

    return {
        "qa_uid": str(qa.uid),
        "video_id": qa.video_key,
        "question": qa.question,
        "gt_phase": phase,
        "gt_answer": qa.answer,
        "options": qa.options,
        "temporal_evidence": {
            "method": recovered_evidence.get("method", "unresolved"),
            "confidence": recovered_evidence.get("confidence", "none"),
            "windows": windows,
            "phase_anchor": recovered_evidence.get("phase"),
            "source_field": recovered_evidence.get("source_field"),
        },
        "candidate_generation": {
            "topk_hit": bool(matching_candidates),
            "top1_hit": bool(top1),
            "best_rank": int(matching_candidates[0].get("rank"))
            if matching_candidates
            else None,
            "match_count": len(matching_candidates),
            "matches": [_compact_candidate(item) for item in matching_candidates[:5]],
            "global_top_candidates": [
                _compact_candidate(item)
                for item in sorted(
                    index["candidates"],
                    key=lambda item: (
                        int(item.get("rank") or 999),
                        -float(item.get("score") or 0.0),
                        float(item.get("start_seconds") or 0.0),
                    ),
                )[:8]
            ],
        },
        "phase_trace": {
            "source": trace_source,
            "events": [_trace_event(index, event) for event in trace_events[:5]],
        },
    }


def _events_overlapping_windows(
    index: dict[str, Any], windows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    events = [
        event
        for event in index["temporal_events"]
        if any(_intervals_overlap(_node_span(event), _window_span(window)) for window in windows)
    ]
    return sorted(events, key=lambda event: _node_span(event)[0])


def _nearest_events_for_phase(index: dict[str, Any], phase: str) -> list[dict[str, Any]]:
    phase_tokens = _tokens(phase)
    scored = []
    for event in index["temporal_events"]:
        text = _event_search_text(index, event)
        matched = phase_tokens & _tokens(text)
        if matched:
            scored.append((len(matched), _node_span(event)[0], event))
    if not scored:
        return []
    return [item[2] for item in sorted(scored, key=lambda item: (-item[0], item[1]))]


def _trace_event(index: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    clip_ids = list((event.get("metadata") or {}).get("supporting_clip_ids") or [])
    span = _node_span(event)
    action_events = [
        node
        for node in index["action_events"]
        if _node_clip_id(node) in set(clip_ids) or _intervals_overlap(_node_span(node), span)
    ]
    entity_mentions = [
        node
        for node in index["entity_mentions"]
        if _node_clip_id(node) in set(clip_ids) or _intervals_overlap(_node_span(node), span)
    ]
    return {
        "event_id": event.get("id"),
        "time": {
            "start_seconds": span[0],
            "end_seconds": span[1],
        },
        "label": event.get("label"),
        "confidence": event.get("confidence"),
        "supporting_clip_ids": clip_ids,
        "metadata": {
            "concepts": list((event.get("metadata") or {}).get("concepts") or [])[:20],
            "predicates": list((event.get("metadata") or {}).get("predicates") or [])[:20],
            "support_score": (event.get("metadata") or {}).get("support_score"),
        },
        "atomic_actions": [_compact_action(node) for node in action_events[:20]],
        "entities": [_compact_entity(node) for node in entity_mentions[:30]],
        "raw_observations": [
            _compact_observation(index, clip_id) for clip_id in clip_ids[:8]
        ],
    }


def _compact_candidate(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "activity_segment_id": item.get("activity_segment_id"),
        "sequence_phase_segment_id": item.get("sequence_phase_segment_id"),
        "rank": item.get("rank"),
        "label": item.get("label"),
        "coarse_phase": item.get("coarse_phase"),
        "decision": item.get("decision"),
        "confidence": item.get("confidence"),
        "score": item.get("score"),
        "accepted": item.get("accepted"),
        "start_seconds": item.get("start_seconds"),
        "end_seconds": item.get("end_seconds"),
        "positive_cues": list(item.get("positive_cues") or [])[:5],
        "negative_cues": list(item.get("negative_cues") or [])[:5],
        "missing_evidence": list(item.get("missing_evidence") or [])[:5],
        "basis": item.get("basis"),
    }


def _compact_action(node: dict[str, Any]) -> dict[str, Any]:
    metadata = node.get("metadata") or {}
    span = _node_span(node)
    return {
        "id": node.get("id"),
        "label": node.get("label"),
        "original_action": metadata.get("original_action"),
        "clip_id": metadata.get("clip_id") or _node_clip_id(node),
        "start_seconds": span[0],
        "end_seconds": span[1],
    }


def _compact_entity(node: dict[str, Any]) -> dict[str, Any]:
    metadata = node.get("metadata") or {}
    span = _node_span(node)
    return {
        "id": node.get("id"),
        "label": node.get("label"),
        "canonical": metadata.get("canonical"),
        "category": metadata.get("category"),
        "source_field": metadata.get("source_field"),
        "attributes": metadata.get("attributes") or {},
        "clip_id": metadata.get("clip_id") or _node_clip_id(node),
        "start_seconds": span[0],
        "end_seconds": span[1],
    }


def _compact_observation(index: dict[str, Any], clip_id: str) -> dict[str, Any]:
    row = index["descriptions_by_clip"].get(clip_id, {})
    description = row.get("description") or {}
    observed = description.get("observed_facts") or {}
    segment = index["segment_nodes_by_clip"].get(clip_id, {})
    return {
        "clip_id": clip_id,
        "clip_index": row.get("clip_index"),
        "start_seconds": row.get("start_seconds"),
        "end_seconds": row.get("end_seconds"),
        "summary": description.get("summary"),
        "visible_instruments": observed.get("visible_instruments") or [],
        "visible_objects": observed.get("visible_objects") or [],
        "visible_anatomy": observed.get("visible_anatomy") or [],
        "actions": observed.get("actions") or [],
        "state_changes": observed.get("state_changes") or [],
        "frame_samples": _sample_frame_paths(segment),
    }


def _sample_frame_paths(node: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for evidence in node.get("evidence") or []:
        paths.extend(str(path) for path in evidence.get("frame_paths") or [])
    if not paths:
        return []
    indexes = sorted({0, len(paths) // 2, len(paths) - 1})
    return [paths[index] for index in indexes]


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    topk_hits = [row for row in rows if row["candidate_generation"]["topk_hit"]]
    top1_hits = [row for row in rows if row["candidate_generation"]["top1_hit"]]
    anchor_rows = [row for row in rows if row["temporal_evidence"]["windows"]]
    missing = [row for row in rows if not row["candidate_generation"]["topk_hit"]]
    return {
        "questions": len(rows),
        "candidate_topk_hits": len(topk_hits),
        "candidate_topk_recall": len(topk_hits) / len(rows) if rows else None,
        "candidate_top1_hits": len(top1_hits),
        "candidate_top1_recall": len(top1_hits) / len(rows) if rows else None,
        "topk_not_top1": len(topk_hits) - len(top1_hits),
        "temporal_anchor_available": len(anchor_rows),
        "missing_with_temporal_anchor": sum(
            bool(row["temporal_evidence"]["windows"]) for row in missing
        ),
        "missing_without_temporal_anchor": sum(
            not row["temporal_evidence"]["windows"] for row in missing
        ),
        "best_rank_counts": dict(
            Counter(
                str(row["candidate_generation"]["best_rank"])
                for row in rows
                if row["candidate_generation"]["best_rank"] is not None
            )
        ),
        "missing_phases": [row["gt_phase"] for row in missing],
    }


def _event_search_text(index: dict[str, Any], event: dict[str, Any]) -> str:
    clip_ids = set((event.get("metadata") or {}).get("supporting_clip_ids") or [])
    pieces = [
        str(event.get("label") or ""),
        " ".join(str(item) for item in (event.get("metadata") or {}).get("concepts") or []),
        " ".join(str(item) for item in (event.get("metadata") or {}).get("predicates") or []),
    ]
    for node in index["action_events"] + index["entity_mentions"]:
        if _node_clip_id(node) in clip_ids:
            pieces.append(str(node.get("label") or ""))
            pieces.append(str((node.get("metadata") or {}).get("original_action") or ""))
            pieces.append(str((node.get("metadata") or {}).get("canonical") or ""))
    return " ".join(pieces)


def _node_clip_id(node: dict[str, Any]) -> str | None:
    metadata = node.get("metadata") or {}
    if metadata.get("clip_id") is not None:
        return str(metadata["clip_id"])
    for evidence in node.get("evidence") or []:
        ev_metadata = evidence.get("metadata") or {}
        if ev_metadata.get("clip_id") is not None:
            return str(ev_metadata["clip_id"])
    return None


def _node_span(node: dict[str, Any]) -> tuple[float, float]:
    starts = []
    ends = []
    for evidence in node.get("evidence") or []:
        if evidence.get("start_seconds") is not None:
            starts.append(float(evidence["start_seconds"]))
        if evidence.get("end_seconds") is not None:
            ends.append(float(evidence["end_seconds"]))
    if not starts or not ends:
        return (0.0, 0.0)
    return (min(starts), max(ends))


def _window_span(window: dict[str, Any]) -> tuple[float, float]:
    return (float(window["start_seconds"]), float(window["end_seconds"]))


def _intervals_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _parse_named_paths(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        if not name or not path:
            raise ValueError(f"Expected NAME=PATH, got {value!r}")
        result[name] = path
    return result


def _tokens(value: str) -> set[str]:
    stop = {
        "and",
        "the",
        "phase",
        "placement",
        "dissection",
        "drain",
        "node",
        "left",
        "right",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if token not in stop and len(token) >= 3
    }


def _canonical(value: str) -> str:
    return "".join(re.findall(r"[a-z0-9]+", value.lower()))


if __name__ == "__main__":
    main()
