"""Reproject saved sequence phases onto an observation graph without inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medhorizon_videorag.graph_rag import (
    augment_with_semantic_hypotheses,
    load_evidence_graph,
    project_sequence_phases_to_events,
    retrieve_phase_boundary_instruments,
    write_semantic_layer_artifacts,
)
from medhorizon_videorag.graph_rag.phase_projection import PHASE_PROJECTION_VERSION


def audit_projection(base, graph, segments):
    """Check serialized graph relations against source times, not scalar labels."""
    nodes = {n.id: n for n in graph.nodes}
    sources = {s["segment_id"]: s for s in segments}
    phases = [n for n in graph.nodes if n.node_type == "phase_hypothesis"]
    violations = []
    details = []
    if Counter(n.metadata.get("sequence_phase_segment_id") for n in phases) != Counter(
        sources.keys()
    ):
        violations.append("source_segment_set_changed")
    if (
        any(nodes.get(n.id) != n for n in base.nodes)
        or graph.edges[: len(base.edges)] != base.edges
    ):
        violations.append("observation_graph_changed")
    for phase in phases:
        sid = phase.metadata.get("sequence_phase_segment_id")
        source = sources[sid]
        start, end = source["start_seconds"], source["end_seconds"]
        if phase.metadata.get("source_segment") != source:
            violations.append(f"source_segment_changed:{sid}")
        boundaries = [
            n
            for n in graph.nodes
            if n.node_type == "phase_boundary"
            and n.metadata.get("phase_hypothesis_id") == phase.id
        ]
        if Counter(
            (n.metadata["boundary_kind"], n.metadata["timestamp_seconds"])
            for n in boundaries
        ) != Counter([("onset", start), ("offset", end)]):
            violations.append(f"source_boundary_changed:{sid}")
        projected = [
            e
            for e in graph.edges
            if e.source == phase.id and e.relation == "derived_from"
        ]
        for event in [n for n in base.nodes if n.node_type == "temporal_event"]:
            expected = sorted(
                (
                    max(start, e.start_seconds),
                    min(end, e.end_seconds),
                    e.metadata.get("clip_id", ""),
                )
                for e in event.evidence
                if max(start, e.start_seconds) < min(end, e.end_seconds)
                and (
                    not e.metadata.get("clip_id")
                    or e.metadata["clip_id"] in source["supporting_clip_ids"]
                )
            )
            links = [e for e in projected if e.target == event.id]
            actual = sorted(
                (e.start_seconds, e.end_seconds, e.metadata.get("clip_id", ""))
                for link in links
                for e in link.evidence
            )
            if expected != actual or len(links) != bool(expected):
                violations.append(f"intersection_mismatch:{sid}:{event.id}")
            for link in links:
                intervals = sorted(
                    (e.start_seconds, e.end_seconds) for e in link.evidence
                )
                total, right = 0, -1
                for left, stop in intervals:
                    total += max(0, stop - max(left, right))
                    right = max(right, stop)
                if abs(link.metadata["overlap_seconds"] - total) > 1e-8:
                    violations.append(f"duration_mismatch:{sid}:{event.id}")
                details.append(
                    {
                        "segment_id": sid,
                        "phase_label": phase.label,
                        "event_id": event.id,
                        "source_interval": [start, end],
                        "overlap_seconds": total,
                        "event_overlap_ratio": link.metadata["event_overlap_ratio"],
                        "phase_overlap_ratio": link.metadata["phase_overlap_ratio"],
                        "intervals": actual,
                    }
                )
        related = [
            e
            for e in graph.edges
            if (e.source == phase.id and e.relation == "derived_from")
            or (e.target == phase.id and e.relation == "co_occurs")
            or (e.source in {n.id for n in boundaries} and e.relation == "grounded_by")
        ]
        for edge in related:
            for interval in edge.evidence:
                if not start <= interval.start_seconds < interval.end_seconds <= end:
                    violations.append(f"outside_phase_evidence:{sid}")
                cid = interval.metadata.get("clip_id")
                if cid and cid not in source["supporting_clip_ids"]:
                    violations.append(f"outside_phase_clip:{sid}:{cid}")
        if phase.label != "unknown":
            for kind in ["onset", "offset"]:
                retrieval = retrieve_phase_boundary_instruments(
                    graph, phase.label, boundary_kind=kind, context_events=2
                )
                # Repeated labels may select an earlier same-label source phase.
                selected = nodes[retrieval["phase_hypothesis_id"]].metadata[
                    "source_segment"
                ]
                if any(
                    not selected["start_seconds"]
                    <= e["start_seconds"]
                    < e["end_seconds"]
                    <= selected["end_seconds"]
                    for e in retrieval["evidence"]
                ):
                    violations.append(f"retrieval_outside_phase:{sid}")
    report = {
        "video_id": graph.video_id,
        "projection_version": PHASE_PROJECTION_VERSION,
        "source_segment_count": len(segments),
        "persistent_phase_count": len(phases),
        "unknown_phase_count": sum(n.label == "unknown" for n in phases),
        "boundary_count": sum(n.node_type == "phase_boundary" for n in graph.nodes),
        "event_count": sum(n.node_type == "temporal_event" for n in base.nodes),
        "phase_event_intersection_count": len(details),
        "violations": violations,
        "violation_count": len(violations),
        "passed": not violations,
        "scope": "Source interval consistency; not clinical phase or QA accuracy",
    }
    return report, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--sequence-phases", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--instrument-track-source",
        choices=["appearance_mentions", "semantic_hypotheses"],
        default="appearance_mentions",
    )
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    graph = load_evidence_graph(args.graph)
    payload = json.loads(Path(args.sequence_phases).read_text())
    if payload["video_id"] != graph.video_id:
        raise ValueError("Sequence phases and graph belong to different videos")
    rows = project_sequence_phases_to_events(
        graph,
        payload["segments"],
        source=f"saved_sequence:{payload.get('version', 'unspecified')}",
    )
    artifacts = augment_with_semantic_hypotheses(
        graph, rows, instrument_track_source=args.instrument_track_source
    )
    report, details = audit_projection(graph, artifacts.graph, payload["segments"])
    if not report["passed"]:
        raise ValueError(f"Projection acceptance failed: {report['violations']}")
    write_semantic_layer_artifacts(artifacts, output)
    assert (
        load_evidence_graph(output / "semantic_evidence_graph.json").to_dict()
        == artifacts.graph.to_dict()
    )
    report["schema_roundtrip_passed"] = True
    (output / "projection_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    (output / "projection_details.jsonl").write_text(
        "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in details)
    )
    (output / "sequence_phase_segments.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    (output / "run_metadata.json").write_text(
        json.dumps(
            {
                "projection_version": PHASE_PROJECTION_VERSION,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "graph_sha256": hashlib.sha256(
                    Path(args.graph).read_bytes()
                ).hexdigest(),
                "sequence_phases_sha256": hashlib.sha256(
                    Path(args.sequence_phases).read_bytes()
                ).hexdigest(),
                "instrument_track_source": args.instrument_track_source,
                "answers_used": False,
                "inference_performed": False,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
