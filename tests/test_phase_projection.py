import json
from copy import deepcopy
from dataclasses import replace

import pytest

from medhorizon_videorag.graph_rag import (
    EvidenceInterval,
    GraphNode,
    VideoEvidenceGraph,
    augment_with_semantic_hypotheses,
    build_phase_instrument_reader_input,
    load_evidence_graph,
    project_sequence_phases_to_events,
    retrieve_phase_boundary_instruments,
    write_semantic_layer_artifacts,
)


def _fixture(tmp_path):
    intervals, nodes = [], []
    for index in range(8):
        cid = f"c{index}"
        frame = tmp_path / f"{cid}.jpg"
        frame.write_bytes(b"frame")
        interval = EvidenceInterval(
            "case",
            640 + index * 64,
            704 + index * 64,
            [str(frame)],
            metadata={"clip_id": cid},
        )
        intervals.append(interval)
        nodes.append(
            GraphNode(
                f"clip:{cid}",
                "case",
                "segment",
                cid,
                [interval],
                metadata={"clip_id": cid},
            )
        )
        nodes.append(
            GraphNode(
                f"mention:{cid}",
                "case",
                "entity_mention",
                f"tool {index}",
                [interval],
                metadata={
                    "clip_id": cid,
                    "category": "instrument",
                    "canonical": "generic_instrument",
                    "source_field": "visible_instruments",
                    "attributes": {"color": [f"color{index}"]},
                },
            )
        )
    for index, group in enumerate([intervals[:3], intervals[3:]]):
        nodes.append(
            GraphNode(
                f"event:{index}",
                "case",
                "temporal_event",
                f"event {index}",
                group,
                metadata={
                    "supporting_clip_ids": [e.metadata["clip_id"] for e in group]
                },
            )
        )
    graph = VideoEvidenceGraph("case", nodes, [])
    segments = []
    for index, (label, start, end) in enumerate(
        [("unknown", 0, 2), ("Left Atrium Dissection", 2, 5), ("Suturing", 5, 8)]
    ):
        segments.append(
            {
                "segment_id": f"phase:{index}",
                "label": label,
                "start_seconds": 640 + start * 64,
                "end_seconds": 640 + end * 64,
                "supporting_clip_ids": [f"c{i}" for i in range(start, end)],
                "confidence": "medium",
                "basis": "source observation",
                "phase_candidates": [{"label": label, "score": 0.7}],
            }
        )
    return graph, segments


def _augment(graph, segments):
    return augment_with_semantic_hypotheses(
        graph,
        project_sequence_phases_to_events(graph, segments, source="test"),
        instrument_track_source="appearance_mentions",
    )


def test_short_phase_survives_two_coarse_events_with_original_boundaries(tmp_path):
    graph, segments = _fixture(tmp_path)
    before = deepcopy(graph.to_dict())
    rows = project_sequence_phases_to_events(graph, segments, source="test")
    assert [r["phase_hypothesis"]["label"] for r in rows] == ["unknown", "unknown"]
    short = [
        o
        for r in rows
        for o in r["phase_overlaps"]
        if o["label"] == "Left Atrium Dissection"
    ]
    assert [o["overlap_seconds"] for o in short] == [64, 128]
    assert [o["event_overlap_ratio"] for o in short] == pytest.approx([1 / 3, 2 / 5])
    assert sum(o["phase_overlap_ratio"] for o in short) == pytest.approx(1)
    result = _augment(graph, segments)
    phases = [n for n in result.graph.nodes if n.node_type == "phase_hypothesis"]
    assert len(phases) == 3  # unknown is not erased either.
    assert phases[1].metadata["source_segment"] == segments[1]
    assert phases[1].metadata["supporting_event_ids"] == ["event:0", "event:1"]
    bounds = [
        n
        for n in result.graph.nodes
        if n.node_type == "phase_boundary"
        and n.metadata["phase_hypothesis_id"] == phases[1].id
    ]
    assert [n.metadata["timestamp_seconds"] for n in bounds] == [768, 960]
    assert all(n.metadata["boundary_accuracy_seconds"] is None for n in bounds)
    assert graph.to_dict() == before
    output = tmp_path / "roundtrip"
    write_semantic_layer_artifacts(result, output)
    assert (
        load_evidence_graph(output / "semantic_evidence_graph.json").to_dict()
        == result.graph.to_dict()
    )


def test_track_cooccurrence_retrieval_and_reader_exclude_outside_clips(tmp_path):
    graph, segments = _fixture(tmp_path)
    augmented = _augment(graph, segments).graph
    phase = next(
        n
        for n in augmented.nodes
        if n.node_type == "phase_hypothesis" and n.label == "Left Atrium Dissection"
    )
    cooccurs = [
        e for e in augmented.edges if e.relation == "co_occurs" and e.target == phase.id
    ]
    assert {e.metadata["clip_id"] for edge in cooccurs for e in edge.evidence} == {
        "c2",
        "c3",
        "c4",
    }
    for kind in ["onset", "offset"]:
        retrieval = retrieve_phase_boundary_instruments(
            augmented, phase.label, boundary_kind=kind, context_events=10
        )
        assert retrieval["phase_scoped_evidence"]
        assert all(
            768 <= e["start_seconds"] < e["end_seconds"] <= 960
            for e in retrieval["evidence"]
        )
        assert len(retrieval["instruments"]) == 3
    reader = build_phase_instrument_reader_input(
        augmented, phase.label, context_events=10, max_tracks=12, max_evidence_clips=12
    )
    assert {g["clip_id"] for g in reader["evidence_groups"]} == {"c2", "c3", "c4"}


def test_partial_clip_withholds_untimed_frames_and_touching_endpoint(tmp_path):
    graph, segments = _fixture(tmp_path)
    partial = {
        **segments[1],
        "start_seconds": 800,
        "end_seconds": 832,
        "supporting_clip_ids": ["c2"],
    }
    rows = project_sequence_phases_to_events(graph, [partial], source="test")
    assert len(rows[0]["phase_overlaps"]) == 1
    assert rows[1]["phase_overlaps"] == []
    evidence = rows[0]["phase_overlaps"][0]["evidence"]
    assert [(e["start_seconds"], e["end_seconds"]) for e in evidence] == [(800, 832)]
    assert evidence[0]["frame_paths"] == []
    augmented = _augment(graph, [partial]).graph
    with pytest.raises(ValueError):
        build_phase_instrument_reader_input(augmented, partial["label"])


def test_gaps_are_not_filled_and_unsupported_source_phase_is_retained(tmp_path):
    graph, segments = _fixture(tmp_path)
    nodes = [
        replace(n, evidence=[n.evidence[0], n.evidence[2]]) if n.id == "event:0" else n
        for n in graph.nodes
    ]
    graph = replace(graph, nodes=nodes)
    missing = {
        **segments[0],
        "start_seconds": 704,
        "end_seconds": 768,
        "supporting_clip_ids": ["c1"],
    }
    result = _augment(graph, [missing])
    phase = next(n for n in result.graph.nodes if n.node_type == "phase_hypothesis")
    assert phase.metadata["supporting_event_ids"] == []
    assert phase.metadata["source_segment"] == missing
    assert not any(e.relation == "grounded_by" for e in result.graph.edges)


def test_repeated_labels_keep_separate_source_segments(tmp_path):
    graph, segments = _fixture(tmp_path)
    for segment in segments:
        segment["label"] = "Suturing"
    result = _augment(graph, segments)
    assert result.report["phase_hypothesis_count"] == 3
    assert result.report["phase_boundary_count"] == 6


def test_stale_or_tampered_projection_cannot_attach_to_a_new_graph(tmp_path):
    graph, segments = _fixture(tmp_path)
    rows = project_sequence_phases_to_events(graph, segments, source="test")
    rows[0]["phase_overlaps"][0]["overlap_seconds"] += 1
    with pytest.raises(ValueError, match="Stale or invalid"):
        augment_with_semantic_hypotheses(graph, rows)
    rows = project_sequence_phases_to_events(graph, segments, source="test")
    with pytest.raises(ValueError, match="current graph event set"):
        augment_with_semantic_hypotheses(graph, rows[:1])


@pytest.mark.parametrize("end", [float("nan"), float("inf"), 640])
def test_invalid_source_interval_is_rejected(tmp_path, end):
    graph, segments = _fixture(tmp_path)
    segments[0]["end_seconds"] = end
    with pytest.raises(ValueError, match="Invalid sequence"):
        project_sequence_phases_to_events(graph, segments, source="test")


def test_phase_catalog_and_candidates_survive_jsonl_roundtrip(tmp_path):
    graph, segments = _fixture(tmp_path)
    rows = project_sequence_phases_to_events(graph, segments, source="test")
    rows = [json.loads(json.dumps(row)) for row in rows]
    phases = [
        n
        for n in augment_with_semantic_hypotheses(graph, rows).graph.nodes
        if n.node_type == "phase_hypothesis"
    ]
    assert [n.metadata["phase_candidates"] for n in phases] == [
        s["phase_candidates"] for s in segments
    ]


def test_projection_audit_rejects_widened_support_and_shifted_boundary(tmp_path):
    from experiments.reproject_sequence_phases import audit_projection

    base, segments = _fixture(tmp_path)
    graph = _augment(base, segments).graph
    assert audit_projection(base, graph, segments)[0]["passed"]
    boundary = next(n for n in graph.nodes if n.node_type == "phase_boundary")
    bad = replace(
        graph,
        nodes=[
            replace(n, metadata={**n.metadata, "timestamp_seconds": 641})
            if n.id == boundary.id
            else n
            for n in graph.nodes
        ],
    )
    assert not audit_projection(base, bad, segments)[0]["passed"]
    edge = next(
        e
        for e in graph.edges
        if e.source.startswith("phase_hypothesis") and e.relation == "derived_from"
    )
    bad = replace(
        graph,
        edges=[
            replace(e, evidence=[EvidenceInterval("case", 600, 1200)])
            if e is edge
            else e
            for e in graph.edges
        ],
    )
    report = audit_projection(base, bad, segments)[0]
    assert not report["passed"]
    assert any(v.startswith("outside_phase_evidence") for v in report["violations"])


def test_offline_cli_preserves_inputs_and_refuses_overwrite(tmp_path, monkeypatch):
    from experiments.reproject_sequence_phases import main

    graph, segments = _fixture(tmp_path)
    graph_path = tmp_path / "graph.json"
    phases_path = tmp_path / "phases.json"
    graph_path.write_text(json.dumps(graph.to_dict()))
    phases_path.write_text(json.dumps({"video_id": "case", "segments": segments}))
    inputs = graph_path.read_bytes(), phases_path.read_bytes()
    output = tmp_path / "projected"
    monkeypatch.setattr(
        "sys.argv",
        [
            "reproject_sequence_phases",
            "--graph",
            str(graph_path),
            "--sequence-phases",
            str(phases_path),
            "--output-dir",
            str(output),
        ],
    )
    main()
    assert json.loads((output / "projection_audit.json").read_text())["passed"]
    assert inputs == (graph_path.read_bytes(), phases_path.read_bytes())
    saved = (output / "semantic_evidence_graph.json").read_bytes()
    with pytest.raises(FileExistsError):
        main()
    assert (output / "semantic_evidence_graph.json").read_bytes() == saved
