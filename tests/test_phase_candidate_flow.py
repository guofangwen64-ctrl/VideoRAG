import json
from copy import deepcopy
from types import MethodType, SimpleNamespace

import pytest

from medhorizon_videorag.graph_rag import (
    EvidenceInterval,
    GraphNode,
    OpenAICompatibleGraphQA,
    VideoEvidenceGraph,
    augment_with_semantic_hypotheses,
    project_sequence_phases_to_events,
)
from medhorizon_videorag.graph_rag.phase_candidate_flow import (
    candidate_reader_packet,
    full_event_catalog,
    load_phase_candidates,
    run_candidate_question,
)
from medhorizon_videorag.graph_rag.phase_candidate_metrics import (
    question_metrics,
    summarize_metrics,
)


def _case(tmp_path):
    nodes, intervals = [], []
    for i in range(3):
        path = tmp_path / f"{i}.jpg"
        path.write_bytes(b"frame")
        e = EvidenceInterval(
            "case", i * 64, (i + 1) * 64, [str(path)], metadata={"clip_id": f"c{i}"}
        )
        intervals.append(e)
        nodes.append(
            GraphNode(
                f"clip:c{i}",
                "case",
                "segment",
                f"clip {i}",
                [e],
                metadata={
                    "clip_id": f"c{i}",
                    "observation": {
                        "observed_facts": {
                            "visual_evidence": [f"complete distinctive observation {i}"]
                        }
                    },
                },
            )
        )
        nodes.append(
            GraphNode(
                f"mention:{i}",
                "case",
                "entity_mention",
                f"tool {i}",
                [e],
                metadata={
                    "clip_id": f"c{i}",
                    "source_field": "visible_instruments",
                    "category": "instrument",
                    "canonical": "generic_instrument",
                    "attributes": {"color": [f"color{i}"]},
                },
            )
        )
    for i, group in enumerate([intervals[:2], intervals[2:]]):
        nodes.append(
            GraphNode(
                f"event:{i}",
                "case",
                "temporal_event",
                "truncated label",
                group,
                metadata={
                    "supporting_clip_ids": [e.metadata["clip_id"] for e in group]
                },
            )
        )
    segments, raw = [], []
    for i, decision in enumerate(["tentative", "contradicted", "insufficient"]):
        candidate = {
            "label": "Target Phase",
            "rank": 2 if i == 0 else 1,
            "score": [0.6, 0.95, 0.7][i],
            "decision": decision,
            "accepted": True,
            "confidence": "medium",
            "confidence_score": 0.65,
            "positive_cues": ["source cue"],
            "negative_cues": ["conflict"] if i == 1 else [],
            "missing_evidence": [],
        }
        segment = {
            "segment_id": f"s{i}",
            "label": "Other Phase",
            "start_seconds": i * 64,
            "end_seconds": (i + 1) * 64,
            "supporting_clip_ids": [f"c{i}"],
            "confidence": "medium",
            "phase_candidates": [candidate],
        }
        segments.append(segment)
        raw.append(
            {
                "video_id": "case",
                "sequence_phase_segment_id": f"s{i}",
                "start_seconds": i * 64,
                "end_seconds": (i + 1) * 64,
                **candidate,
            }
        )
    base = VideoEvidenceGraph("case", nodes, [])
    graph = augment_with_semantic_hypotheses(
        base,
        project_sequence_phases_to_events(base, segments, source="test"),
        instrument_track_source="appearance_mentions",
    ).graph
    path = tmp_path / "candidates.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in raw))
    return graph, path


class FakeReader:
    def __init__(self, selected=None):
        self.selected = selected
        self.calls = []

    def verify_phase_candidates(self, request):
        self.calls.append(("verify", deepcopy(request)))
        return {
            "selected_candidate_id": self.selected
            or request["candidates"][1]["candidate_id"],
            "assessments": [
                {
                    "candidate_id": c["candidate_id"],
                    "decision": "supported",
                    "confidence": "high",
                    "positive_evidence": ["visible cue"],
                    "counter_evidence": [],
                    "missing_evidence": [],
                }
                for c in request["candidates"]
            ],
        }

    def answer_phase_instrument(self, question, options, packet):
        self.calls.append(("reader", deepcopy(packet)))
        return "B", "visible", [packet["candidate_tracks"][0]["track_id"]]

    def answer_phase_instrument_with_option_verifier(self, question, options, packet):
        self.calls.append(("options", deepcopy(packet)))
        return (
            "B",
            "visible",
            [packet["candidate_tracks"][0]["track_id"]],
            [{"option_label": "B", "support": ["visible"], "contradiction": []}],
        )


def _run(graph, candidates, **kwargs):
    return run_candidate_question(
        graph,
        candidates,
        question_id="q1",
        question="At phase onset for Target Phase, which tool?",
        options=["A. one", "B. two"],
        phase="Target Phase",
        **kwargs,
    )


def test_ids_survive_file_order_and_rank_changes(tmp_path):
    graph, path = _case(tmp_path)
    before = load_phase_candidates(path, graph)
    rows = [json.loads(line) for line in path.read_text().splitlines()][::-1]
    for row in rows:
        row["rank"] = 9
        row["score"] = 0.2
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    after = load_phase_candidates(path, graph)
    assert {r["candidate_id"] for r in before} == {r["candidate_id"] for r in after}
    assert before[1]["evidence_role"] == "counter_evidence"
    assert (
        before[2]["evidence_role"] == "uncertain"
    )  # accepted does not make insufficient positive.


@pytest.mark.parametrize("mutation", ["duplicate", "interval", "video", "id"])
def test_loader_rejects_bad_identity_or_provenance(tmp_path, mutation):
    graph, path = _case(tmp_path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if mutation == "duplicate":
        rows.append(rows[0])
    if mutation == "interval":
        rows[0]["end_seconds"] = 63
    if mutation == "video":
        rows[0]["video_id"] = "another"
    if mutation == "id":
        rows[0]["candidate_id"] = "forged"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError):
        load_phase_candidates(path, graph)


@pytest.mark.parametrize("option_verifier", [False, True])
def test_candidate_id_reaches_verifier_reader_and_all_stage_logs(
    tmp_path, option_verifier
):
    graph, path = _case(tmp_path)
    candidates = load_phase_candidates(path, graph)
    reader = FakeReader(selected=candidates[0]["candidate_id"])
    before = graph.to_dict()
    trace = _run(graph, candidates, reader=reader, option_verifier=option_verifier)
    assert trace["status"] == "completed"
    assert trace["retrieval_top1_candidate_id"] == candidates[2]["candidate_id"]
    assert (
        trace["verified_candidate_id"]
        == trace["reader_candidate_id"]
        == candidates[0]["candidate_id"]
    )
    assert (
        reader.calls[-1][1]["phase_candidate"]["candidate_id"]
        == candidates[0]["candidate_id"]
    )
    assert (
        reader.calls[-1][1]["phase_counter_evidence"][0]["candidate_id"]
        == candidates[1]["candidate_id"]
    )
    assert {g["clip_id"] for g in trace["reader_input"]["evidence_groups"]} == {"c0"}
    assert all(
        set(c["lifecycle"]) == {"load", "graph", "retrieval", "verification", "reader"}
        for c in trace["candidates"]
    )
    assert graph.to_dict() == before


def test_counter_cannot_be_selected_even_if_verifier_claims_support(tmp_path):
    graph, path = _case(tmp_path)
    candidates = load_phase_candidates(path, graph)
    reader = FakeReader(selected=candidates[1]["candidate_id"])
    trace = _run(graph, candidates, reader=reader)
    assert trace["status"] == "unresolved"
    assert trace["reader_candidate_id"] is None
    assert len(reader.calls) == 1
    with pytest.raises(ValueError, match="Counter-evidence"):
        candidate_reader_packet(graph, candidates[1])


def test_dry_run_prepares_but_does_not_claim_verification_or_reader(tmp_path):
    graph, path = _case(tmp_path)
    candidates = load_phase_candidates(path, graph)
    trace = _run(graph, candidates, top_k=1)
    assert trace["status"] == "prepared"
    assert trace["verified_candidate_id"] is None and trace["prediction"] is None
    assert trace["candidates"][0]["lifecycle"]["retrieval"]["reason"] == "top_k_limit"
    metrics = question_metrics(trace, reference_answer="B")
    assert metrics["metrics"]["answer_correct_all_requested"] is None
    assert metrics["metrics"]["evidence_interval_correct"] is None


def test_metrics_separate_counter_name_time_rank_graph_and_answer(tmp_path):
    graph, path = _case(tmp_path)
    candidates = load_phase_candidates(path, graph)
    trace = _run(
        graph, candidates, reader=FakeReader(selected=candidates[0]["candidate_id"])
    )
    metrics = question_metrics(
        trace,
        reference_answer="B",
        weak_phase_anchors=[{"start_seconds": 32, "end_seconds": 96}],
        gold_evidence_windows=[{"start_seconds": 0, "end_seconds": 64}],
    )
    m = metrics["metrics"]
    assert (
        m["weak_phase_anchor_recall"] == 0.5
    )  # contradicted interval is not positive recall.
    assert m["phase_time_recall_gold"] is None
    assert m["graph_primary_phase_name_survival"] is False
    assert m["graph_candidate_record_survival"] is True
    assert m["answer_and_evidence_interval_correct"] is True
    missing = question_metrics(trace, reference_answer="A")
    summary = summarize_metrics([metrics, missing])
    assert summary["metrics"]["weak_phase_anchor_recall"]["denominator"] == 1
    assert summary["metrics"]["evidence_interval_correct"]["unavailable"] == 1
    assert missing["metrics"]["answer_correct_all_requested"] is False


def test_all_counter_counts_as_name_only_and_full_catalog_is_not_label(tmp_path):
    graph, path = _case(tmp_path)
    candidates = load_phase_candidates(path, graph)[1:2]
    trace = _run(graph, candidates)
    m = question_metrics(trace)["metrics"]
    assert m["candidate_name_coverage"] is True
    assert m["candidate_non_counter_coverage"] is False
    assert m["retrieval_top1_name_match"] is False
    catalog = list(full_event_catalog(graph))
    assert "complete distinctive observation 1" in json.dumps(catalog[0])
    assert len(catalog[0]["entity_mentions"]) == 2


def test_verifier_and_reader_prompts_preserve_candidate_polarity(monkeypatch):
    reader = object.__new__(OpenAICompatibleGraphQA)
    reader.max_image_pixels = 1
    captured = {}

    def fake(self, content, *, max_tokens):
        captured["prompt"] = content[0]["text"]
        return {
            "selected_candidate_id": "c",
            "assessments": [
                {
                    "candidate_id": "c",
                    "decision": "supported",
                    "confidence": "high",
                    "positive_evidence": ["cue"],
                    "counter_evidence": [],
                    "missing_evidence": [],
                }
            ],
        }

    reader._vision_json = MethodType(fake, reader)
    result = reader.verify_phase_candidates(
        {
            "target_phase": "Target",
            "candidates": [
                {
                    "candidate_id": "c",
                    "decision": "tentative",
                    "evidence_role": "positive",
                }
            ],
            "evidence_groups": [],
        }
    )
    assert result["selected_candidate_id"] == "c"
    assert (
        "counter_evidence" in captured["prompt"]
        and "MUST NOT be selected" in captured["prompt"]
    )


def test_legacy_cli_does_not_dispatch_candidate_mode(tmp_path, monkeypatch):
    import experiments.evaluate_phase_instrument_reader as cli
    import medhorizon_videorag.graph_rag.phase_candidate_experiment as candidate_cli

    def forbidden(args):
        raise AssertionError("new route called by legacy CLI")

    monkeypatch.setattr(candidate_cli, "run_candidate_cli", forbidden)
    monkeypatch.setattr(
        cli, "MedHorizonDataset", lambda _: SimpleNamespace(questions=[])
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "reader",
            "--annotations",
            "unused",
            "--graph",
            "unused",
            "--video-key",
            "case",
            "--qa-uids",
            "missing",
            "--output-dir",
            str(tmp_path / "legacy"),
        ],
    )
    with pytest.raises(ValueError, match="Missing requested QA"):
        cli.main()


def test_failed_verifier_logs_stage_without_provider_secret(tmp_path):
    graph, path = _case(tmp_path)
    reader = FakeReader()

    def fail(request):
        raise RuntimeError("provider exception contains SECRET_PLACEHOLDER")

    reader.verify_phase_candidates = fail
    trace = _run(graph, load_phase_candidates(path, graph), reader=reader)
    assert trace["failure_stage"] == "verification"
    assert trace["status"] == "failed"
    assert "SECRET_PLACEHOLDER" not in json.dumps(trace)
    assert any(
        c["lifecycle"]["verification"]["status"] == "failed"
        for c in trace["candidates"]
    )


def test_actual_reader_prompt_contains_selected_and_counter_candidate_ids():
    reader = object.__new__(OpenAICompatibleGraphQA)
    reader.max_image_pixels = 1
    captured = {}

    def fake(self, content, *, max_tokens):
        captured["text"] = content[0]["text"]
        return {"choice": "B", "selected_track_ids": ["t"], "rationale": "visible"}

    reader._vision_json = MethodType(fake, reader)
    packet = {
        "phase_label": "Target",
        "phase_candidate": {
            "candidate_id": "chosen-id",
            "decision": "tentative",
            "start_seconds": 64,
            "end_seconds": 128,
        },
        "phase_counter_evidence": [
            {"candidate_id": "counter-id", "decision": "contradicted"}
        ],
        "candidate_tracks": [
            {
                "track_id": "t",
                "graph_rank": 1,
                "label": "tool",
                "appearance_family": "generic_instrument",
                "appearance_signature": {},
                "surface_forms": ["tool"],
                "action_roles": [],
                "reader_clip_ids": ["c"],
            }
        ],
        "evidence_groups": [],
    }
    assert (
        reader.answer_phase_instrument("Which?", ["A. one", "B. two"], packet)[0] == "B"
    )
    assert "chosen-id" in captured["text"] and "counter-id" in captured["text"]
    assert "never use these as positive phase support" in captured["text"]


def test_candidate_cli_dry_run_is_offline_and_never_overwrites(tmp_path, monkeypatch):
    from medhorizon_videorag.graph_rag import phase_candidate_experiment as cli

    graph, candidates = _case(tmp_path)
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(graph.to_dict()))
    question = SimpleNamespace(
        uid="q1",
        video_key="case",
        task_name="Phase-Instrument Association",
        question="At phase onset for Target Phase, which tool?",
        options=["A. one", "B. two"],
        answer="B",
    )
    monkeypatch.setattr(
        cli, "MedHorizonDataset", lambda _: SimpleNamespace(questions=[question])
    )
    monkeypatch.setattr(cli, "recover_evidence", lambda _: [])

    def forbidden(**kwargs):
        raise AssertionError("Model instantiated during dry run")

    monkeypatch.setattr(cli, "OpenAICompatibleGraphQA", forbidden)
    args = SimpleNamespace(
        open_activity_segments=None,
        graph=str(graph_path),
        video_key="case",
        phase_candidates=str(candidates),
        qa_uids="q1",
        annotations="unused",
        gold_evidence=None,
        output_dir=str(tmp_path / "flow"),
        candidate_dry_run=True,
        candidate_top_k=3,
        candidate_min_confidence="medium",
        option_verifier=False,
        option_aware_tracks=False,
        context_events=1,
        max_tracks=6,
        max_evidence_clips=4,
        frames_per_clip=4,
        evidence_recall_threshold=0.5,
    )
    cli.run_candidate_cli(args)
    output = tmp_path / "flow"
    report = json.loads((output / "report.json").read_text())
    assert report["metrics"]["answer_correct_all_requested"]["denominator"] == 0
    assert (
        "complete distinctive observation 2"
        in (output / "event_details.md").read_text()
    )
    saved_trace = json.loads((output / "candidate_traces.jsonl").read_text())
    assert any(c.get("prepared_reader_input") for c in saved_trace["candidates"])
    original = (output / "candidate_traces.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        cli.run_candidate_cli(args)
    assert (output / "candidate_traces.jsonl").read_bytes() == original
