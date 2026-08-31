"""Opt-in CLI orchestration; keep inference and hidden-reference scoring separate."""

import hashlib
import json
from pathlib import Path

from medhorizon_videorag.datasets import MedHorizonDataset, recover_evidence

from .phase_candidate_flow import (
    PHASE_CANDIDATE_FLOW_VERSION,
    full_event_catalog,
    load_phase_candidates,
    render_event_catalog,
    run_candidate_question,
)
from .phase_candidate_metrics import question_metrics, summarize_metrics
from .qa_experiment import OpenAICompatibleGraphQA
from .retrieval import load_evidence_graph
from .semantic_layer import extract_phase_name


def run_candidate_cli(args):
    if (
        min(
            args.candidate_top_k,
            args.max_tracks,
            args.max_evidence_clips,
            args.frames_per_clip,
        )
        < 1
        or args.context_events < 0
    ):
        raise ValueError(
            "Candidate and frame limits must be positive; context must be non-negative"
        )
    if not 0 < args.evidence_recall_threshold <= 1:
        raise ValueError("Evidence recall threshold must be in (0, 1]")
    if args.open_activity_segments:
        raise ValueError(
            "Candidate-file mode does not silently mix the legacy activity fallback"
        )
    graph = load_evidence_graph(args.graph)
    if graph.video_id != args.video_key:
        raise ValueError("Graph video mismatch")
    candidates = load_phase_candidates(args.phase_candidates, graph)
    requested = [value.strip() for value in args.qa_uids.split(",") if value.strip()]
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("QA IDs must be non-empty and unique")
    dataset = MedHorizonDataset(args.annotations)
    by_id = {str(q.uid): q for q in dataset.questions if q.video_key == graph.video_id}
    selected = [by_id[uid] for uid in requested]
    if any(
        q.task_name != "Phase-Instrument Association"
        or not extract_phase_name(q.question)
        for q in selected
    ):
        raise ValueError("Candidate-file mode requires phase-instrument questions")
    gold = {}
    if args.gold_evidence:
        for line in Path(args.gold_evidence).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                row.get("reviewed") is not True
                or row["video_key"] != graph.video_id
                or str(row["id"]) in gold
            ):
                raise ValueError(
                    "Gold windows need explicit review, matching video and unique QA IDs"
                )
            gold[str(row["id"])] = row
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    reader = (
        None
        if args.candidate_dry_run
        else OpenAICompatibleGraphQA(
            model=args.model,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            max_image_pixels=args.max_image_pixels,
        )
    )
    metadata = {
        "version": PHASE_CANDIDATE_FLOW_VERSION,
        "candidate_file_enabled": True,
        "dry_run": args.candidate_dry_run,
        "video_key": graph.video_id,
        "requested_ids": requested,
        "candidate_count": len(candidates),
        "candidate_source_status": "available"
        if candidates
        else "empty_candidate_catalog",
        "candidate_top_k": args.candidate_top_k,
        "verification_min_confidence": args.candidate_min_confidence,
        "verification_max_clips_per_candidate": 2,
        "verification_frames_per_clip": 4,
        "context_events": args.context_events,
        "max_tracks": args.max_tracks,
        "max_evidence_clips": args.max_evidence_clips,
        "reader_frames_per_clip": args.frames_per_clip,
        "model": None if args.candidate_dry_run else args.model,
        "candidate_file_sha256": hashlib.sha256(
            Path(args.phase_candidates).read_bytes()
        ).hexdigest(),
        "graph_sha256": hashlib.sha256(Path(args.graph).read_bytes()).hexdigest(),
        "answers_and_gt_times_used_for_inference": False,
        "event_display_mode": "full",
        "option_verifier": args.option_verifier,
        "option_aware_tracks": args.option_aware_tracks,
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (output / "event_catalog.jsonl").open("w") as handle:
        for event in full_event_catalog(graph):
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    (output / "event_details.md").write_text(render_event_catalog(graph))
    traces = []
    with (output / "candidate_traces.jsonl").open("w") as handle:
        for item in selected:
            trace = run_candidate_question(
                graph,
                candidates,
                question_id=item.uid,
                question=item.question,
                options=item.options,
                phase=extract_phase_name(item.question),
                reader=reader,
                top_k=args.candidate_top_k,
                option_verifier=args.option_verifier,
                option_aware=args.option_aware_tracks,
                context_events=args.context_events,
                max_tracks=args.max_tracks,
                max_evidence_clips=args.max_evidence_clips,
                frames_per_clip=args.frames_per_clip,
                verification_min_confidence=args.candidate_min_confidence,
            )
            traces.append(trace)
            handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
            handle.flush()
    # No hidden answer, temporal annotation, or recovered weak anchor reaches
    # run_candidate_question / the verifier / the Reader above.
    anchors = {
        str(e.qa_uid): e.to_dict()
        for e in recover_evidence(dataset)
        if e.video_key == graph.video_id and e.method == "phase_anchor"
    }
    metrics = [
        question_metrics(
            trace,
            reference_answer=by_id[trace["id"]].answer,
            weak_phase_anchors=anchors.get(trace["id"], {}).get("windows", []),
            gold_phase_windows=gold.get(trace["id"], {}).get("phase_windows", []),
            gold_evidence_windows=gold.get(trace["id"], {}).get("evidence_windows", []),
            evidence_recall_threshold=args.evidence_recall_threshold,
        )
        for trace in traces
    ]
    (output / "question_metrics.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in metrics)
    )
    report = summarize_metrics(metrics)
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
