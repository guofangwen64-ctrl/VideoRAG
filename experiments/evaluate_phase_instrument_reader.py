"""Answer phase-instrument QA from appearance tracks and grounded frames."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medhorizon_videorag.datasets import MedHorizonDataset
from medhorizon_videorag.graph_rag import (
    PHASE_INSTRUMENT_READER_VERSION,
    OpenAICompatibleGraphQA,
    build_open_activity_catalog,
    build_phase_instrument_reader_input,
    build_query_conditioned_phase_reader_input,
    extract_phase_name,
    load_evidence_graph,
    load_open_activity_segments,
    rank_open_activity_segments,
    select_activity_candidate_frame_groups,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--video-key", required=True)
    parser.add_argument("--qa-uids", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--open-activity-segments")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--context-events", type=int, default=1)
    parser.add_argument("--max-tracks", type=int, default=6)
    parser.add_argument("--max-evidence-clips", type=int, default=4)
    parser.add_argument("--frames-per-clip", type=int, default=8)
    parser.add_argument("--max-image-pixels", type=int, default=200704)
    parser.add_argument("--fallback-top-segments", type=int, default=3)
    parser.add_argument("--fallback-max-clips-per-segment", type=int, default=2)
    parser.add_argument("--fallback-frames-per-clip", type=int, default=4)
    parser.add_argument("--option-aware-tracks", action="store_true")
    parser.add_argument("--option-verifier", action="store_true")
    parser.add_argument(
        "--fallback-min-confidence",
        choices=("low", "medium", "high"),
        default="low",
    )
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    predictions_path = output / "predictions.jsonl"
    errors_path = output / "errors.jsonl"

    requested = [item.strip() for item in args.qa_uids.split(",") if item.strip()]
    dataset = MedHorizonDataset(args.annotations)
    by_uid = {
        str(item.uid): item
        for item in dataset.questions
        if item.video_key == args.video_key
    }
    missing = sorted(set(requested) - by_uid.keys())
    if missing:
        raise ValueError(f"Missing requested QA IDs: {missing}")
    selected = [by_uid[uid] for uid in requested]
    invalid_tasks = [
        str(item.uid)
        for item in selected
        if item.task_name != "Phase-Instrument Association"
    ]
    if invalid_tasks:
        raise ValueError(
            f"Requested QA IDs are not Phase-Instrument tasks: {invalid_tasks}"
        )

    graph = load_evidence_graph(args.graph)
    if graph.video_id != args.video_key:
        raise ValueError(
            f"Graph video {graph.video_id} does not match {args.video_key}"
        )
    activity_segments = (
        load_open_activity_segments(
            args.open_activity_segments, video_id=args.video_key
        )
        if args.open_activity_segments
        else []
    )
    activity_catalog = build_open_activity_catalog(activity_segments)
    reader = OpenAICompatibleGraphQA(
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        max_image_pixels=args.max_image_pixels,
    )
    metadata = {
        "experiment_version": PHASE_INSTRUMENT_READER_VERSION,
        "model": args.model,
        "base_url": args.base_url,
        "video_key": args.video_key,
        "qa_uids": requested,
        "context_events": args.context_events,
        "max_tracks": args.max_tracks,
        "max_evidence_clips": args.max_evidence_clips,
        "frames_per_clip": args.frames_per_clip,
        "max_image_pixels": args.max_image_pixels,
        "open_activity_segments": args.open_activity_segments,
        "open_activity_segment_count": len(activity_segments),
        "fallback_top_segments": args.fallback_top_segments,
        "fallback_retrieval_method": "deterministic_phase_action_cues_v1",
        "fallback_max_clips_per_segment": args.fallback_max_clips_per_segment,
        "fallback_frames_per_clip": args.fallback_frames_per_clip,
        "fallback_min_confidence": args.fallback_min_confidence,
        "query_conditioned_fallback_enabled": bool(activity_segments),
        "instrument_identity_annotations_used": False,
        "answers_used_for_retrieval_or_reader": False,
        "qa_options_used_only_by_reader": not args.option_aware_tracks,
        "qa_options_used_for_track_rerank": args.option_aware_tracks,
        "option_verifier_enabled": args.option_verifier,
        "candidate_aware_phase_graph": True,
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    rows = []
    for number, item in enumerate(selected, start=1):
        started = time.monotonic()
        phase = extract_phase_name(item.question)
        if not phase:
            raise ValueError(
                f"Cannot extract phase from QA {item.uid}: {item.question}"
            )
        formal_error: str | None = None
        phase_fallback: dict[str, Any] | None = None
        try:
            reader_input = build_phase_instrument_reader_input(
                graph,
                phase,
                qa_options=item.options if args.option_aware_tracks else None,
                context_events=args.context_events,
                max_tracks=args.max_tracks,
                max_evidence_clips=args.max_evidence_clips,
                frames_per_clip=args.frames_per_clip,
            )
        except ValueError as error:
            formal_error = str(error)
            if not activity_segments:
                row = _result_row(
                    item,
                    phase,
                    status="unresolved_graph",
                    elapsed=time.monotonic() - started,
                    unresolved_reason=formal_error,
                    phase_route="unresolved",
                )
                _append_jsonl(predictions_path, row)
                rows.append(row)
                print(
                    f"[{number}/{len(selected)}] {item.uid}: unresolved graph: {error}",
                    flush=True,
                )
                continue
            try:
                ranked_catalog = rank_open_activity_segments(
                    phase,
                    activity_catalog,
                    top_segments=args.fallback_top_segments,
                )
                by_segment_id = {
                    str(segment["segment_id"]): segment for segment in activity_segments
                }
                candidate_ids = [str(item["segment_id"]) for item in ranked_catalog]
                candidate_segments = [
                    {**by_segment_id[str(item["segment_id"])], **item}
                    for item in ranked_catalog
                ]
                verification_frames = select_activity_candidate_frame_groups(
                    graph,
                    candidate_segments,
                    max_clips_per_segment=args.fallback_max_clips_per_segment,
                    frames_per_clip=args.fallback_frames_per_clip,
                )
                verification = reader.verify_phase_activity_candidates(
                    phase, candidate_segments, verification_frames
                )
                selected_segment_id = verification["selected_segment_id"]
                phase_fallback = {
                    "formal_phase_error": formal_error,
                    "top_segment_ids": candidate_ids,
                    "retrieval_method": "deterministic_phase_action_cues_v1",
                    "retrieval_candidates": ranked_catalog,
                    "verification": verification,
                    "verification_evidence": verification_frames,
                    "qa_options_used": False,
                    "answer_used": False,
                }
                if selected_segment_id is None or not _confidence_meets(
                    verification["confidence"], args.fallback_min_confidence
                ):
                    reason = (
                        "Query-conditioned phase verification did not accept a "
                        f"candidate: {verification}"
                    )
                    row = _result_row(
                        item,
                        phase,
                        status="unresolved_graph",
                        elapsed=time.monotonic() - started,
                        unresolved_reason=reason,
                        phase_route="query_conditioned_activity_fallback",
                        phase_fallback=phase_fallback,
                    )
                    _append_jsonl(predictions_path, row)
                    rows.append(row)
                    print(
                        f"[{number}/{len(selected)}] {item.uid}: "
                        "fallback found no acceptable phase candidate",
                        flush=True,
                    )
                    continue
                reader_input = build_query_conditioned_phase_reader_input(
                    graph,
                    phase,
                    by_segment_id[selected_segment_id],
                    verification_confidence=verification["confidence"],
                    verification_rationale=verification["rationale"],
                    qa_options=item.options if args.option_aware_tracks else None,
                    context_events=args.context_events,
                    max_tracks=args.max_tracks,
                    max_evidence_clips=args.max_evidence_clips,
                    frames_per_clip=args.frames_per_clip,
                )
            except ValueError as fallback_error:
                reason = f"{formal_error}; fallback unresolved: {fallback_error}"
                row = _result_row(
                    item,
                    phase,
                    status="unresolved_graph",
                    elapsed=time.monotonic() - started,
                    unresolved_reason=reason,
                    phase_route="query_conditioned_activity_fallback",
                    phase_fallback=phase_fallback,
                )
                _append_jsonl(predictions_path, row)
                rows.append(row)
                print(f"[{number}/{len(selected)}] {item.uid}: {reason}", flush=True)
                continue
            except Exception as fallback_error:  # noqa: BLE001
                _append_jsonl(
                    errors_path,
                    {
                        "id": str(item.uid),
                        "stage": "phase_fallback",
                        "error_type": type(fallback_error).__name__,
                        "error": _safe_error(fallback_error),
                    },
                )
                row = _result_row(
                    item,
                    phase,
                    status="reader_failed",
                    elapsed=time.monotonic() - started,
                    unresolved_reason=_safe_error(fallback_error),
                    phase_route="query_conditioned_activity_fallback",
                    phase_fallback=phase_fallback,
                )
                _append_jsonl(predictions_path, row)
                rows.append(row)
                print(
                    f"[{number}/{len(selected)}] FAILED phase fallback {item.uid}: "
                    f"{type(fallback_error).__name__}: {_safe_error(fallback_error)}",
                    flush=True,
                )
                continue

        try:
            option_assessments = []
            if args.option_verifier:
                (
                    prediction,
                    rationale,
                    selected_track_ids,
                    option_assessments,
                ) = reader.answer_phase_instrument_with_option_verifier(
                    item.question, item.options, reader_input
                )
            else:
                prediction, rationale, selected_track_ids = (
                    reader.answer_phase_instrument(
                        item.question, item.options, reader_input
                    )
                )
            row = _result_row(
                item,
                phase,
                status="completed",
                elapsed=time.monotonic() - started,
                prediction=prediction,
                rationale=rationale,
                selected_track_ids=selected_track_ids,
                option_assessments=option_assessments,
                reader_input=reader_input,
                phase_route=reader_input["phase_route"],
                phase_fallback=phase_fallback,
            )
            _append_jsonl(predictions_path, row)
            rows.append(row)
            print(
                f"[{number}/{len(selected)}] {item.uid}: {prediction} "
                f"(reference {item.answer}) tracks={selected_track_ids}",
                flush=True,
            )
        except Exception as error:  # noqa: BLE001
            _append_jsonl(
                errors_path,
                {
                    "id": str(item.uid),
                    "error_type": type(error).__name__,
                    "error": _safe_error(error),
                },
            )
            row = _result_row(
                item,
                phase,
                status="reader_failed",
                elapsed=time.monotonic() - started,
                unresolved_reason=_safe_error(error),
                reader_input=reader_input,
                phase_route=reader_input["phase_route"],
                phase_fallback=phase_fallback,
            )
            _append_jsonl(predictions_path, row)
            rows.append(row)
            print(
                f"[{number}/{len(selected)}] FAILED {item.uid}: "
                f"{type(error).__name__}: {_safe_error(error)}",
                flush=True,
            )

    completed = [row for row in rows if row["status"] == "completed"]
    route_counts = Counter(str(row["phase_route"]) for row in rows)
    report: dict[str, Any] = {
        "requested": len(rows),
        "completed": len(completed),
        "unresolved_graph": sum(row["status"] == "unresolved_graph" for row in rows),
        "reader_failed": sum(row["status"] == "reader_failed" for row in rows),
        "correct": sum(bool(row["correct"]) for row in completed),
        "accuracy_all_requested": (
            sum(bool(row["correct"]) for row in completed) / len(rows) if rows else None
        ),
        "accuracy_resolved": (
            sum(bool(row["correct"]) for row in completed) / len(completed)
            if completed
            else None
        ),
        "prediction_counts": dict(Counter(str(row["prediction"]) for row in completed)),
        "phase_route_counts": dict(route_counts),
        "query_conditioned_completed": sum(
            row["status"] == "completed"
            and row["phase_route"] == "query_conditioned_activity_fallback"
            for row in rows
        ),
        "instrument_identity_annotations_used": False,
        "answers_used_for_retrieval_or_reader": False,
        "candidate_aware_phase_graph": True,
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _result_row(
    item: Any,
    phase: str,
    *,
    status: str,
    elapsed: float,
    prediction: str | None = None,
    rationale: str = "",
    selected_track_ids: list[str] | None = None,
    unresolved_reason: str | None = None,
    reader_input: dict[str, Any] | None = None,
    phase_route: str | None = None,
    phase_fallback: dict[str, Any] | None = None,
    option_assessments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "id": str(item.uid),
        "video_key": item.video_key,
        "task_name": item.task_name,
        "question": item.question,
        "options": item.options,
        "phase_query": phase,
        "status": status,
        "prediction": prediction,
        "reference": item.answer,
        "correct": prediction == item.answer if prediction is not None else None,
        "rationale": rationale,
        "selected_track_ids": selected_track_ids or [],
        "option_assessments": option_assessments or [],
        "reader_input": reader_input,
        "phase_route": phase_route,
        "phase_fallback": phase_fallback,
        "unresolved_reason": unresolved_reason,
        "elapsed_seconds": round(elapsed, 3),
    }


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _safe_error(error: Exception) -> str:
    message = re.sub(
        r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "<image-data>", str(error)
    )
    return message[:2000]


def _confidence_meets(value: str, threshold: str) -> bool:
    ranks = {"low": 0, "medium": 1, "high": 2}
    return ranks.get(str(value).lower(), -1) >= ranks[threshold]


if __name__ == "__main__":
    main()
