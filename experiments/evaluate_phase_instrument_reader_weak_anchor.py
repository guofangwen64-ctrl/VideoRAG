"""Diagnostic phase-instrument QA with weak Action Recognition phase anchors."""

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

from medhorizon_videorag.datasets import (  # noqa: E402
    MedHorizonDataset,
    parse_temporal_query,
)
from medhorizon_videorag.graph_rag import (  # noqa: E402
    OpenAICompatibleGraphQA,
    build_query_conditioned_phase_reader_input,
    extract_phase_name,
    load_evidence_graph,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--video-key", required=True)
    parser.add_argument("--qa-uids", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="qwen3-vl-235b-a22b-instruct")
    parser.add_argument("--base-url", default="https://api.agicto.cn/v1")
    parser.add_argument("--api-key-env", default="AGICTO_API_KEY")
    parser.add_argument("--context-events", type=int, default=1)
    parser.add_argument("--max-tracks", type=int, default=6)
    parser.add_argument("--max-evidence-clips", type=int, default=4)
    parser.add_argument("--frames-per-clip", type=int, default=8)
    parser.add_argument("--max-image-pixels", type=int, default=200704)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    predictions_path = output / "predictions.jsonl"
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
    invalid = [
        str(item.uid)
        for item in selected
        if item.task_name != "Phase-Instrument Association"
    ]
    if invalid:
        raise ValueError(f"Requested QA IDs are not Phase-Instrument tasks: {invalid}")

    graph = load_evidence_graph(args.graph)
    if graph.video_id != args.video_key:
        raise ValueError(f"Graph video {graph.video_id} does not match {args.video_key}")
    anchors = _phase_anchors(dataset, args.video_key)
    clip_segments = _clip_segments(graph)
    reader = OpenAICompatibleGraphQA(
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        max_image_pixels=args.max_image_pixels,
    )
    metadata = {
        "experiment": "phase_instrument_reader_weak_action_anchor",
        "diagnostic_only": True,
        "official_baseline_score": False,
        "phase_anchor_source": "Action Recognition QA temporal ranges and answers",
        "phase_instrument_answers_used": False,
        "model": args.model,
        "base_url": args.base_url,
        "video_key": args.video_key,
        "qa_uids": requested,
        "context_events": args.context_events,
        "max_tracks": args.max_tracks,
        "max_evidence_clips": args.max_evidence_clips,
        "frames_per_clip": args.frames_per_clip,
        "max_image_pixels": args.max_image_pixels,
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    rows = []
    for number, item in enumerate(selected, start=1):
        started = time.monotonic()
        phase = extract_phase_name(item.question)
        if not phase:
            raise ValueError(f"Cannot extract phase from QA {item.uid}: {item.question}")
        anchor = _best_anchor(phase, anchors)
        if anchor is None:
            row = _result_row(
                item,
                phase,
                status="unresolved_anchor",
                elapsed=time.monotonic() - started,
                unresolved_reason="No same-video Action Recognition weak anchor matched this phase.",
            )
            _append_jsonl(predictions_path, row)
            rows.append(row)
            print(f"[{number}/{len(selected)}] {item.uid}: no weak anchor", flush=True)
            continue
        activity_segment = _anchor_activity_segment(
            args.video_key, phase, anchor, clip_segments
        )
        try:
            reader_input = build_query_conditioned_phase_reader_input(
                graph,
                phase,
                activity_segment,
                verification_confidence="high",
                verification_rationale=(
                    "Diagnostic weak anchor from Action Recognition temporal QA; "
                    "not an observation-derived persistent phase."
                ),
                context_events=args.context_events,
                max_tracks=args.max_tracks,
                max_evidence_clips=args.max_evidence_clips,
                frames_per_clip=args.frames_per_clip,
            )
            reader_input["phase_route"] = "weak_action_recognition_anchor"
            reader_input["query_conditioned_phase_candidate"] = {
                **reader_input["query_conditioned_phase_candidate"],
                "anchor_source_qa_uid": anchor["source_qa_uid"],
                "anchor_phase_label": anchor["phase_label"],
                "anchor_start_seconds": anchor["start_seconds"],
                "anchor_end_seconds": anchor["end_seconds"],
            }
            prediction, rationale, selected_track_ids = reader.answer_phase_instrument(
                item.question, item.options, reader_input
            )
            row = _result_row(
                item,
                phase,
                status="completed",
                elapsed=time.monotonic() - started,
                prediction=prediction,
                rationale=rationale,
                selected_track_ids=selected_track_ids,
                reader_input=reader_input,
                anchor=anchor,
            )
            print(
                f"[{number}/{len(selected)}] {item.uid}: {prediction} "
                f"(reference {item.answer}) tracks={selected_track_ids}",
                flush=True,
            )
        except Exception as error:  # noqa: BLE001
            row = _result_row(
                item,
                phase,
                status="reader_failed",
                elapsed=time.monotonic() - started,
                unresolved_reason=_safe_error(error),
                anchor=anchor,
            )
            print(
                f"[{number}/{len(selected)}] FAILED {item.uid}: "
                f"{type(error).__name__}: {_safe_error(error)}",
                flush=True,
            )
        _append_jsonl(predictions_path, row)
        rows.append(row)

    completed = [row for row in rows if row["status"] == "completed"]
    report = {
        "requested": len(rows),
        "completed": len(completed),
        "unresolved_anchor": sum(row["status"] == "unresolved_anchor" for row in rows),
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
        "diagnostic_only": True,
        "official_baseline_score": False,
        "phase_anchor_source": "Action Recognition QA temporal ranges and answers",
        "phase_instrument_answers_used": False,
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _phase_anchors(
    dataset: MedHorizonDataset, video_key: str
) -> dict[str, list[dict[str, Any]]]:
    anchors: dict[str, list[dict[str, Any]]] = {}
    for qa in dataset.questions:
        if qa.video_key != video_key or qa.task_name != "Action Recognition":
            continue
        temporal = parse_temporal_query(qa.question)
        if temporal is None or temporal.kind != "range":
            continue
        phase = _answer_text(qa.answer, qa.options)
        if not phase:
            continue
        anchors.setdefault(_canonical(phase), []).append(
            {
                "source_qa_uid": str(qa.uid),
                "phase_label": phase,
                "start_seconds": temporal.start_seconds,
                "end_seconds": temporal.end_seconds,
            }
        )
    return anchors


def _clip_segments(graph: Any) -> list[dict[str, Any]]:
    rows = []
    for node in graph.nodes:
        if node.node_type != "segment" or not node.evidence:
            continue
        clip_id = str(node.metadata.get("clip_id") or node.id.removeprefix("clip:"))
        rows.append(
            {
                "clip_id": clip_id,
                "start_seconds": float(node.evidence[0].start_seconds),
                "end_seconds": float(node.evidence[0].end_seconds),
            }
        )
    rows.sort(key=lambda item: (float(item["start_seconds"]), str(item["clip_id"])))
    return rows


def _best_anchor(
    phase: str, anchors: dict[str, list[dict[str, Any]]]
) -> dict[str, Any] | None:
    key = _canonical(phase)
    if key in anchors:
        return anchors[key][0]
    matches = [
        anchor
        for anchor_key, rows in anchors.items()
        if len(anchor_key) >= 4 and (anchor_key in key or key in anchor_key)
        for anchor in rows
    ]
    return matches[0] if matches else None


def _anchor_activity_segment(
    video_key: str,
    phase: str,
    anchor: dict[str, Any],
    clip_segments: list[dict[str, Any]],
) -> dict[str, Any]:
    clips = [
        clip
        for clip in clip_segments
        if float(clip["start_seconds"]) < float(anchor["end_seconds"])
        and float(anchor["start_seconds"]) < float(clip["end_seconds"])
    ]
    if not clips:
        raise ValueError(f"Weak anchor for {phase!r} overlaps no graph clips")
    return {
        "segment_id": f"weak_action_anchor:{video_key}:{anchor['source_qa_uid']}",
        "activity_label": f"weak anchor for {phase}",
        "observed_pattern": (
            "Diagnostic phase window inherited from Action Recognition QA time range."
        ),
        "start_seconds": float(anchor["start_seconds"]),
        "end_seconds": float(anchor["end_seconds"]),
        "supporting_clip_ids": [str(clip["clip_id"]) for clip in clips],
        "basis_clip_ids": [str(clip["clip_id"]) for clip in clips[:5]],
        "fact_status": "diagnostic_weak_anchor",
    }


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
    anchor: dict[str, Any] | None = None,
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
        "reader_input": reader_input,
        "phase_route": "weak_action_recognition_anchor",
        "weak_anchor": anchor,
        "unresolved_reason": unresolved_reason,
        "elapsed_seconds": round(elapsed, 3),
    }


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _answer_text(label: str | None, options: list[str]) -> str | None:
    if not label:
        return None
    wanted = str(label).strip().upper()
    for index, option in enumerate(options):
        match = re.match(r"^\s*([A-Za-z0-9]+)[.):：]", option)
        parsed = match.group(1).upper() if match else chr(ord("A") + index)
        if parsed == wanted:
            return re.sub(r"^\s*[A-Za-z0-9]+[.):：]\s*", "", option).strip()
    return None


def _canonical(value: str) -> str:
    return "".join(re.findall(r"[a-z0-9]+", value.lower()))


def _safe_error(error: Exception) -> str:
    message = re.sub(
        r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "<image-data>", str(error)
    )
    return message[:2000]


if __name__ == "__main__":
    main()
