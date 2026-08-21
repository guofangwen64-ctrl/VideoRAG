"""Run Qwen2.5-VL QA with stripped time text or Qwen-reranked graph events."""

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

from medhorizon_videorag.datasets import (
    MedHorizonDataset,
    parse_temporal_query,
)
from medhorizon_videorag.graph_rag import load_evidence_graph
from medhorizon_videorag.graph_rag.qa_experiment import (
    GRAPH_QA_EXPERIMENT_VERSION,
    OpenAICompatibleGraphQA,
    build_event_observation_catalog,
    select_event_frame_groups,
    strip_explicit_time_range,
)
from medhorizon_videorag.ingestion import FineFrameExtractor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--video-key", required=True)
    parser.add_argument("--qa-uids", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--top-events", type=int, default=2)
    parser.add_argument("--frames-per-event", type=int, default=8)
    parser.add_argument("--direct-window-frames", type=int, default=16)
    parser.add_argument("--max-image-pixels", type=int, default=200704)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = output_dir / "predictions.jsonl"
    errors_path = output_dir / "errors.jsonl"
    metadata_path = output_dir / "run_metadata.json"
    report_path = output_dir / "report.json"

    requested_uids = [item.strip() for item in args.qa_uids.split(",") if item.strip()]
    dataset = MedHorizonDataset(args.annotations)
    selected = [
        item
        for item in dataset.questions
        if item.video_key == args.video_key and str(item.uid) in requested_uids
    ]
    selected_by_uid = {str(item.uid): item for item in selected}
    missing = [item for item in requested_uids if item not in selected_by_uid]
    if missing:
        raise ValueError(
            f"Requested QA IDs are missing for video {args.video_key}: {missing}"
        )
    selected = [selected_by_uid[item] for item in requested_uids]

    graph = load_evidence_graph(args.graph)
    if graph.video_id != args.video_key:
        raise ValueError(
            f"Graph video {graph.video_id} does not match {args.video_key}"
        )
    catalog = build_event_observation_catalog(graph)
    client = OpenAICompatibleGraphQA(
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        max_image_pixels=args.max_image_pixels,
    )
    frame_extractor = FineFrameExtractor(
        output_dir / "direct_window_frames", args.direct_window_frames
    )
    metadata = {
        "experiment_version": GRAPH_QA_EXPERIMENT_VERSION,
        "model": args.model,
        "base_url": args.base_url,
        "video_key": args.video_key,
        "qa_uids": requested_uids,
        "question_time_text_removed": True,
        "direct_windows_used_only_for_visual_evidence": True,
        "top_events": args.top_events,
        "frames_per_event": args.frames_per_event,
        "direct_window_frames": args.direct_window_frames,
        "event_catalog_count": len(catalog),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    rows = []
    for number, item in enumerate(selected, start=1):
        started = time.monotonic()
        try:
            original_temporal = parse_temporal_query(item.question)
            question = strip_explicit_time_range(item.question)
            if parse_temporal_query(question) is not None:
                raise ValueError(
                    f"Model-facing question still contains time: {question}"
                )
            retrieved_event_ids: list[str] = []
            rerank_rationale = ""
            if original_temporal and original_temporal.kind == "range":
                video_path = Path(args.video_root) / item.video_path
                frames = frame_extractor.extract_window(
                    item.video_key,
                    str(video_path),
                    original_temporal.start_seconds,
                    original_temporal.end_seconds,
                )
                evidence_groups = [
                    {
                        "event_id": None,
                        "clip_id": f"direct_window_{item.uid}",
                        "start_seconds": original_temporal.start_seconds,
                        "end_seconds": original_temporal.end_seconds,
                        "reader_frame_paths": frames,
                        "selection": "direct_range_removed_from_question_text",
                    }
                ]
                route = "direct_window_time_stripped"
            else:
                retrieved_event_ids, rerank_rationale = client.rerank_events(
                    question, item.options, catalog, top_events=args.top_events
                )
                onset_query = bool(
                    re.search(
                        r"\b(?:onset|begins?|transition|opening moments?)\b",
                        question,
                        re.IGNORECASE,
                    )
                )
                evidence_groups = select_event_frame_groups(
                    graph,
                    retrieved_event_ids,
                    frames_per_event=args.frames_per_event,
                    prefer_onset=onset_query,
                )
                route = "qwen25_event_rerank"
            prediction, rationale = client.answer(
                question, item.options, evidence_groups
            )
            row = {
                "id": str(item.uid),
                "video_key": item.video_key,
                "question_original": item.question,
                "question": question,
                "options": item.options,
                "prediction": prediction,
                "reference": item.answer,
                "correct": prediction == item.answer,
                "route": route,
                "retrieved_event_ids": retrieved_event_ids,
                "rerank_rationale": rerank_rationale,
                "rationale": rationale,
                "evidence": evidence_groups,
                "task_name": item.task_name,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            _append_jsonl(predictions_path, row)
            rows.append(row)
            print(
                f"[{number}/{len(selected)}] {item.uid}: {route} -> {prediction} "
                f"(reference {item.answer})",
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
            print(
                f"[{number}/{len(selected)}] FAILED {item.uid}: "
                f"{type(error).__name__}: {_safe_error(error)}",
                flush=True,
            )

    by_route: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_route.setdefault(str(row["route"]), []).append(row)
    report = {
        "requested": len(selected),
        "completed": len(rows),
        "failed": len(selected) - len(rows),
        "correct": sum(bool(row["correct"]) for row in rows),
        "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows)
        if rows
        else None,
        "prediction_counts": dict(Counter(str(row["prediction"]) for row in rows)),
        "by_route": {
            route: {
                "count": len(items),
                "correct": sum(bool(item["correct"]) for item in items),
                "accuracy": sum(bool(item["correct"]) for item in items) / len(items),
            }
            for route, items in sorted(by_route.items())
        },
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


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


if __name__ == "__main__":
    main()
