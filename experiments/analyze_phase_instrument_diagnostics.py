"""Diagnose phase-instrument QA failures from saved graph-reader artifacts."""

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
from medhorizon_videorag.datasets import MedHorizonDataset, parse_temporal_query


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
    parser.add_argument("--output", required=True)
    parser.add_argument("--details", required=True)
    args = parser.parse_args()

    dataset = MedHorizonDataset(args.annotations)
    phase_anchors = _phase_anchors(dataset)
    details: list[dict[str, Any]] = []
    videos: dict[str, dict[str, Any]] = {}
    for entry in args.run_root:
        video_id, run_root = _parse_named_path(entry)
        root = Path(run_root)
        video_rows = _diagnose_video(video_id, root, phase_anchors.get(video_id, {}))
        details.extend(video_rows)
        videos[video_id] = _summarize(video_rows, root)

    report = {
        "annotations": args.annotations,
        "videos": videos,
        "overall": _summarize(details, None),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_jsonl(args.details, details)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {output}\nDetails: {args.details}")


def _diagnose_video(
    video_id: str, root: Path, anchors: dict[str, list[dict[str, float]]]
) -> list[dict[str, Any]]:
    reader_dir = _find_reader_dir(root)
    rows = _read_jsonl(reader_dir / "predictions.jsonl")
    graph_report = _read_json(root / "phase_instrument_graph_eval/report.json")
    semantic_report = _read_json(root / "combined_semantic_graph/semantic_graph_report.json")
    phase_segments = _read_json(root / "sequence_phase_segments.json")["segments"]
    accepted_segments = [
        row for row in phase_segments if bool(row.get("mapping_accepted"))
    ]

    details = []
    for row in rows:
        fallback = row.get("phase_fallback") or {}
        verification = fallback.get("verification") or {}
        reader_input = row.get("reader_input") or {}
        candidate_tracks = reader_input.get("candidate_tracks") or []
        selected_track_ids = list(row.get("selected_track_ids") or [])
        selected_tracks = [
            track
            for track in candidate_tracks
            if str(track.get("track_id")) in set(selected_track_ids)
        ]
        phase = str(row.get("phase_query") or "")
        selected_segment_id = verification.get("selected_segment_id")
        selected_candidate = _find_segment(
            fallback.get("retrieval_candidates") or [], selected_segment_id
        )
        anchor_windows = anchors.get(_canonical(phase), [])
        detail = {
            "id": str(row.get("id")),
            "video_key": video_id,
            "phase_query": phase,
            "status": row.get("status"),
            "phase_route": row.get("phase_route"),
            "prediction": row.get("prediction"),
            "reference": row.get("reference"),
            "correct": row.get("correct"),
            "persistent_phase_available": bool(
                semantic_report.get("phase_hypothesis_count")
            ),
            "persistent_phase_coverage_video": graph_report.get("completed"),
            "accepted_named_phase_segment_count_video": len(accepted_segments),
            "fallback_selected_segment_id": selected_segment_id,
            "fallback_verification_confidence": verification.get("confidence"),
            "fallback_top_segment_ids": list(fallback.get("top_segment_ids") or []),
            "fallback_top_score": _score_at(fallback, 0),
            "fallback_score_margin": _score_margin(fallback),
            "fallback_selected_segment": selected_candidate,
            "weak_phase_anchor_windows": anchor_windows,
            "selected_segment_overlaps_weak_anchor": _overlaps_any(
                selected_candidate, anchor_windows
            ),
            "candidate_track_count": len(candidate_tracks),
            "selected_track_ids": selected_track_ids,
            "selected_tracks": [_compact_track(track) for track in selected_tracks],
            "top_candidate_tracks": [
                _compact_track(track) for track in candidate_tracks[:5]
            ],
            "correct_option_track_overlap": _max_option_overlap(
                row.get("reference"), row.get("options") or [], candidate_tracks
            ),
            "predicted_option_track_overlap": _max_option_overlap(
                row.get("prediction"), row.get("options") or [], candidate_tracks
            ),
            "unresolved_reason": row.get("unresolved_reason"),
        }
        details.append(detail)
    return details


def _summarize(rows: list[dict[str, Any]], root: Path | None) -> dict[str, Any]:
    completed = [row for row in rows if row["status"] == "completed"]
    weak_anchor_rows = [
        row for row in rows if row["fallback_selected_segment_id"] is not None
    ]
    return {
        "run_root": str(root) if root is not None else None,
        "requested": len(rows),
        "completed": len(completed),
        "correct": sum(bool(row["correct"]) for row in completed),
        "accuracy_all_requested": (
            sum(bool(row["correct"]) for row in completed) / len(rows)
            if rows
            else None
        ),
        "accuracy_resolved": (
            sum(bool(row["correct"]) for row in completed) / len(completed)
            if completed
            else None
        ),
        "status_counts": dict(Counter(str(row["status"]) for row in rows)),
        "phase_route_counts": dict(Counter(str(row["phase_route"]) for row in rows)),
        "persistent_phase_available_count": sum(
            bool(row["persistent_phase_available"]) for row in rows
        ),
        "weak_anchor_available_count": sum(
            bool(row["weak_phase_anchor_windows"]) for row in rows
        ),
        "selected_segment_weak_anchor_overlap_count": sum(
            row["selected_segment_overlaps_weak_anchor"] is True for row in rows
        ),
        "selected_segment_weak_anchor_overlap_rate": (
            sum(row["selected_segment_overlaps_weak_anchor"] is True for row in rows)
            / len(weak_anchor_rows)
            if weak_anchor_rows
            else None
        ),
        "mean_candidate_track_count": (
            sum(int(row["candidate_track_count"]) for row in rows) / len(rows)
            if rows
            else None
        ),
        "correct_option_any_track_overlap_count": sum(
            row["correct_option_track_overlap"]["max_overlap"] > 0 for row in rows
        ),
    }


def _phase_anchors(
    dataset: MedHorizonDataset,
) -> dict[str, dict[str, list[dict[str, float]]]]:
    anchors: dict[str, dict[str, list[dict[str, float]]]] = {}
    for qa in dataset.questions:
        if qa.task_name != "Action Recognition":
            continue
        temporal = parse_temporal_query(qa.question)
        if temporal is None or temporal.kind != "range":
            continue
        phase = _answer_text(qa.answer, qa.options)
        if not phase:
            continue
        anchors.setdefault(qa.video_key, {}).setdefault(_canonical(phase), []).append(
            {
                "start_seconds": temporal.start_seconds,
                "end_seconds": temporal.end_seconds,
                "source_qa_uid": qa.uid,
                "phase_label": phase,
            }
        )
    return anchors


def _parse_named_path(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError("Expected VIDEO=DIR")
    name, path = value.split("=", 1)
    if not name or not path:
        raise ValueError("VIDEO and DIR must be non-empty")
    return name, path


def _find_reader_dir(root: Path) -> Path:
    matches = sorted(root.glob("phase_instrument_reader*/predictions.jsonl"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one phase_instrument_reader*/predictions.jsonl under {root}; "
            f"found {len(matches)}"
        )
    return matches[0].parent


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _find_segment(
    candidates: list[dict[str, Any]], segment_id: str | None
) -> dict[str, Any] | None:
    if not segment_id:
        return None
    for item in candidates:
        if str(item.get("segment_id")) == str(segment_id):
            return {
                "segment_id": str(item.get("segment_id")),
                "sequence_index": item.get("sequence_index"),
                "start_seconds": item.get("start_seconds"),
                "end_seconds": item.get("end_seconds"),
                "activity_label": item.get("activity_label"),
                "retrieval_score": item.get("retrieval_score"),
                "direct_phase_hits": item.get("direct_phase_hits"),
                "activity_cue_hits": item.get("activity_cue_hits"),
                "phase_specific_cue_hits": item.get("phase_specific_cue_hits"),
            }
    return None


def _score_at(fallback: dict[str, Any], index: int) -> float | None:
    candidates = fallback.get("retrieval_candidates") or []
    if index >= len(candidates):
        return None
    value = candidates[index].get("retrieval_score")
    return float(value) if value is not None else None


def _score_margin(fallback: dict[str, Any]) -> float | None:
    first = _score_at(fallback, 0)
    second = _score_at(fallback, 1)
    if first is None or second is None:
        return None
    return round(first - second, 4)


def _overlaps_any(
    segment: dict[str, Any] | None, windows: list[dict[str, float]]
) -> bool | None:
    if segment is None:
        return None
    if not windows:
        return None
    start = float(segment["start_seconds"])
    end = float(segment["end_seconds"])
    return any(
        start < float(window["end_seconds"]) and float(window["start_seconds"]) < end
        for window in windows
    )


def _compact_track(track: dict[str, Any]) -> dict[str, Any]:
    return {
        "track_id": track.get("track_id"),
        "graph_rank": track.get("graph_rank"),
        "graph_score": track.get("graph_score"),
        "label": track.get("label"),
        "appearance_family": track.get("appearance_family"),
        "surface_forms": list(track.get("surface_forms") or [])[:5],
        "action_roles": list(track.get("action_roles") or []),
        "reader_clip_ids": list(track.get("reader_clip_ids") or []),
    }


def _max_option_overlap(
    label: str | None, options: list[str], tracks: list[dict[str, Any]]
) -> dict[str, Any]:
    option = _option_by_label(label, options)
    option_tokens = _tokens(_option_text(option or ""))
    best = {"option": option, "track_id": None, "max_overlap": 0, "matched_tokens": []}
    for track in tracks:
        text = " ".join(
            [
                str(track.get("label") or ""),
                str(track.get("appearance_family") or ""),
                " ".join(str(item) for item in track.get("surface_forms") or []),
                " ".join(str(item) for item in track.get("action_roles") or []),
            ]
        )
        matched = sorted(option_tokens & _tokens(text))
        if len(matched) > int(best["max_overlap"]):
            best = {
                "option": option,
                "track_id": track.get("track_id"),
                "max_overlap": len(matched),
                "matched_tokens": matched,
            }
    return best


def _answer_text(label: str | None, options: list[str]) -> str | None:
    option = _option_by_label(label, options)
    return _option_text(option) if option else None


def _option_by_label(label: str | None, options: list[str]) -> str | None:
    if not label:
        return None
    wanted = str(label).strip().upper()
    for index, option in enumerate(options):
        parsed = _option_label(option, index)
        if parsed == wanted:
            return option
    return None


def _option_label(option: str, index: int) -> str:
    match = re.match(r"^\s*([A-Za-z0-9]+)[.):：]", option)
    return match.group(1).upper() if match else chr(ord("A") + index)


def _option_text(option: str) -> str:
    return re.sub(r"^\s*[A-Za-z0-9]+[.):：]\s*", "", option).strip()


def _tokens(value: str) -> set[str]:
    stop = {"instrument", "surgical", "forceps", "holder", "large", "small"}
    return {token for token in re.findall(r"[a-z0-9]+", value.lower()) if token not in stop}


def _canonical(value: str) -> str:
    return "".join(re.findall(r"[a-z0-9]+", value.lower()))


if __name__ == "__main__":
    main()
