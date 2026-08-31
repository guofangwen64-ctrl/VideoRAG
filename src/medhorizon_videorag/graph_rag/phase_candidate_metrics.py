"""Shared metric definitions; hidden references enter only after inference."""

from __future__ import annotations

import math

from .phase_candidate_flow import phase_key

METRIC_VERSION = "phase-candidate-metrics-v3"


def _union(windows):
    merged = []
    for start, end in sorted(
        (float(w["start_seconds"]), float(w["end_seconds"])) for w in windows
    ):
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError("Invalid evaluation window")
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def interval_scores(predicted, reference):
    if not reference:
        return {"recall": None, "iou": None}
    pred, gold = _union(predicted), _union(reference)
    intersection = sum(max(0, min(b, d) - max(a, c)) for a, b in pred for c, d in gold)
    gold_length = sum(b - a for a, b in gold)
    union = sum(b - a for a, b in pred) + gold_length - intersection
    return {"recall": intersection / gold_length, "iou": intersection / union}


def question_metrics(
    trace,
    *,
    reference_answer=None,
    weak_phase_anchors=None,
    gold_phase_windows=None,
    gold_evidence_windows=None,
    evidence_recall_threshold=0.5,
):
    if not 0 < evidence_recall_threshold <= 1:
        raise ValueError("Evidence recall threshold must be in (0, 1]")
    rows = trace["candidates"]
    matching = [
        c for c in rows if phase_key(c["label"]) == phase_key(trace["phase_query"])
    ]
    non_counter = [c for c in matching if c["decision"] != "contradicted"]
    by_id = {c["candidate_id"]: c for c in rows}
    retrieved = by_id.get(trace.get("retrieval_top1_candidate_id"))
    verified = by_id.get(trace.get("verified_candidate_id"))
    gold_phase_windows = gold_phase_windows or []
    weak_phase_anchors = weak_phase_anchors or []
    gold_evidence_windows = gold_evidence_windows or []
    weak = interval_scores(non_counter, weak_phase_anchors)
    gold = interval_scores(non_counter, gold_phase_windows)
    packet = trace.get("reader_input") or {}
    selected_tracks = set(trace.get("selected_track_ids", []))
    selected_clips = {
        cid
        for t in packet.get("candidate_tracks", [])
        if t["track_id"] in selected_tracks
        for cid in t["reader_clip_ids"]
    }
    used = [
        g for g in packet.get("evidence_groups", []) if g["clip_id"] in selected_clips
    ]
    attempted = trace.get("inference_requested", False)
    answer_correct = (
        (trace.get("prediction") == reference_answer)
        if attempted and reference_answer is not None
        else None
    )
    evidence_correct = None
    if trace.get("status") == "completed" and gold_evidence_windows:
        evidence_correct = all(
            interval_scores(used, [window])["recall"] >= evidence_recall_threshold
            for window in gold_evidence_windows
        )
    metrics = {
        "candidate_name_coverage": bool(matching),
        "candidate_non_counter_coverage": bool(non_counter),
        "candidate_positive_coverage": any(
            c["evidence_role"] == "positive" for c in matching
        ),
        "source_candidate_rank1_name_coverage": any(
            c.get("rank") == 1 for c in matching
        ),
        "graph_primary_phase_name_survival": bool(
            trace.get("graph_primary_phase_matches", [])
        ),
        "graph_candidate_record_survival": any(
            c["persistent_candidate_record"] for c in matching
        ),
        "runtime_candidate_interval_grounding": any(
            c["event_support"] for c in non_counter
        ),
        "retrieval_top1_name_match": bool(
            retrieved
            and phase_key(retrieved["label"]) == phase_key(trace["phase_query"])
        ),
        "verified_selection_name_match": bool(
            verified and phase_key(verified["label"]) == phase_key(trace["phase_query"])
        )
        if attempted
        else None,
        "phase_time_recall_gold": gold["recall"],
        "phase_time_union_iou_gold": gold["iou"],
        "phase_time_best_candidate_iou_gold": max(
            (interval_scores([c], gold_phase_windows)["iou"] for c in non_counter),
            default=0.0,
        )
        if gold_phase_windows
        else None,
        "weak_phase_anchor_recall": weak["recall"],
        "weak_phase_anchor_union_iou": weak["iou"],
        "weak_phase_anchor_any_overlap": weak["recall"] > 0
        if weak["recall"] is not None
        else None,
        "weak_phase_anchor_best_candidate_iou": max(
            (interval_scores([c], weak_phase_anchors)["iou"] for c in non_counter),
            default=0.0,
        )
        if weak_phase_anchors
        else None,
        "retrieval_top1_weak_anchor_recall": interval_scores(
            [retrieved] if retrieved else [], weak_phase_anchors
        )["recall"],
        "answer_correct_all_requested": answer_correct,
        "answer_correct_completed": answer_correct
        if trace.get("status") == "completed"
        else None,
        "evidence_interval_correct": evidence_correct,
        "answer_and_evidence_interval_correct": bool(
            answer_correct and evidence_correct
        )
        if answer_correct is not None and evidence_correct is not None
        else None,
    }
    return {
        "metric_version": METRIC_VERSION,
        "id": trace["id"],
        "video_key": trace["video_key"],
        "metrics": metrics,
        "evaluation_only": {
            "reference_answer": reference_answer,
            "weak_phase_anchors": weak_phase_anchors,
            "gold_phase_windows": gold_phase_windows,
            "gold_evidence_windows": gold_evidence_windows,
            "selected_evidence_intervals": [
                {k: g[k] for k in ["clip_id", "start_seconds", "end_seconds"]}
                for g in used
            ],
            "evidence_recall_threshold": evidence_recall_threshold,
            "clinical_evidence_correctness": "not_annotated",
        },
    }


def summarize_metrics(rows):
    if any(row.get("metric_version") != METRIC_VERSION for row in rows):
        raise ValueError("Do not aggregate different metric protocols")
    keys = sorted({k for row in rows for k in row["metrics"]})
    return {
        "metric_version": METRIC_VERSION,
        "questions": len(rows),
        "metrics": {
            key: {
                "value": sum(values) / len(values) if values else None,
                "value_sum": sum(values),
                "denominator": len(values),
                "unavailable": len(rows) - len(values),
            }
            for key in keys
            for values in [
                [row["metrics"][key] for row in rows if row["metrics"][key] is not None]
            ]
        },
        "definitions": {
            "name_coverage": "Any same-name candidate, including contradicted; not usable evidence coverage.",
            "non_counter": "Decision is not contradicted; insufficient remains uncertain, not positive.",
            "source_rank1": "Rank within the source segment candidate file, never final retrieval Top1.",
            "graph_survival": "Primary label nodes and source-segment candidate metadata are reported separately; metadata is not a candidate node.",
            "retrieval_top1": "Actual deterministic pre-verifier selection; exact-name filtering makes name match a routing diagnostic, not an independent phase recognition score.",
            "time": "Duration-weighted interval-union recall and IoU; weak anchors and reviewed gold phase windows have separate denominators.",
            "evidence": "All reviewed gold evidence windows must meet the explicit recall threshold using clips cited through selected Reader tracks. This is temporal coverage, not clinical adjudication.",
            "missing": "No model run or no applicable annotation => null and excluded denominator; never silently scored as correct/incorrect.",
        },
    }
