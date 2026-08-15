"""Temporal retrieval metrics for MedHorizon QA with recovered evidence."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

from medhorizon_videorag.core.schemas import RetrievalResult
from medhorizon_videorag.datasets.temporal_ground_truth import TemporalEvidence


def temporal_iou(first: tuple[float, float], second: tuple[float, float]) -> float:
    start = max(first[0], second[0])
    end = min(first[1], second[1])
    intersection = max(0.0, end - start)
    union = max(first[1], second[1]) - min(first[0], second[0])
    return intersection / union if union else float(first == second)


def evaluate_retrieval(
    evidence: Sequence[TemporalEvidence], retrieved: Sequence[Sequence[RetrievalResult]], top_ks: Sequence[int], iou_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate direct ranges and points; weak phase anchors are intentionally excluded."""
    if len(evidence) != len(retrieved):
        raise ValueError("Evidence and retrieval result counts differ")
    top_ks = sorted(set(top_ks))
    range_values: dict[int, list[float]] = defaultdict(list)
    point_values: dict[int, list[float]] = defaultdict(list)
    details: list[dict[str, Any]] = []

    for item, results in zip(evidence, retrieved, strict=True):
        row: dict[str, Any] = {
            "qa_uid": item.qa_uid, "video_id": item.video_key, "task_name": item.task_name,
            "method": item.method, "question": item.question, "windows": item.to_dict()["windows"],
        }
        for top_k in top_ks:
            candidates = results[:top_k]
            if item.method == "direct_range":
                target = item.windows[0]
                best_iou = max((
                    temporal_iou(target, (candidate.chunk.start_seconds, candidate.chunk.end_seconds))
                    if candidate.chunk.video_id == item.video_key else 0.0
                    for candidate in candidates
                ), default=0.0)
                range_values[top_k].append(best_iou)
                row[f"best_iou_at_{top_k}"] = best_iou
                row[f"hit_at_{top_k}"] = best_iou >= iou_threshold
            elif item.method == "direct_point":
                point = item.windows[0][0]
                hit = any(
                    candidate.chunk.video_id == item.video_key
                    and candidate.chunk.start_seconds <= point <= candidate.chunk.end_seconds
                    for candidate in candidates
                )
                point_values[top_k].append(float(hit))
                row[f"point_hit_at_{top_k}"] = hit
        row["retrieved"] = [
            {"chunk_id": result.chunk.id, "video_id": result.chunk.video_id, "start_seconds": result.chunk.start_seconds,
             "end_seconds": result.chunk.end_seconds, "score": result.score}
            for result in results
        ]
        details.append(row)

    def range_metrics(top_k: int) -> dict[str, float]:
        values = range_values[top_k]
        return {
            f"recall_at_{top_k}": sum(value >= iou_threshold for value in values) / len(values) if values else 0.0,
            f"mean_best_iou_at_{top_k}": sum(values) / len(values) if values else 0.0,
        }

    def point_metrics(top_k: int) -> dict[str, float]:
        values = point_values[top_k]
        return {f"point_hit_at_{top_k}": sum(values) / len(values) if values else 0.0}

    return {
        "direct_range_questions": len(range_values[top_ks[0]]) if top_ks else 0,
        "direct_point_questions": len(point_values[top_ks[0]]) if top_ks else 0,
        "iou_threshold": iou_threshold,
        "range_metrics": {f"at_{top_k}": range_metrics(top_k) for top_k in top_ks},
        "point_metrics": {f"at_{top_k}": point_metrics(top_k) for top_k in top_ks},
    }, details
