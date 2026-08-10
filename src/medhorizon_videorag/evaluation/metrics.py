from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

from medhorizon_videorag.core.schemas import Prediction


def _tokens(value: str) -> list[str]:
    return re.findall(r"[\w]+|[\u4e00-\u9fff]", value.lower())


def _f1(prediction: str, reference: str) -> float:
    pred, ref = Counter(_tokens(prediction)), Counter(_tokens(reference))
    overlap = sum((pred & ref).values())
    if not pred or not ref:
        return float(pred == ref)
    precision, recall = overlap / sum(pred.values()), overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def evaluate_predictions(predictions: Iterable[Prediction]) -> dict[str, float | int]:
    rows = [p for p in predictions if p.reference is not None]
    if not rows:
        return {"count": 0, "exact_match": 0.0, "token_f1": 0.0}
    exact = [p.prediction.strip().lower() == p.reference.strip().lower() for p in rows]
    return {
        "count": len(rows),
        "exact_match": sum(exact) / len(rows),
        "token_f1": sum(_f1(p.prediction, p.reference or "") for p in rows) / len(rows),
    }
