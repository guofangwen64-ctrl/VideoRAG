"""Pair two QA prediction files to measure whether a condition helps per question."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medhorizon_videorag.core.io import read_jsonl, write_jsonl  # noqa: E402


def _correct(row: dict) -> bool:
    return row.get("reference") is not None and str(row.get("prediction", "")).strip().lower() == str(row["reference"]).strip().lower()


def _summary(rows: list[dict]) -> dict[str, object]:
    outcomes = Counter(row["outcome"] for row in rows)
    return {
        "paired_questions": len(rows),
        "both_correct": outcomes["both_correct"],
        "left_only_correct": outcomes["left_only_correct"],
        "right_only_correct": outcomes["right_only_correct"],
        "both_wrong": outcomes["both_wrong"],
        "left_accuracy": sum(row["left_correct"] for row in rows) / len(rows) if rows else 0.0,
        "right_accuracy": sum(row["right_correct"] for row in rows) / len(rows) if rows else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, help="First prediction JSONL, e.g. question-only")
    parser.add_argument("--right", required=True, help="Second prediction JSONL, e.g. video evidence")
    parser.add_argument("--left-name", default="left")
    parser.add_argument("--right-name", default="right")
    parser.add_argument("--output", default="artifacts/qa_pairwise_report.json")
    parser.add_argument("--details", default="artifacts/qa_pairwise_details.jsonl")
    args = parser.parse_args()

    left = {str(row["id"]): row for row in read_jsonl(args.left) if row.get("id") is not None}
    right = {str(row["id"]): row for row in read_jsonl(args.right) if row.get("id") is not None}
    common_ids = sorted(left.keys() & right.keys(), key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))
    details: list[dict] = []
    for qa_id in common_ids:
        first, second = left[qa_id], right[qa_id]
        first_correct, second_correct = _correct(first), _correct(second)
        outcome = (
            "both_correct" if first_correct and second_correct else
            "left_only_correct" if first_correct else
            "right_only_correct" if second_correct else "both_wrong"
        )
        details.append({
            "id": qa_id, "task_name": second.get("metadata", {}).get("task_name", first.get("metadata", {}).get("task_name", "unknown")),
            "reference": second.get("reference", first.get("reference")),
            "left_prediction": first.get("prediction"), "right_prediction": second.get("prediction"),
            "left_correct": first_correct, "right_correct": second_correct, "outcome": outcome,
        })
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in details:
        by_task[row["task_name"]].append(row)
    report = {
        "left_name": args.left_name, "right_name": args.right_name,
        "left_path": args.left, "right_path": args.right,
        "left_only_ids": len(left) - len(common_ids), "right_only_ids": len(right) - len(common_ids),
        "overall": _summary(details),
        "by_task": {name: _summary(rows) for name, rows in sorted(by_task.items())},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_jsonl(args.details, details)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {output}\nDetails: {args.details}")


if __name__ == "__main__":
    main()
