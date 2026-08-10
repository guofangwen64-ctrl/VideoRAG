"""Produce a reproducible summary report for a MedHorizon JSONL split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medhorizon_videorag.datasets import MedHorizonDataset  # noqa: E402


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "N/A"
    hours, remainder = divmod(round(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}h {minutes:02d}m {seconds:02d}s"


def _print_counter(title: str, counter: dict[str, int]) -> None:
    print(f"\n{title}")
    for name, count in counter.items():
        print(f"  - {name}: {count}")


def print_report(report: dict[str, Any]) -> None:
    videos, questions = report["videos"], report["questions"]
    print("MedHorizon dataset report")
    print(f"Source: {report['source']}")
    print(f"Videos: {videos['count']} (duration available: {videos['with_duration']})")
    print(f"Total duration: {_format_duration(videos['total_duration_seconds'])}")
    print(f"Average duration: {_format_duration(videos['average_duration_seconds'])} ({videos['average_duration_seconds']} seconds)")
    print(f"QA questions: {questions['count']} ({questions['per_video']} per video)")
    _print_counter("Task names", report["task_categories"]["task_name"])
    _print_counter("Task classes", report["task_categories"]["task_class"])
    _print_counter("Reasoning categories", report["task_categories"]["category"])
    _print_counter("Question types", report["question_types"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="medhorizon_test.jsonl", help="Path to a MedHorizon JSONL annotation file")
    parser.add_argument("--output", help="Optional path for a machine-readable JSON report")
    args = parser.parse_args()

    report = MedHorizonDataset(args.annotations).report()
    print_report(report)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nJSON report written to: {destination}")


if __name__ == "__main__":
    main()
