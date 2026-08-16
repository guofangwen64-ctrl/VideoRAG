"""Evaluate MedHorizon multiple-choice predictions, including route/task slices."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medhorizon_videorag.core.io import read_jsonl  # noqa: E402
from medhorizon_videorag.core.schemas import Prediction  # noqa: E402
from medhorizon_videorag.evaluation import evaluate_predictions  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default="artifacts/qa_report.json")
    args = parser.parse_args()
    rows = [Prediction(**row) for row in read_jsonl(args.predictions)]
    by_task: dict[str, list[Prediction]] = defaultdict(list)
    by_route: dict[str, list[Prediction]] = defaultdict(list)
    for row in rows:
        by_task[row.metadata.get("task_name", "unknown")].append(row)
        by_route[row.metadata.get("route", "unknown")].append(row)
    report = {
        "overall": evaluate_predictions(rows),
        "by_task": {name: evaluate_predictions(items) for name, items in sorted(by_task.items())},
        "by_route": {name: evaluate_predictions(items) for name, items in sorted(by_route.items())},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
