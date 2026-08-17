"""Produce the fixed MedHorizon VideoRAG evaluation-protocol report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medhorizon_videorag.core.io import read_jsonl  # noqa: E402
from medhorizon_videorag.core.schemas import Prediction  # noqa: E402
from medhorizon_videorag.datasets import MedHorizonDataset, recover_evidence  # noqa: E402
from medhorizon_videorag.evaluation import evaluate_predictions  # noqa: E402


def _named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("Use NAME=PATH, for example --qa question_only=artifacts/qa.jsonl")
    name, path = value.split("=", 1)
    if not name or not path:
        raise ValueError("NAME and PATH must both be non-empty")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="medhorizon_test.jsonl")
    parser.add_argument("--qa", action="append", default=[], metavar="NAME=JSONL", help="Repeat for each QA condition")
    parser.add_argument("--retrieval", action="append", default=[], metavar="NAME=JSON", help="Repeat for retrieval reports")
    parser.add_argument("--output", default="artifacts/baseline_suite_report.json")
    args = parser.parse_args()

    dataset = MedHorizonDataset(args.annotations)
    evidence = {str(item.qa_uid): item for item in recover_evidence(dataset)}
    partitions = {
        "explicit_time": {qa_id for qa_id, item in evidence.items() if item.method in {"direct_range", "direct_point"} and item.source_field == "question"},
        "implicit_time": {qa_id for qa_id, item in evidence.items() if item.method in {"direct_range", "direct_point"} and item.source_field != "question"},
        "no_reliable_time": {qa_id for qa_id, item in evidence.items() if item.method in {"unresolved", "phase_anchor"}},
    }
    qa_conditions: dict[str, object] = {}
    for entry in args.qa:
        name, path = _named_path(entry)
        rows = {str(row["id"]): Prediction(**row) for row in read_jsonl(path) if row.get("id") is not None}
        qa_conditions[name] = {
            "path": str(path), "available_predictions": len(rows),
            "overall": evaluate_predictions(list(rows.values())),
            "by_protocol_partition": {
                partition: {
                    "expected_questions": len(ids), "available_predictions": sum(qa_id in rows for qa_id in ids),
                    "metrics": evaluate_predictions([rows[qa_id] for qa_id in ids if qa_id in rows]),
                }
                for partition, ids in partitions.items()
            },
        }
    retrieval_reports = {}
    for entry in args.retrieval:
        name, path = _named_path(entry)
        retrieval_reports[name] = {"path": str(path), "report": json.loads(path.read_text(encoding="utf-8"))}
    report = {
        "protocol": {
            "explicit_time": "Direct range/point time stated in the deployed rewritten question; report retrieval IoU/Hit and QA accuracy.",
            "implicit_time": "Direct time recoverable only from an original annotation field; report visual/hybrid QA and localization separately from oracle.",
            "no_reliable_time": "Unresolved or weak phase-anchor time evidence; do not report temporal localization, report end-to-end QA accuracy only.",
        },
        "dataset": args.annotations,
        "partition_counts": {name: len(ids) for name, ids in partitions.items()},
        "retrieval_conditions": retrieval_reports,
        "qa_conditions": qa_conditions,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
