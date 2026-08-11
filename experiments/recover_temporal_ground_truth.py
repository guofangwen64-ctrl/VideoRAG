"""Recover stated or weakly anchored temporal evidence from MedHorizon QA."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medhorizon_videorag.datasets import MedHorizonDataset, recover_evidence, recovery_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="medhorizon_test.jsonl")
    parser.add_argument("--output", default="artifacts/recovered_temporal_evidence.jsonl")
    parser.add_argument("--report", default="artifacts/temporal_recovery_report.json")
    args = parser.parse_args()

    evidence = recover_evidence(MedHorizonDataset(args.annotations))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(item.to_dict(), ensure_ascii=False) + "\n" for item in evidence), encoding="utf-8")
    report = recovery_report(evidence)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Evidence: {output}\nReport: {report_path}")


if __name__ == "__main__":
    main()
