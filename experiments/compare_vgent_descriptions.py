"""Create a paired report for two structured VGent clip-description runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medhorizon_videorag.vgent_baseline.description import (  # noqa: E402
    find_summary_rule_violations,
)


def _load(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["clip_id"])] = row
    return rows


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    descriptions = [row["description"] for row in rows]
    return {
        "count": len(rows),
        "summary_rule_violation_count": sum(
            bool(find_summary_rule_violations(str(item["summary"])))
            for item in descriptions
        ),
        "nonempty_medical_inference_count": sum(
            bool(item["medical_inferences"]) for item in descriptions
        ),
        "nonempty_uncertainty_count": sum(
            bool(item["uncertainties"]) for item in descriptions
        ),
        "mean_summary_words": round(
            mean(len(str(item["summary"]).split()) for item in descriptions), 3
        ),
        "mean_elapsed_seconds": round(
            mean(float(row.get("elapsed_seconds", 0)) for row in rows), 3
        ),
        "total_generation_attempts": sum(
            int(row.get("generation_attempts", 1)) for row in rows
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline = _load(Path(args.baseline))
    candidate = _load(Path(args.candidate))
    matched_ids = sorted(
        baseline.keys() & candidate.keys(),
        key=lambda clip_id: int(candidate[clip_id]["clip_index"]),
    )
    if not matched_ids:
        raise ValueError("No matching clip IDs between baseline and candidate")
    baseline_rows = [baseline[clip_id] for clip_id in matched_ids]
    candidate_rows = [candidate[clip_id] for clip_id in matched_ids]
    report = {
        "baseline_path": args.baseline,
        "candidate_path": args.candidate,
        "matched_clip_count": len(matched_ids),
        "baseline": _summary(baseline_rows),
        "candidate": _summary(candidate_rows),
        "pairs": [
            {
                "clip_id": clip_id,
                "clip_index": candidate[clip_id]["clip_index"],
                "start_seconds": candidate[clip_id]["start_seconds"],
                "end_seconds": candidate[clip_id]["end_seconds"],
                "baseline_summary": baseline[clip_id]["description"]["summary"],
                "candidate_summary": candidate[clip_id]["description"]["summary"],
                "baseline_medical_inferences": baseline[clip_id]["description"][
                    "medical_inferences"
                ],
                "candidate_medical_inferences": candidate[clip_id]["description"][
                    "medical_inferences"
                ],
            }
            for clip_id in matched_ids
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Compared {len(matched_ids)} clips -> {output}")


if __name__ == "__main__":
    main()
