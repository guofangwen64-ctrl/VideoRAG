"""Aggregate explicitly separated metrics and expose complete event evidence."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from medhorizon_videorag.graph_rag.phase_candidate_metrics import summarize_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    metrics, traces, events = [], [], []
    for name in args.run_dir:
        root = Path(name)
        for filename, destination in [
            ("question_metrics.jsonl", metrics),
            ("candidate_traces.jsonl", traces),
            ("event_catalog.jsonl", events),
        ]:
            destination.extend(
                json.loads(line)
                for line in (root / filename).read_text().splitlines()
                if line.strip()
            )
    keys = [(r["video_key"], r["id"]) for r in metrics]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate video/QA rows across runs; compare runs separately")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    report = summarize_metrics(metrics)
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    for name, rows in [
        ("question_metrics.jsonl", metrics),
        ("candidate_traces.jsonl", traces),
        ("event_catalog.jsonl", events),
    ]:
        (output / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
