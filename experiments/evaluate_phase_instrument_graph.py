"""Evaluate phase-boundary to instrument-track retrieval on selected questions."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medhorizon_videorag.datasets import MedHorizonDataset
from medhorizon_videorag.graph_rag import (
    extract_phase_name,
    load_evidence_graph,
    retrieve_phase_boundary_instruments,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--video-key", required=True)
    parser.add_argument("--qa-uids", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--context-events", type=int, default=1)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    graph = load_evidence_graph(args.graph)
    dataset = MedHorizonDataset(args.annotations)
    requested = [item.strip() for item in args.qa_uids.split(",") if item.strip()]
    by_uid = {
        str(item.uid): item
        for item in dataset.questions
        if item.video_key == args.video_key
    }
    missing = sorted(set(requested) - by_uid.keys())
    if missing:
        raise ValueError(f"Missing requested QA IDs: {missing}")

    rows = []
    for uid in requested:
        item = by_uid[uid]
        phase = extract_phase_name(item.question)
        if not phase:
            raise ValueError(f"Cannot extract phase from QA {uid}: {item.question}")
        try:
            retrieval = retrieve_phase_boundary_instruments(
                graph, phase, context_events=args.context_events
            )
            unresolved_reason = None
        except ValueError as error:
            retrieval = None
            unresolved_reason = str(error)
        option_map = {
            _canonical(_option_text(option)): _option_label(option, index)
            for index, option in enumerate(item.options)
        }
        matched = [
            instrument
            for instrument in (retrieval["instruments"] if retrieval else [])
            if str(instrument["canonical_label"]) in option_map
        ]
        prediction = option_map[str(matched[0]["canonical_label"])] if matched else None
        row = {
            "id": uid,
            "video_key": item.video_key,
            "question": item.question,
            "options": item.options,
            "phase_query": phase,
            "prediction": prediction,
            "reference": item.answer,
            "correct": prediction == item.answer,
            "matched_instrument": matched[0] if matched else None,
            "retrieval": retrieval,
            "unresolved_reason": unresolved_reason,
            "candidate_aware_diagnostic": True,
        }
        rows.append(row)
        print(
            f"{uid}: phase={phase} instrument="
            f"{matched[0]['label'] if matched else None} -> {prediction} "
            f"(reference {item.answer})"
        )
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    completed = [row for row in rows if row["prediction"] is not None]
    report: dict[str, Any] = {
        "requested": len(rows),
        "completed": len(completed),
        "unresolved": len(rows) - len(completed),
        "correct": sum(bool(row["correct"]) for row in rows),
        "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows)
        if rows
        else None,
        "resolved_accuracy": sum(bool(row["correct"]) for row in completed)
        / len(completed)
        if completed
        else None,
        "prediction_counts": dict(Counter(str(row["prediction"]) for row in rows)),
        "context_events": args.context_events,
        "candidate_aware_diagnostic": True,
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _option_text(option: str) -> str:
    return re.sub(r"^\s*[A-Za-z0-9]+[.):：]\s*", "", option).strip()


def _option_label(option: str, index: int) -> str:
    match = re.match(r"^\s*([A-Za-z0-9]+)[.):：]", option)
    return match.group(1).upper() if match else chr(ord("A") + index)


def _canonical(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


if __name__ == "__main__":
    main()
