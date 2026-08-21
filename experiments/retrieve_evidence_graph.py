"""Retrieve traceable event evidence from a v2.1 observation graph."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medhorizon_videorag.graph_rag import (
    DeterministicEventGraphRetriever,
    load_evidence_graph,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--question", action="append")
    inputs.add_argument("--questions-jsonl")
    parser.add_argument("--question-id", default="graph-query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument("--max-evidence-intervals", type=int, default=5)
    parser.add_argument("--max-representatives-per-event", type=int, default=2)
    parser.add_argument("--output")
    args = parser.parse_args()

    graph = load_evidence_graph(args.graph)
    questions = _load_questions(args.question, args.questions_jsonl, args.question_id)
    retriever = DeterministicEventGraphRetriever(
        max_hops=args.max_hops,
        max_evidence_intervals=args.max_evidence_intervals,
        max_representatives_per_event=args.max_representatives_per_event,
    )
    results = []
    for row in questions:
        video_id = row.get("video_id")
        if video_id is not None and str(video_id) != graph.video_id:
            raise ValueError(
                f"Question {row['id']} targets video {video_id}, not {graph.video_id}"
            )
        results.append(
            retriever.retrieve(
                question_id=str(row["id"]),
                question=str(row["question"]),
                graph=graph,
                top_k=args.top_k,
            ).to_dict()
        )

    if args.output:
        output = Path(args.output)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite retrieval output: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    else:
        for result in results:
            print(json.dumps(result, ensure_ascii=False, indent=2))


def _load_questions(
    inline: list[str] | None, jsonl_path: str | None, default_id: str
) -> list[dict[str, Any]]:
    if inline:
        return [
            {
                "id": default_id if len(inline) == 1 else f"{default_id}-{index:03d}",
                "question": question,
            }
            for index, question in enumerate(inline)
        ]
    assert jsonl_path is not None
    rows = [
        json.loads(line)
        for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("Question JSONL is empty")
    for row in rows:
        if not row.get("id") or not row.get("question"):
            raise ValueError("Each question row must contain non-empty id and question")
    return rows


if __name__ == "__main__":
    main()
