"""Augment an observation evidence graph with auditable semantic hypotheses."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medhorizon_videorag.graph_rag import (
    augment_with_semantic_hypotheses,
    load_evidence_graph,
    load_semantic_hypotheses,
    write_semantic_layer_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--hypotheses", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-instrument-gap-events", type=int, default=1)
    parser.add_argument(
        "--instrument-track-source",
        choices=("semantic_hypotheses", "appearance_mentions"),
        default="semantic_hypotheses",
        help=(
            "Build tracks from inferred instrument labels or from observation-first "
            "visible-instrument mentions."
        ),
    )
    args = parser.parse_args()

    graph = load_evidence_graph(args.graph)
    hypotheses = load_semantic_hypotheses(args.hypotheses)
    artifacts = augment_with_semantic_hypotheses(
        graph,
        hypotheses,
        max_instrument_gap_events=args.max_instrument_gap_events,
        instrument_track_source=args.instrument_track_source,
    )
    write_semantic_layer_artifacts(artifacts, args.output_dir)
    report = artifacts.report
    print(
        f"Built {report['schema_version']} for {report['video_id']}: "
        f"{report['phase_hypothesis_count']} phases, "
        f"{report['phase_boundary_count']} boundaries, "
        f"{report['instrument_track_count']} "
        f"{report['instrument_track_source']} instrument tracks -> "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
