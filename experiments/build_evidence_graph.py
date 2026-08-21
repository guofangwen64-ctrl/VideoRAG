"""Build a traceable evidence graph from observation-first clip descriptions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medhorizon_videorag.graph_rag import (
    build_evidence_graph,
    load_description_rows,
    load_manifest_frame_paths,
    write_evidence_graph_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptions", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--merge-threshold", type=float, default=0.45)
    parser.add_argument("--max-merged-clips", type=int, default=5)
    parser.add_argument("--max-representative-clips", type=int, default=3)
    args = parser.parse_args()

    rows = load_description_rows(args.descriptions)
    frame_paths = load_manifest_frame_paths(args.manifest)
    missing = sorted({str(row["clip_id"]) for row in rows} - frame_paths.keys())
    if missing:
        raise ValueError(f"Manifest is missing description clips: {missing[:5]}")
    artifacts = build_evidence_graph(
        rows,
        frame_paths_by_clip=frame_paths,
        merge_threshold=args.merge_threshold,
        max_merged_clips=args.max_merged_clips,
        max_representative_clips=args.max_representative_clips,
    )
    write_evidence_graph_artifacts(artifacts, args.output_dir)
    report = artifacts.report
    print(
        f"Built {report['schema_version']} for {report['video_id']}: "
        f"{report['node_count']} nodes, {report['edge_count']} edges, "
        f"{report['temporal_event_count']} temporal events -> {args.output_dir}"
    )


if __name__ == "__main__":
    main()
