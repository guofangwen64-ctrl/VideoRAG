"""Prepare non-destructive red-region pixelation probes for selected clips."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medhorizon_videorag.vgent_baseline import load_video_plan, save_video_plan
from medhorizon_videorag.vgent_baseline.redaction_probe import (
    prepare_redacted_video_plan,
)


def _parse_indices(value: str) -> list[int]:
    indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not indices or len(indices) != len(set(indices)) or any(i < 0 for i in indices):
        raise ValueError("--clip-indices must contain unique non-negative indices")
    return indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--clip-indices", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-red", type=int, default=70)
    parser.add_argument("--dominance-ratio", type=float, default=1.18)
    parser.add_argument("--dilation-size", type=int, default=21)
    parser.add_argument("--pixel-block-size", type=int, default=18)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    frames_dir = output_dir / "frames"
    plan = load_video_plan(args.manifest)
    derived_plan, report = prepare_redacted_video_plan(
        plan,
        clip_indices=_parse_indices(args.clip_indices),
        output_dir=frames_dir,
        min_red=args.min_red,
        dominance_ratio=args.dominance_ratio,
        dilation_size=args.dilation_size,
        pixel_block_size=args.pixel_block_size,
        jpeg_quality=args.jpeg_quality,
    )
    save_video_plan(derived_plan, output_dir / "redacted_manifest.json")
    (output_dir / "redaction_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared {len(report['clips'])} redacted clips in {output_dir}")
    for clip in report["clips"]:
        top = ",".join(
            str(item["frame_index"]) for item in clip["highest_redaction_frames"][:5]
        )
        print(
            f"{clip['clip_id']}: mean redacted="
            f"{clip['mean_redacted_fraction']:.3f}; top frames={top}"
        )


if __name__ == "__main__":
    main()
