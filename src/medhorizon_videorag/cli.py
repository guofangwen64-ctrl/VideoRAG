from __future__ import annotations

import argparse
import json
from pathlib import Path

from medhorizon_videorag.core.config import load_config
from medhorizon_videorag.core.io import read_jsonl, write_jsonl
from medhorizon_videorag.core.schemas import Chunk, Prediction, QAExample
from medhorizon_videorag.datasets import MedHorizonDataset
from medhorizon_videorag.evaluation import evaluate_predictions
from medhorizon_videorag.ingestion import VideoChunker
from medhorizon_videorag.pipelines import build_index, run_qa


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medrag")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("chunk", "index", "answer"):
        command = sub.add_parser(name)
        command.add_argument("--config", required=True)
    sub.choices["chunk"].add_argument("--annotations", required=True)
    sub.choices["chunk"].add_argument("--output", default="artifacts/chunks.jsonl")
    sub.choices["chunk"].add_argument("--frame-root", help="Directory for sampled frames (default: <artifact_dir>/frames)")
    sub.choices["chunk"].add_argument("--errors", help="Path for failed-video JSONL records (default: <artifact_dir>/chunk_errors.jsonl)")
    sub.choices["chunk"].add_argument("--restart", action="store_true", help="Discard existing chunk manifest and process every video again")
    sub.choices["chunk"].add_argument(
        "--video-root", help="Directory containing paths referenced by MedHorizon video_path (for example /mnt/medhorizon/videos)",
    )
    sub.choices["index"].add_argument("--chunks", required=True)
    sub.choices["answer"].add_argument("--annotations", required=True)
    sub.choices["answer"].add_argument("--output", required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--predictions", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "evaluate":
        rows = [Prediction(**row) for row in read_jsonl(args.predictions)]
        print(json.dumps(evaluate_predictions(rows), ensure_ascii=False, indent=2))
        return
    config = load_config(args.config)
    if args.command == "chunk":
        chunker = VideoChunker(**config.chunking)
        dataset = MedHorizonDataset(args.annotations)
        configured_root = config.data.get("video_root")
        video_root = Path(args.video_root or configured_root) if (args.video_root or configured_root) else None
        artifact_dir = Path(config.project.get("artifact_dir", "artifacts"))
        output = Path(args.output)
        frame_root = Path(args.frame_root) if args.frame_root else artifact_dir / "frames"
        errors = Path(args.errors) if args.errors else artifact_dir / "chunk_errors.jsonl"
        if args.restart and output.exists():
            output.unlink()
        completed = {row["video_id"] for row in read_jsonl(output)} if output.exists() else set()
        output.parent.mkdir(parents=True, exist_ok=True)
        errors.parent.mkdir(parents=True, exist_ok=True)
        processed = failed = written = 0
        with output.open("a", encoding="utf-8") as manifest, errors.open("a", encoding="utf-8") as error_log:
            for number, video in enumerate(dataset.iter_videos(), start=1):
                if video.key in completed:
                    continue
                processed += 1
                path = video_root / video.video_path if video_root else Path(video.video_path)
                try:
                    chunks = chunker.chunk(video.key, str(path), frame_root)
                except (OSError, ValueError, RuntimeError) as error:
                    failed += 1
                    error_log.write(json.dumps({"video_id": video.key, "video_path": str(path), "error": str(error)}, ensure_ascii=False) + "\n")
                    print(f"[{number}/{len(dataset.videos)}] FAILED {video.key}: {error}", flush=True)
                    continue
                for chunk in chunks:
                    manifest.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
                manifest.flush()
                written += len(chunks)
                incomplete = sum(len(chunk.frame_paths) < max(1, chunker.frames_per_chunk) for chunk in chunks)
                if incomplete:
                    error_log.write(json.dumps({
                        "video_id": video.key, "video_path": str(path), "warning": "incomplete_frame_decode",
                        "incomplete_chunks": incomplete, "total_chunks": len(chunks),
                    }, ensure_ascii=False) + "\n")
                    error_log.flush()
                suffix = f", {incomplete} incomplete" if incomplete else ""
                print(f"[{number}/{len(dataset.videos)}] {video.key}: {len(chunks)} chunks{suffix}", flush=True)
        print(f"Finished: {processed} processed, {written} chunks written, {failed} failed. Manifest: {output}")
    elif args.command == "index":
        chunks = [Chunk(**row) for row in read_jsonl(args.chunks)]
        build_index(chunks, config)
        print(f"Indexed {len(chunks)} chunks at {config.retrieval['index_path']}")
    else:
        examples = [QAExample(**row) for row in read_jsonl(args.annotations)]
        predictions = run_qa(examples, config)
        write_jsonl(args.output, (prediction.to_dict() for prediction in predictions))
        print(f"Wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
