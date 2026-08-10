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
        chunks = []
        dataset = MedHorizonDataset(args.annotations)
        video_root = Path(args.video_root) if args.video_root else None
        for video in dataset.iter_videos():
            path = video_root / video.video_path if video_root else Path(video.video_path)
            chunks.extend(chunker.chunk(video.key, str(path)))
        write_jsonl(args.output, (chunk.to_dict() for chunk in chunks))
        print(f"Wrote {len(chunks)} chunks to {args.output}")
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
