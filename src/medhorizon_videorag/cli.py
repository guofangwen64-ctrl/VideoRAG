from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from medhorizon_videorag.core.config import load_config
from medhorizon_videorag.core.io import read_jsonl
from medhorizon_videorag.core.schemas import Chunk, Prediction
from medhorizon_videorag.datasets import MedHorizonDataset
from medhorizon_videorag.evaluation import evaluate_predictions
from medhorizon_videorag.ingestion import VideoChunker
from medhorizon_videorag.pipelines import build_index, run_medhorizon_qa
from medhorizon_videorag.retrieval import HybridRetriever, NumpyVectorIndex, TemporalRetriever, VisualRetriever
from medhorizon_videorag.features import build_visual_embedder


def _retry_video_ids(path: str | Path, minimum_ratio: float) -> set[str]:
    selected: set[str] = set()
    for row in read_jsonl(path):
        if "error" in row:
            selected.add(str(row["video_id"]))
        elif row.get("warning") == "incomplete_frame_decode":
            ratio = row["incomplete_chunks"] / max(1, row["total_chunks"])
            if ratio >= minimum_ratio:
                selected.add(str(row["video_id"]))
    return selected


def _append_jsonl(handle, row: dict) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
            cwd=Path(__file__).resolve().parents[2],
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_versions() -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    for package in ("openai", "torch", "open-clip-torch", "transformers"):
        try:
            values[package] = version(package)
        except PackageNotFoundError:
            values[package] = None
    return values


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medrag")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("chunk", "index", "answer", "retrieve"):
        command = sub.add_parser(name)
        command.add_argument("--config", required=True)
    sub.choices["chunk"].add_argument("--annotations", required=True)
    sub.choices["chunk"].add_argument("--output", default="artifacts/chunks.jsonl")
    sub.choices["chunk"].add_argument("--frame-root", help="Directory for sampled frames (default: <artifact_dir>/frames)")
    sub.choices["chunk"].add_argument("--errors", help="Path for failed-video JSONL records (default: <artifact_dir>/chunk_errors.jsonl)")
    sub.choices["chunk"].add_argument("--restart", action="store_true", help="Discard existing chunk manifest and process every video again")
    sub.choices["chunk"].add_argument("--retry-errors", help="Retry failed/severely incomplete videos recorded in this JSONL")
    sub.choices["chunk"].add_argument("--retry-min-incomplete-ratio", type=float, default=0.1, help="Retry warning records at or above this incomplete-chunk ratio")
    sub.choices["chunk"].add_argument(
        "--video-root", help="Directory containing paths referenced by MedHorizon video_path (for example /mnt/medhorizon/videos)",
    )
    sub.choices["index"].add_argument("--chunks", required=True)
    sub.choices["answer"].add_argument("--annotations", required=True)
    sub.choices["answer"].add_argument("--output", required=True)
    sub.choices["answer"].add_argument("--limit", type=int, help="Only run the first N MedHorizon QA examples")
    sub.choices["answer"].add_argument("--top-k", type=int, help="Override retrieval.top_k for this run")
    sub.choices["answer"].add_argument("--reader-frame-root", help="Cache directory for dense Reader-stage frames")
    sub.choices["answer"].add_argument("--question-only", action="store_true", help="Do not retrieve or send video frames to the Reader")
    sub.choices["retrieve"].add_argument("--question", required=True)
    sub.choices["retrieve"].add_argument("--video-id", required=True)
    sub.choices["retrieve"].add_argument("--top-k", type=int)
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
        chunker = VideoChunker(**{name: config.chunking[name] for name in ("duration_seconds", "stride_seconds", "frames_per_chunk") if name in config.chunking})
        dataset = MedHorizonDataset(args.annotations)
        configured_root = config.data.get("video_root")
        video_root = Path(args.video_root or configured_root) if (args.video_root or configured_root) else None
        artifact_dir = Path(config.project.get("artifact_dir", "artifacts"))
        output = Path(args.output)
        frame_root = Path(args.frame_root) if args.frame_root else artifact_dir / "frames"
        errors = Path(args.errors) if args.errors else artifact_dir / "chunk_errors.jsonl"
        if args.restart and args.retry_errors:
            raise ValueError("--restart and --retry-errors cannot be used together")
        if args.restart and output.exists():
            output.unlink()
        existing_rows = read_jsonl(output) if output.exists() else []
        retry_ids = _retry_video_ids(args.retry_errors, args.retry_min_incomplete_ratio) if args.retry_errors else set()
        completed = {row["video_id"] for row in existing_rows} if not retry_ids else set()
        if retry_ids:
            print(f"Retrying {len(retry_ids)} videos from {args.retry_errors}", flush=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        errors.parent.mkdir(parents=True, exist_ok=True)
        processed = failed = written = 0
        replacements: dict[str, list[Chunk]] = {}
        with output.open("a", encoding="utf-8") as manifest, errors.open("a", encoding="utf-8") as error_log:
            for number, video in enumerate(dataset.iter_videos(), start=1):
                if retry_ids and video.key not in retry_ids:
                    continue
                if video.key in completed:
                    continue
                processed += 1
                path = video_root / video.video_path if video_root else Path(video.video_path)
                try:
                    chunks = chunker.chunk(
                        video.key, str(path), frame_root,
                        config.chunking.get("ffmpeg_fallback_min_incomplete_ratio"),
                    )
                except (OSError, ValueError, RuntimeError) as error:
                    failed += 1
                    error_log.write(json.dumps({"video_id": video.key, "video_path": str(path), "error": str(error)}, ensure_ascii=False) + "\n")
                    print(f"[{number}/{len(dataset.videos)}] FAILED {video.key}: {error}", flush=True)
                    continue
                if retry_ids:
                    replacements[video.key] = chunks
                else:
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
        if retry_ids and replacements:
            temporary_manifest = output.with_name(output.name + ".retry-tmp")
            with temporary_manifest.open("w", encoding="utf-8") as handle:
                for row in existing_rows:
                    if row["video_id"] not in replacements:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                for chunks in replacements.values():
                    for chunk in chunks:
                        handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
            temporary_manifest.replace(output)
            print(f"Replaced manifest records for {len(replacements)} successfully retried videos.")
        print(f"Finished: {processed} processed, {written} chunks written, {failed} failed. Manifest: {output}")
    elif args.command == "index":
        chunks = [Chunk(**row) for row in read_jsonl(args.chunks)]
        build_index(chunks, config)
        print(f"Indexed {len(chunks)} chunks at {config.retrieval['index_path']}")
    elif args.command == "retrieve":
        index = NumpyVectorIndex.load(Path(config.retrieval["index_path"]))
        retriever = HybridRetriever(
            TemporalRetriever(index),
            visual_factory=lambda: VisualRetriever(index, build_visual_embedder(config.vision)),
        )
        response = retriever.retrieve(args.question, args.video_id, args.top_k or int(config.retrieval.get("top_k", 5)))
        print(json.dumps({
            "route": response.route,
            "temporal_query": None if response.temporal_query is None else {
                "start_seconds": response.temporal_query.start_seconds, "end_seconds": response.temporal_query.end_seconds,
                "kind": response.temporal_query.kind,
            },
            "results": [
                {"chunk_id": item.chunk.id, "start_seconds": item.chunk.start_seconds, "end_seconds": item.chunk.end_seconds,
                 "score": item.score, "source": item.source}
                for item in response.results
            ],
        }, ensure_ascii=False, indent=2))
    else:
        examples = list(MedHorizonDataset(args.annotations).iter_questions())
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        existing = read_jsonl(output) if output.exists() else []
        completed_ids = {str(row["id"]) for row in existing if row.get("id") is not None and row.get("prediction") is not None}
        error_path = output.with_name(f"{output.stem}.errors.jsonl")
        run_path = output.with_name(f"{output.stem}.run.json")
        candidate_ids = {str(item.uid) for item in (examples[:args.limit] if args.limit is not None else examples)}
        started_at = datetime.now(timezone.utc)
        run_record = {
            "status": "running", "started_at": started_at.isoformat(), "finished_at": None,
            "config_path": str(Path(args.config).resolve()), "config": asdict(config),
            "git_commit": _git_commit(), "python": sys.version, "packages": _package_versions(),
            "model": config.llm.get("model"), "reader_provider": config.llm.get("provider"),
            "frames_per_chunk": config.llm.get("frames_per_chunk"),
            "top_k": args.top_k or config.retrieval.get("top_k"), "question_only": args.question_only,
            "annotations": args.annotations, "requested_limit": args.limit,
            "candidate_questions": len(candidate_ids), "completed_before": len(candidate_ids & completed_ids),
            "scheduled_questions": len(candidate_ids - completed_ids), "output": str(output),
            "error_log": str(error_path), "new_predictions": 0, "failed_questions": 0,
        }
        _write_json(run_path, run_record)
        started_perf = time.perf_counter()
        predictions: list[Prediction] = []
        failures = 0
        try:
            with output.open("a", encoding="utf-8") as prediction_log, error_path.open("a", encoding="utf-8") as error_log:
                def save_prediction(prediction: Prediction) -> None:
                    _append_jsonl(prediction_log, prediction.to_dict())

                def save_error(item, error: Exception) -> None:
                    nonlocal failures
                    failures += 1
                    _append_jsonl(error_log, {
                        "id": str(item.uid), "video_id": item.video_key, "error_type": type(error).__name__,
                        "error": str(error),
                    })

                predictions = run_medhorizon_qa(
                    examples, config, limit=args.limit, top_k=args.top_k, reader_frame_root=args.reader_frame_root,
                    question_only=args.question_only, completed_ids=completed_ids,
                    on_prediction=save_prediction, on_error=save_error,
                )
        except Exception as error:
            run_record["status"] = "failed"
            run_record["fatal_error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            run_record.update({
                "finished_at": datetime.now(timezone.utc).isoformat(), "runtime_seconds": round(time.perf_counter() - started_perf, 3),
                "new_predictions": len(predictions), "failed_questions": failures,
            })
            if run_record["status"] == "running":
                run_record["status"] = "completed" if not failures else "completed_with_failures"
            _write_json(run_path, run_record)
        print(
            f"Appended {len(predictions)} predictions to {output}; skipped {len(candidate_ids & completed_ids)} completed IDs. "
            f"Failures, if any: {error_path}; run record: {run_path}"
        )


if __name__ == "__main__":
    main()
