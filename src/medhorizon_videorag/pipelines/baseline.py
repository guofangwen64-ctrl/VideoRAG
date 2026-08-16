from __future__ import annotations

from pathlib import Path
from typing import Sequence

from medhorizon_videorag.core.config import ExperimentConfig
from medhorizon_videorag.core.schemas import Chunk, Prediction, QAExample
from medhorizon_videorag.features import build_visual_embedder
from medhorizon_videorag.features.utils import chunks_with_existing_frames
from medhorizon_videorag.datasets import MedHorizonQA
from medhorizon_videorag.generation import ExtractiveGenerator, build_video_reader
from medhorizon_videorag.ingestion import FineFrameExtractor
from medhorizon_videorag.retrieval import HybridRetriever, NumpyVectorIndex, TemporalRetriever, VisualRetriever


def build_index(chunks: Sequence[Chunk], config: ExperimentConfig) -> NumpyVectorIndex:
    encoder = build_visual_embedder(config.vision)
    chunks, skipped = chunks_with_existing_frames(chunks)
    if not chunks:
        raise ValueError("No chunks with decodable frames are available for indexing")
    index = NumpyVectorIndex()
    index.add(chunks, encoder.embed_chunks(chunks))
    index_path = Path(config.retrieval["index_path"])
    index.save(index_path)
    if skipped:
        from medhorizon_videorag.core.io import write_jsonl
        write_jsonl(index_path / "skipped_chunks.jsonl", skipped)
        print(f"Skipped {len(skipped)} chunks with no usable frame; audit: {index_path / 'skipped_chunks.jsonl'}")
    return index


def run_qa(examples: Sequence[QAExample], config: ExperimentConfig) -> list[Prediction]:
    index = NumpyVectorIndex.load(Path(config.retrieval["index_path"]))
    retriever = HybridRetriever(
        TemporalRetriever(index),
        visual_factory=lambda: VisualRetriever(index, build_visual_embedder(config.vision)),
    )
    if config.llm.get("provider", "extractive") != "extractive":
        raise NotImplementedError("Register an LLM provider adapter before use")
    generator = ExtractiveGenerator()
    top_k = int(config.retrieval.get("top_k", 5))
    predictions = []
    for item in examples:
        response = retriever.retrieve(item.question, item.video_id, top_k)
        predictions.append(Prediction(
            id=item.id, question=item.question,
            prediction=generator.answer(item.question, response.results),
            evidence=[
                {"chunk_id": hit.chunk.id, "score": hit.score, "start_seconds": hit.chunk.start_seconds,
                 "end_seconds": hit.chunk.end_seconds, "source": hit.source, "route": response.route}
                for hit in response.results
            ],
            reference=item.answer,
        ))
    return predictions


def run_medhorizon_qa(
    examples: Sequence[MedHorizonQA], config: ExperimentConfig, *, limit: int | None = None,
    top_k: int | None = None, reader_frame_root: str | Path | None = None,
) -> list[Prediction]:
    """Run retrieval, on-demand dense sampling, and a multiple-choice VLM reader."""
    index = NumpyVectorIndex.load(Path(config.retrieval["index_path"]))
    retriever = HybridRetriever(
        TemporalRetriever(index),
        visual_factory=lambda: VisualRetriever(index, build_visual_embedder(config.vision)),
    )
    selected = list(examples[:limit] if limit is not None else examples)
    effective_top_k = top_k or int(config.retrieval.get("top_k", 5))
    artifact_dir = Path(config.project.get("artifact_dir", "artifacts"))
    reader_config = config.llm
    extractor = FineFrameExtractor(
        reader_frame_root or artifact_dir / "reader_frames",
        int(reader_config.get("frames_per_chunk", 16)),
    )
    reader = build_video_reader(reader_config)
    predictions: list[Prediction] = []
    for number, item in enumerate(selected, start=1):
        response = retriever.retrieve(item.question, item.video_key, effective_top_k)
        evidence: list[dict] = []
        for hit in response.results:
            frames = extractor.extract(hit.chunk)
            evidence.append({
                "chunk_id": hit.chunk.id, "score": hit.score, "start_seconds": hit.chunk.start_seconds,
                "end_seconds": hit.chunk.end_seconds, "source": hit.source, "route": response.route,
                "reader_frame_paths": frames,
            })
        answer = reader.answer(item.question, item.options, evidence)
        predictions.append(Prediction(
            id=str(item.uid), question=item.question, prediction=answer.choice, reference=item.answer,
            evidence=evidence, metadata={
                "task_name": item.task_name, "task_id": item.task_id, "route": response.route,
                "reader_provider": reader_config.get("provider", "mock"), "rationale": answer.rationale,
            },
        ))
        print(f"[{number}/{len(selected)}] {item.uid}: {response.route} -> {answer.choice}", flush=True)
    return predictions
