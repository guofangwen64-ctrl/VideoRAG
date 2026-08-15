from __future__ import annotations

from pathlib import Path
from typing import Sequence

from medhorizon_videorag.core.config import ExperimentConfig
from medhorizon_videorag.core.schemas import Chunk, Prediction, QAExample
from medhorizon_videorag.features import build_visual_embedder
from medhorizon_videorag.features.utils import chunks_with_existing_frames
from medhorizon_videorag.generation import ExtractiveGenerator
from medhorizon_videorag.retrieval import NumpyVectorIndex, VideoRetriever


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
    retriever = VideoRetriever(index, build_visual_embedder(config.vision))
    if config.llm.get("provider", "extractive") != "extractive":
        raise NotImplementedError("Register an LLM provider adapter before use")
    generator = ExtractiveGenerator()
    top_k = int(config.retrieval.get("top_k", 5))
    return [Prediction(
        id=item.id, question=item.question,
        prediction=generator.answer(item.question, evidence := retriever.retrieve(item.question, top_k)),
        evidence=[{"chunk_id": hit.chunk.id, "score": hit.score, "start_seconds": hit.chunk.start_seconds, "end_seconds": hit.chunk.end_seconds} for hit in evidence],
        reference=item.answer,
    ) for item in examples]
