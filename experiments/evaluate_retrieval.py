"""Evaluate a visual index against MedHorizon recovered temporal evidence."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medhorizon_videorag.core.config import load_config  # noqa: E402
from medhorizon_videorag.core.io import write_jsonl  # noqa: E402
from medhorizon_videorag.datasets import MedHorizonDataset, recover_evidence  # noqa: E402
from medhorizon_videorag.evaluation import evaluate_retrieval  # noqa: E402
from medhorizon_videorag.features import build_visual_embedder  # noqa: E402
from medhorizon_videorag.retrieval import NumpyVectorIndex  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Encoder configuration used to build the index")
    parser.add_argument("--annotations", default="medhorizon_test.jsonl")
    parser.add_argument("--index", help="Override retrieval.index_path from the config")
    parser.add_argument("--name", default="visual_index", help="Experiment name recorded in the report")
    parser.add_argument("--top-k", default="1,4,8", help="Comma-separated retrieval cutoffs")
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--output", default="artifacts/retrieval_report.json")
    parser.add_argument("--details", default="artifacts/retrieval_details.jsonl")
    args = parser.parse_args()

    top_ks = sorted({int(value) for value in args.top_k.split(",") if value.strip()})
    if not top_ks or min(top_ks) <= 0:
        raise ValueError("--top-k must contain positive integers")
    config = load_config(args.config)
    index_path = Path(args.index or config.retrieval["index_path"])
    index = NumpyVectorIndex.load(index_path)
    encoder = build_visual_embedder(config.vision)
    evidence = [
        item for item in recover_evidence(MedHorizonDataset(args.annotations))
        if item.method in {"direct_range", "direct_point"}
    ]
    questions = [item.question or "" for item in evidence]
    query_vectors = encoder.embed_text(questions)
    if query_vectors.shape[1] != index.vectors.shape[1]:
        raise ValueError(f"Embedding dimensions differ: query={query_vectors.shape[1]}, index={index.vectors.shape[1]}")
    try:
        from tqdm.auto import tqdm
        iterator = tqdm(zip(evidence, query_vectors, strict=True), total=len(evidence), desc="Evaluating retrieval", unit="question")
    except ImportError:
        iterator = zip(evidence, query_vectors, strict=True)
    results = [index.search(vector, max(top_ks)) for _, vector in iterator]
    report, details = evaluate_retrieval(evidence, results, top_ks, args.iou_threshold)
    report.update({
        "experiment": args.name, "index_path": str(index_path), "encoder": config.vision,
        "indexed_chunks": len(index.chunks), "annotations": args.annotations,
    })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_jsonl(args.details, details)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {output}\nDetails: {args.details}")


if __name__ == "__main__":
    main()
