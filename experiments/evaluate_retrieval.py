"""Evaluate a visual index against MedHorizon recovered temporal evidence."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medhorizon_videorag.core.config import load_config  # noqa: E402
from medhorizon_videorag.core.io import write_jsonl  # noqa: E402
from medhorizon_videorag.datasets import MedHorizonDataset, recover_evidence  # noqa: E402
from medhorizon_videorag.evaluation import evaluate_retrieval  # noqa: E402
from medhorizon_videorag.features import build_visual_embedder  # noqa: E402
from medhorizon_videorag.retrieval import HybridRetriever, NumpyVectorIndex, TemporalRetriever, VisualRetriever  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Encoder configuration used to build the index")
    parser.add_argument("--annotations", default="medhorizon_test.jsonl")
    parser.add_argument("--index", help="Override retrieval.index_path from the config")
    parser.add_argument("--name", default="visual_index", help="Experiment name recorded in the report")
    parser.add_argument("--retriever", choices=["visual", "hybrid"], default="visual", help="Use pure visual search or automatic temporal/visual routing")
    parser.add_argument("--scope", choices=["intra_video", "global"], default="intra_video", help="Restrict retrieval to the QA's source video, or search every video")
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
    evidence = [
        item for item in recover_evidence(MedHorizonDataset(args.annotations))
        if item.method in {"direct_range", "direct_point"}
    ]
    routes: Counter[str] = Counter()
    try:
        from tqdm.auto import tqdm
        iterator = tqdm(evidence, total=len(evidence), desc="Evaluating retrieval", unit="question")
    except ImportError:
        iterator = evidence
    if args.retriever == "hybrid":
        retriever = HybridRetriever(
            TemporalRetriever(index),
            visual_factory=lambda: VisualRetriever(index, build_visual_embedder(config.vision)),
        )
        results = []
        for item in iterator:
            response = retriever.retrieve(item.question or "", item.video_key, max(top_ks))
            routes[response.route] += 1
            results.append(response.results)
    else:
        encoder = build_visual_embedder(config.vision)
        questions = [item.question or "" for item in evidence]
        query_vectors = encoder.embed_text(questions)
        if query_vectors.shape[1] != index.vectors.shape[1]:
            raise ValueError(f"Embedding dimensions differ: query={query_vectors.shape[1]}, index={index.vectors.shape[1]}")
        results = []
        for item, vector in zip(iterator, query_vectors, strict=True):
            routes["visual"] += 1
            results.append(index.search(vector, max(top_ks), video_id=item.video_key if args.scope == "intra_video" else None))
    report, details = evaluate_retrieval(evidence, results, top_ks, args.iou_threshold)
    report.update({
        "experiment": args.name, "index_path": str(index_path), "encoder": config.vision,
        "indexed_chunks": len(index.chunks), "annotations": args.annotations, "scope": args.scope,
        "retriever": args.retriever, "route_counts": dict(sorted(routes.items())),
    })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_jsonl(args.details, details)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {output}\nDetails: {args.details}")


if __name__ == "__main__":
    main()
