"""Infer one global phase segmentation from ordered observation descriptions."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medhorizon_videorag.datasets import MedHorizonDataset
from medhorizon_videorag.graph_rag import (
    SEQUENCE_PHASE_VERSION,
    build_sequence_phase_prompt,
    build_video_semantic_ontology,
    compact_observation_sequence,
    load_evidence_graph,
    load_observation_sequence,
    normalize_sequence_phase_response,
    project_sequence_phases_to_events,
)
from medhorizon_videorag.graph_rag.qa_experiment import _parse_json_object


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptions", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--video-key", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="qwen3-vl-235b-a22b-instruct")
    parser.add_argument("--base-url", default="https://api.agicto.cn/v1")
    parser.add_argument("--api-key-env", default="AGICTO_API_KEY")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout-seconds", type=float, default=900)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--initial-retry-seconds", type=float, default=30)
    parser.add_argument("--max-retry-seconds", type=float, default=300)
    parser.add_argument(
        "--response-format-json",
        action="store_true",
        help="Request provider-native JSON mode; disabled for AGICTO by default",
    )
    args = parser.parse_args()

    if args.max_tokens < 1 or args.timeout_seconds <= 0:
        raise ValueError("max-tokens and timeout-seconds must be positive")
    if args.max_retries < 0:
        raise ValueError("max-retries must be non-negative")
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")

    observations = load_observation_sequence(args.descriptions)
    if str(observations[0]["video_id"]) != args.video_key:
        raise ValueError("Observation video does not match --video-key")
    graph = load_evidence_graph(args.graph)
    if graph.video_id != args.video_key:
        raise ValueError("Evidence graph video does not match --video-key")
    dataset = MedHorizonDataset(args.annotations)
    ontology = build_video_semantic_ontology(dataset.questions, args.video_key)
    compact = compact_observation_sequence(observations)
    prompt = build_sequence_phase_prompt(compact, ontology["phases"])

    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError(
            "Install model dependencies: pip install -e '.[models]'"
        ) from error
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Set {args.api_key_env} before sequence phase inference")
    client = OpenAI(
        api_key=api_key,
        base_url=args.base_url,
        timeout=args.timeout_seconds,
        max_retries=0,
    )
    started = time.monotonic()
    response, attempts = _request(
        client,
        model=args.model,
        prompt=prompt,
        max_tokens=args.max_tokens,
        response_format_json=args.response_format_json,
        max_retries=args.max_retries,
        initial_retry_seconds=args.initial_retry_seconds,
        max_retry_seconds=args.max_retry_seconds,
    )
    raw_text = response.choices[0].message.content or ""
    payload = _parse_json_object(raw_text)
    segments = normalize_sequence_phase_response(payload, compact, ontology["phases"])
    source = f"openai_compatible:{args.model}:{SEQUENCE_PHASE_VERSION}"
    event_rows = project_sequence_phases_to_events(graph, segments, source=source)

    output.mkdir(parents=True, exist_ok=False)
    _write_json(
        output / "sequence_phase_segments.json",
        {
            "video_id": args.video_key,
            "version": SEQUENCE_PHASE_VERSION,
            "fact_status": "medical_hypothesis",
            "candidate_aware_diagnostic": True,
            "segments": segments,
        },
    )
    _write_json(output / "raw_response.json", payload)
    with (output / "event_phase_hypotheses.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in event_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    usage = getattr(response, "usage", None)
    _write_json(
        output / "run_metadata.json",
        {
            "version": SEQUENCE_PHASE_VERSION,
            "model": args.model,
            "base_url": args.base_url,
            "video_key": args.video_key,
            "descriptions": args.descriptions,
            "graph": args.graph,
            "observation_count": len(observations),
            "event_count": len(event_rows),
            "phase_segment_count": len(segments),
            "prompt_characters": len(prompt),
            "generation_attempts": attempts,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "usage": usage.to_dict() if hasattr(usage, "to_dict") else None,
            "ontology": ontology,
            "semantic_nodes_are_observed_facts": False,
            "candidate_aware_diagnostic": True,
            "answers_used": False,
        },
    )
    print(
        f"Inferred {len(segments)} sequence phase segments for "
        f"{len(observations)} clips and projected {len(event_rows)} events -> {output}",
        flush=True,
    )


def _request(
    client: Any,
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    response_format_json: bool,
    max_retries: int,
    initial_retry_seconds: float,
    max_retry_seconds: float,
) -> tuple[Any, int]:
    request: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if response_format_json:
        request["response_format"] = {"type": "json_object"}
    for retry_number in range(max_retries + 1):
        try:
            return client.chat.completions.create(**request), retry_number + 1
        except Exception as error:
            status = getattr(error, "status_code", None)
            retryable = status in {408, 409, 429, 502, 503, 504} or type(
                error
            ).__name__ in {"APIConnectionError", "APITimeoutError", "TimeoutException"}
            if retry_number >= max_retries or not retryable:
                raise
            delay = min(initial_retry_seconds * (2**retry_number), max_retry_seconds)
            print(
                f"Retryable API error HTTP {status or 'network'}; retry "
                f"{retry_number + 1}/{max_retries} in {delay:g}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
