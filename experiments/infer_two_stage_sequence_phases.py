"""Infer open activity segments, then map them conservatively to phase labels."""

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
    TWO_STAGE_SEQUENCE_PHASE_VERSION,
    build_open_activity_segmentation_prompt,
    build_strict_phase_mapping_prompt,
    build_video_semantic_ontology,
    compact_observation_sequence,
    load_evidence_graph,
    load_observation_sequence,
    normalize_open_activity_response,
    normalize_strict_phase_mapping_response,
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
    parser.add_argument("--timeout-seconds", type=float, default=900)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--initial-retry-seconds", type=float, default=30)
    parser.add_argument("--max-retry-seconds", type=float, default=300)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir)
    if output.exists() and not args.resume:
        raise FileExistsError(f"Output directory already exists: {output}")
    if args.timeout_seconds <= 0 or args.max_retries < 0:
        raise ValueError("timeout must be positive and retries non-negative")

    observations = load_observation_sequence(args.descriptions)
    if str(observations[0]["video_id"]) != args.video_key:
        raise ValueError("Observation video does not match --video-key")
    graph = load_evidence_graph(args.graph)
    if graph.video_id != args.video_key:
        raise ValueError("Evidence graph video does not match --video-key")
    ontology = build_video_semantic_ontology(
        MedHorizonDataset(args.annotations).questions, args.video_key
    )
    compact = compact_observation_sequence(observations)

    client = _client(args.api_key_env, args.base_url, args.timeout_seconds)
    stage_metrics = []
    resumed_stage1 = args.resume and (output / "open_activity_segments.json").is_file()
    activity_path = output / "open_activity_segments.json"
    stage1_raw_path = output / "stage1_raw_response.json"
    if resumed_stage1:
        activity_payload = json.loads(activity_path.read_text(encoding="utf-8"))
        activities = activity_payload["segments"]
        print(f"Resumed {len(activities)} open activity segments", flush=True)
    elif args.resume and stage1_raw_path.is_file():
        raw = json.loads(stage1_raw_path.read_text(encoding="utf-8"))
        activities = normalize_open_activity_response(raw, compact)
        _write_json(
            activity_path,
            {
                "video_id": args.video_key,
                "version": TWO_STAGE_SEQUENCE_PHASE_VERSION,
                "segments": activities,
            },
        )
        resumed_stage1 = True
        print(
            f"Recovered {len(activities)} open activity segments from raw response",
            flush=True,
        )
    else:
        prompt = build_open_activity_segmentation_prompt(compact)
        response, attempts, elapsed = _request(
            client,
            model=args.model,
            prompt=prompt,
            max_tokens=8192,
            max_retries=args.max_retries,
            initial_retry_seconds=args.initial_retry_seconds,
            max_retry_seconds=args.max_retry_seconds,
        )
        raw = _parse_json_object(response.choices[0].message.content or "")
        output.mkdir(parents=True, exist_ok=True)
        _write_json(stage1_raw_path, raw)
        stage1_metric = _metrics(
            "open_activity_segmentation", prompt, response, attempts, elapsed
        )
        _write_json(output / "stage1_request_metadata.json", stage1_metric)
        activities = normalize_open_activity_response(raw, compact)
        _write_json(
            activity_path,
            {
                "video_id": args.video_key,
                "version": TWO_STAGE_SEQUENCE_PHASE_VERSION,
                "segments": activities,
            },
        )
        stage_metrics.append(stage1_metric)
        print(f"Stage 1: inferred {len(activities)} open activity segments", flush=True)

    final_path = output / "sequence_phase_segments.json"
    if args.resume and final_path.is_file():
        print(f"Two-stage result already complete: {output}", flush=True)
        return
    prompt = build_strict_phase_mapping_prompt(activities, ontology["phases"])
    response, attempts, elapsed = _request(
        client,
        model=args.model,
        prompt=prompt,
        max_tokens=4096,
        max_retries=args.max_retries,
        initial_retry_seconds=args.initial_retry_seconds,
        max_retry_seconds=args.max_retry_seconds,
    )
    raw_mapping = _parse_json_object(response.choices[0].message.content or "")
    segments = normalize_strict_phase_mapping_response(
        raw_mapping, activities, ontology["phases"]
    )
    source = f"openai_compatible:{args.model}:{TWO_STAGE_SEQUENCE_PHASE_VERSION}"
    event_rows = project_sequence_phases_to_events(graph, segments, source=source)
    stage_metrics.append(
        _metrics("strict_phase_mapping", prompt, response, attempts, elapsed)
    )

    _write_json(output / "stage2_raw_response.json", raw_mapping)
    _write_json(
        final_path,
        {
            "video_id": args.video_key,
            "version": TWO_STAGE_SEQUENCE_PHASE_VERSION,
            "fact_status": "medical_hypothesis",
            "candidate_aware_diagnostic": True,
            "segments": segments,
        },
    )
    with (output / "event_phase_hypotheses.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in event_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if resumed_stage1:
        stage1_meta = json.loads(
            (output / "stage1_request_metadata.json").read_text(encoding="utf-8")
        )
        stage_metrics.insert(0, stage1_meta)
    _write_json(
        output / "run_metadata.json",
        {
            "version": TWO_STAGE_SEQUENCE_PHASE_VERSION,
            "model": args.model,
            "base_url": args.base_url,
            "video_key": args.video_key,
            "descriptions": args.descriptions,
            "graph": args.graph,
            "observation_count": len(observations),
            "open_activity_segment_count": len(activities),
            "phase_segment_count": len(segments),
            "event_count": len(event_rows),
            "stage_metrics": stage_metrics,
            "ontology": ontology,
            "semantic_nodes_are_observed_facts": False,
            "candidate_aware_diagnostic": True,
            "answers_used": False,
        },
    )
    accepted = sum(bool(item["mapping_accepted"]) for item in segments)
    print(
        f"Stage 2: accepted {accepted}/{len(segments)} named phase mappings; "
        f"projected {len(event_rows)} events -> {output}",
        flush=True,
    )


def _client(api_key_env: str, base_url: str, timeout: float) -> Any:
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError(
            "Install model dependencies: pip install -e '.[models]'"
        ) from error
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"Set {api_key_env} before sequence phase inference")
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)


def _request(
    client: Any,
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    max_retries: int,
    initial_retry_seconds: float,
    max_retry_seconds: float,
) -> tuple[Any, int, float]:
    started = time.monotonic()
    for retry_number in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
            )
            return response, retry_number + 1, time.monotonic() - started
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


def _metrics(
    stage: str, prompt: str, response: Any, attempts: int, elapsed: float
) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    return {
        "stage": stage,
        "prompt_characters": len(prompt),
        "generation_attempts": attempts,
        "elapsed_seconds": round(elapsed, 3),
        "usage": usage.to_dict() if hasattr(usage, "to_dict") else None,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
