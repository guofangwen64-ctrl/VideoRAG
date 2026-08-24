"""Infer candidate-aware phase and instrument hypotheses for temporal events."""

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
    build_event_observation_catalog,
    build_video_semantic_ontology,
    load_evidence_graph,
    select_event_frame_groups,
)
from medhorizon_videorag.graph_rag.qa_experiment import (
    _encode_resized_jpeg,
    _parse_json_object,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--video-key", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--frames-per-event", type=int, default=8)
    parser.add_argument("--max-image-pixels", type=int, default=200704)
    parser.add_argument("--event-ids")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError(
            "Install model dependencies: pip install -e '.[models]'"
        ) from error
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Set {args.api_key_env} before semantic inference")
    if args.frames_per_event < 1:
        raise ValueError("frames-per-event must be positive")

    graph = load_evidence_graph(args.graph)
    if graph.video_id != args.video_key:
        raise ValueError(
            f"Graph video {graph.video_id} does not match {args.video_key}"
        )
    dataset = MedHorizonDataset(args.annotations)
    ontology = build_video_semantic_ontology(dataset.questions, args.video_key)
    catalog = build_event_observation_catalog(graph)
    catalog_by_id = {str(item["event_id"]): item for item in catalog}
    requested = (
        [item.strip() for item in args.event_ids.split(",") if item.strip()]
        if args.event_ids
        else [str(item["event_id"]) for item in catalog]
    )
    unknown = sorted(set(requested) - catalog_by_id.keys())
    if unknown:
        raise ValueError(f"Unknown requested event IDs: {unknown}")

    output = Path(args.output)
    completed: set[str] = set()
    if output.exists():
        if not args.resume:
            raise FileExistsError(f"Output already exists: {output}")
        completed = {
            str(json.loads(line)["event_id"])
            for line in output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    else:
        output.parent.mkdir(parents=True, exist_ok=True)

    metadata_path = (
        Path(args.metadata_output)
        if args.metadata_output
        else output.with_suffix(".metadata.json")
    )
    metadata = {
        "model": args.model,
        "base_url": args.base_url,
        "video_key": args.video_key,
        "graph": str(args.graph),
        "frames_per_event": args.frames_per_event,
        "max_image_pixels": args.max_image_pixels,
        "ontology": ontology,
        "semantic_nodes_are_observed_facts": False,
        "candidate_aware_diagnostic": True,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=600, max_retries=0)
    ordered_ids = [str(item["event_id"]) for item in catalog]
    for number, event_id in enumerate(requested, start=1):
        if event_id in completed:
            print(f"[{number}/{len(requested)}] skip {event_id}", flush=True)
            continue
        position = ordered_ids.index(event_id)
        context = catalog[max(0, position - 1) : position + 2]
        group = select_event_frame_groups(
            graph,
            [event_id],
            frames_per_event=args.frames_per_event,
            prefer_onset=False,
        )[0]
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": _prompt(event_id, context, ontology),
            }
        ]
        for frame_path in group["reader_frame_paths"]:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,"
                        + _encode_resized_jpeg(frame_path, args.max_image_pixels)
                    },
                }
            )
        started = time.monotonic()
        payload = _request_json(client, args.model, content)
        row = _normalize_response(payload, event_id, ontology)
        row.update(
            {
                "source": f"openai_compatible:{args.model}",
                "evidence_group": {
                    key: value
                    for key, value in group.items()
                    if key != "reader_frame_paths"
                },
                "reader_frame_paths": group["reader_frame_paths"],
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "candidate_aware_diagnostic": True,
            }
        )
        _append_jsonl(output, row)
        print(
            f"[{number}/{len(requested)}] {event_id}: "
            f"phase={row['phase_hypothesis']['label']} "
            f"instruments={[item['label'] for item in row['instrument_hypotheses']]}",
            flush=True,
        )


def _prompt(
    event_id: str, context: list[dict[str, Any]], ontology: dict[str, Any]
) -> str:
    return (
        "You are adding an explicitly uncertain semantic hypothesis layer above an "
        "observation-first graph of one medical procedure video. The supplied images "
        f"are representative frames for {event_id}. Previous/current/next observation "
        "summaries provide local sequence context only. Select a phase only when the "
        "visual activity and sequence context support it. Select instruments only when "
        "their distinctive visible form supports the exact type. Do not infer an "
        "instrument merely because it is expected in a phase. Use only exact labels from "
        "the candidate lists or use phase label 'unknown' and an empty instrument list. "
        "Return JSON with phase_hypothesis={label,confidence,basis} and "
        "instrument_hypotheses=[{label,confidence,basis}]. Confidence is high, medium, "
        "or low. These outputs are medical hypotheses, never direct observations.\n"
        "Candidate phases:\n- "
        + "\n- ".join(ontology["phases"])
        + "\nCandidate instruments:\n- "
        + "\n- ".join(ontology["instruments"])
        + "\nLocal observation context:\n"
        + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    )


def _request_json(
    client: Any, model: str, content: list[dict[str, Any]]
) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0,
        max_tokens=384,
        response_format={"type": "json_object"},
    )
    return _parse_json_object(response.choices[0].message.content or "")


def _normalize_response(
    payload: dict[str, Any], event_id: str, ontology: dict[str, Any]
) -> dict[str, Any]:
    phase_map = {_canonical(item): item for item in ontology["phases"]}
    instrument_map = {_canonical(item): item for item in ontology["instruments"]}
    raw_phase = payload.get("phase_hypothesis", {})
    if not isinstance(raw_phase, dict):
        raw_phase = {}
    phase_label = phase_map.get(_canonical(raw_phase.get("label", "")), "unknown")
    phase = {
        "label": phase_label,
        "confidence": _confidence_label(raw_phase.get("confidence")),
        "basis": str(raw_phase.get("basis", "")).strip(),
    }
    raw_instruments = payload.get("instrument_hypotheses", [])
    if isinstance(raw_instruments, dict):
        raw_instruments = [raw_instruments]
    instruments = []
    seen = set()
    for item in raw_instruments if isinstance(raw_instruments, list) else []:
        if not isinstance(item, dict):
            continue
        label = instrument_map.get(_canonical(item.get("label", "")))
        if not label or label in seen:
            continue
        seen.add(label)
        instruments.append(
            {
                "label": label,
                "confidence": _confidence_label(item.get("confidence")),
                "basis": str(item.get("basis", "")).strip(),
            }
        )
    return {
        "event_id": event_id,
        "phase_hypothesis": phase,
        "instrument_hypotheses": instruments,
    }


def _confidence_label(value: Any) -> str:
    lowered = str(value or "low").strip().lower()
    return lowered if lowered in {"high", "medium", "low"} else "low"


def _canonical(value: Any) -> str:
    import re

    return " ".join(re.findall(r"[a-z0-9]+", str(value).lower()))


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


if __name__ == "__main__":
    main()
