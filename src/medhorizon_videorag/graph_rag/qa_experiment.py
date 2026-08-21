from __future__ import annotations

import base64
import json
import math
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .schemas import VideoEvidenceGraph

GRAPH_QA_EXPERIMENT_VERSION = "qwen25-event-rerank-reader-v1"

_TIME_RANGE_PATTERNS = (
    re.compile(
        r"\s+from\s+\d{1,2}:\d{2}(?::\d{2})?\s+to\s+"
        r"\d{1,2}:\d{2}(?::\d{2})?",
        re.IGNORECASE,
    ),
    re.compile(
        r"\s+from\s+second\s+\d+(?:\.\d+)?\s+to\s+second\s+\d+(?:\.\d+)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"\s+(?:between|during)\s+\d{1,2}:\d{2}(?::\d{2})?\s+"
        r"(?:and|to|-)\s+\d{1,2}:\d{2}(?::\d{2})?",
        re.IGNORECASE,
    ),
)


def strip_explicit_time_range(question: str) -> str:
    """Remove an explicit range from the model-facing question, not from evidence."""
    cleaned = str(question)
    for pattern in _TIME_RANGE_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    cleaned = re.sub(r"\s+([;,?.!])", r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def build_event_observation_catalog(graph: VideoEvidenceGraph) -> list[dict[str, Any]]:
    """Create a compact, observation-only catalog for query-aware event reranking."""
    clips = {
        str(node.metadata.get("clip_id")): node
        for node in graph.nodes
        if node.node_type == "segment"
    }
    events = sorted(
        (node for node in graph.nodes if node.node_type == "temporal_event"),
        key=lambda node: (node.evidence[0].start_seconds, node.id),
    )
    catalog = []
    for event in events:
        representatives = event.metadata.get("representative_evidence", [])
        representative_ids = [str(item["clip_id"]) for item in representatives]
        if not representative_ids:
            representative_ids = [
                str(item) for item in event.metadata.get("supporting_clip_ids", [])[:1]
            ]
        observations = []
        for clip_id in representative_ids[:3]:
            clip = clips.get(clip_id)
            if clip is None:
                continue
            facts = clip.metadata.get("observation", {}).get("observed_facts", {})
            observations.append(
                {
                    "summary": clip.label,
                    "visible_instruments": list(facts.get("visible_instruments", [])),
                    "actions": list(facts.get("actions", [])),
                    "state_changes": list(facts.get("state_changes", [])),
                }
            )
        catalog.append(
            {
                "event_id": event.id,
                "predicates": list(event.metadata.get("predicates", [])),
                "concepts": list(event.metadata.get("concepts", [])),
                "observations": observations,
            }
        )
    return catalog


def select_event_frame_groups(
    graph: VideoEvidenceGraph,
    event_ids: Sequence[str],
    *,
    frames_per_event: int,
    prefer_onset: bool,
) -> list[dict[str, Any]]:
    """Resolve ranked events to traceable frames, preferring the first clip for onset QA."""
    if frames_per_event < 1:
        raise ValueError("frames_per_event must be at least 1")
    events = {
        node.id: node for node in graph.nodes if node.node_type == "temporal_event"
    }
    clips = {
        str(node.metadata.get("clip_id")): node
        for node in graph.nodes
        if node.node_type == "segment"
    }
    groups = []
    for event_id in event_ids:
        if event_id not in events:
            raise ValueError(f"Unknown reranked event ID: {event_id}")
        event = events[event_id]
        supporting = [
            str(item) for item in event.metadata.get("supporting_clip_ids", [])
        ]
        representatives = [
            str(item["clip_id"])
            for item in event.metadata.get("representative_evidence", [])
        ]
        candidates = supporting if prefer_onset else representatives + supporting
        clip_id = next((item for item in candidates if item in clips), None)
        if clip_id is None:
            raise ValueError(f"Event {event_id} has no resolvable clip evidence")
        clip = clips[clip_id]
        source_frames = [
            path for path in clip.evidence[0].frame_paths if Path(path).is_file()
        ]
        if not source_frames:
            raise ValueError(f"Clip {clip_id} has no existing frame paths")
        groups.append(
            {
                "event_id": event_id,
                "clip_id": clip_id,
                "start_seconds": clip.evidence[0].start_seconds,
                "end_seconds": clip.evidence[0].end_seconds,
                "reader_frame_paths": _uniform_sample(
                    source_frames, min(frames_per_event, len(source_frames))
                ),
                "selection": "event_onset_clip"
                if prefer_onset
                else "event_representative_clip",
            }
        )
    return groups


class OpenAICompatibleGraphQA:
    """Use one OpenAI-compatible VLM for event reranking and visual QA."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key_env: str = "OPENAI_API_KEY",
        max_image_pixels: int = 200704,
        timeout_seconds: float = 600.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError(
                "Install model dependencies: pip install -e '.[models]'"
            ) from error
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"Set {api_key_env} before running graph QA")
        if max_image_pixels < 1:
            raise ValueError("max_image_pixels must be positive")
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=0,
        )
        self.model = model
        self.max_image_pixels = max_image_pixels

    def rerank_events(
        self,
        question: str,
        choices: Sequence[str],
        catalog: Sequence[dict[str, Any]],
        *,
        top_events: int,
    ) -> tuple[list[str], str]:
        if top_events < 1:
            raise ValueError("top_events must be at least 1")
        known_ids = {str(item["event_id"]) for item in catalog}
        prompt = (
            "You are retrieving evidence from an observation-first graph of one long "
            "medical procedure video. The graph deliberately contains no surgical phase "
            "labels. Infer which temporal events are most relevant to the question from "
            "their visible instruments, generic entities, actions, and state changes. "
            "Do not answer the multiple-choice question yet. Select the best event IDs "
            f"in ranked order, with at most {top_events} IDs. Return only JSON with keys "
            "event_ids and rationale.\nQuestion: "
            + question
            + "\nChoices:\n"
            + "\n".join(choices)
            + "\nObservation event catalog:\n"
            + json.dumps(list(catalog), ensure_ascii=False, separators=(",", ":"))
        )
        payload = self._text_json(prompt, max_tokens=512)
        raw_ids = payload.get("event_ids")
        if not isinstance(raw_ids, list):
            raise TypeError(f"Event reranker returned invalid event_ids: {payload}")
        event_ids = []
        for item in raw_ids:
            event_id = str(item)
            if event_id in known_ids and event_id not in event_ids:
                event_ids.append(event_id)
            if len(event_ids) >= top_events:
                break
        if not event_ids:
            raise RuntimeError(f"Event reranker returned no valid event IDs: {payload}")
        return event_ids, str(payload.get("rationale", ""))

    def answer(
        self,
        question: str,
        choices: Sequence[str],
        evidence_groups: Sequence[dict[str, Any]],
    ) -> tuple[str, str]:
        labels = _choice_labels(choices)
        prompt = (
            "You are answering a medical-video multiple-choice question from retrieved "
            "visual evidence. Compare every option with the visible frames. Retrieved "
            "events can be imperfect; do not assume that an instrument or phase is present "
            "unless supported by the frames. Return only JSON with keys choice and "
            f"rationale. choice must be one of {labels}.\nQuestion: {question}\n"
            "Choices:\n" + "\n".join(choices)
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for number, group in enumerate(evidence_groups, start=1):
            content.append(
                {
                    "type": "text",
                    "text": f"Candidate evidence group {number} ({group['clip_id']}):",
                }
            )
            for frame_path in group["reader_frame_paths"]:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,"
                            + _encode_resized_jpeg(frame_path, self.max_image_pixels)
                        },
                    }
                )
        payload = self._vision_json(content, max_tokens=256)
        choice = str(payload.get("choice", "")).strip().upper()
        if choice not in labels:
            raise RuntimeError(
                f"Reader returned invalid choice {choice!r}; expected one of {labels}"
            )
        return choice, str(payload.get("rationale", ""))

    def _text_json(self, prompt: str, *, max_tokens: int) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_tokens,
        )
        return _parse_json_object(response.choices[0].message.content or "")

    def _vision_json(
        self, content: list[dict[str, Any]], *, max_tokens: int
    ) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=0,
            max_tokens=max_tokens,
        )
        return _parse_json_object(response.choices[0].message.content or "")


def _choice_labels(choices: Sequence[str]) -> list[str]:
    labels = []
    for index, choice in enumerate(choices):
        match = re.match(r"\s*([A-Za-z0-9]+)[.):：]\s*", choice)
        labels.append(match.group(1).upper() if match else chr(ord("A") + index))
    if not labels:
        raise ValueError("Graph QA requires multiple-choice options")
    return labels


def _uniform_sample(items: Sequence[str], count: int) -> list[str]:
    if count < 1 or not items:
        raise ValueError("Cannot sample an empty frame sequence")
    if count == 1:
        return [str(items[len(items) // 2])]
    positions = [
        round(index * (len(items) - 1) / (count - 1)) for index in range(count)
    ]
    return [str(items[position]) for position in positions]


def _encode_resized_jpeg(path: str | Path, max_pixels: int) -> str:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "Install video dependencies: pip install -e '.[video]'"
        ) from error
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot decode frame: {path}")
    height, width = image.shape[:2]
    if height * width > max_pixels:
        scale = math.sqrt(max_pixels / (height * width))
        image = cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError(f"Cannot encode frame: {path}")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.replace("```json", "").replace("```", "").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError(f"Model response did not contain a JSON object: {text}")
    try:
        payload = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Model response contained invalid JSON: {text}") from error
    if not isinstance(payload, dict):
        raise TypeError("Model JSON response must be an object")
    return payload
