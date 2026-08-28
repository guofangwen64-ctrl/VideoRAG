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

_CATALOG_STOP_CONCEPTS = frozenset(
    {
        "clear_fluid",
        "generic_instrument",
        "generic_material",
        "generic_object",
        "generic_structure",
        "red_fluid",
        "surface_region",
        "tissue",
        "tissue_region",
    }
)

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
        for clip_id in representative_ids[:1]:
            clip = clips.get(clip_id)
            if clip is None:
                continue
            facts = clip.metadata.get("observation", {}).get("observed_facts", {})
            observations.append(
                {
                    "summary": clip.label,
                    "visible_instruments": list(
                        facts.get("visible_instruments", [])[:3]
                    ),
                }
            )
        catalog.append(
            {
                "event_id": event.id,
                "predicates": list(event.metadata.get("predicates", [])),
                "concepts": [
                    item
                    for item in event.metadata.get("concepts", [])
                    if item not in _CATALOG_STOP_CONCEPTS
                ],
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
            "their visible instruments, generic entities, actions, and summaries. "
            "Do not answer the multiple-choice question yet. Select the best event IDs "
            f"in ranked order, with at most {top_events} IDs. Copy one or more complete "
            "event IDs exactly from the catalog without changing any digits. Return "
            'one JSON object with keys "event_ids" and "rationale"; keep rationale '
            "under 10 words. Do not include Markdown or code fences.\nQuestion: "
            + question
            + "\nChoices:\n"
            + "\n".join(choices)
            + "\nObservation event catalog:\n"
            + json.dumps(list(catalog), ensure_ascii=False, separators=(",", ":"))
        )
        text = self._text_response(prompt, max_tokens=128)
        try:
            payload = _parse_json_object(text)
            raw_ids = payload.get("event_ids")
        except RuntimeError:
            raw_ids = re.findall(r"event:[A-Za-z0-9_-]+:\d{5}", text)
        if isinstance(raw_ids, str):
            raw_ids = [raw_ids]
        if not isinstance(raw_ids, list):
            raise TypeError(f"Event reranker returned invalid event_ids: {text}")
        event_ids = []
        for item in raw_ids:
            event_id = str(item)
            if event_id not in known_ids:
                match = re.fullmatch(r"(event:[A-Za-z0-9_-]+:)(\d+)", event_id)
                if match:
                    normalized_id = f"{match.group(1)}{int(match.group(2)):05d}"
                    if normalized_id in known_ids:
                        event_id = normalized_id
            if event_id in known_ids and event_id not in event_ids:
                event_ids.append(event_id)
            if len(event_ids) >= top_events:
                break
        if not event_ids:
            raise RuntimeError(f"Event reranker returned no valid event IDs: {text}")
        return event_ids, ""

    def verify_phase_activity_candidates(
        self,
        phase_label: str,
        candidates: Sequence[dict[str, Any]],
        evidence_groups: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        """Visually select one query-conditioned phase candidate without QA options."""
        known_ids = {str(item["segment_id"]) for item in candidates}
        if not known_ids:
            raise ValueError("Phase verification requires activity candidates")
        catalog = [
            {
                "segment_id": item["segment_id"],
                "activity_label": item.get("activity_label"),
                "observed_pattern": item.get("observed_pattern"),
                "start_seconds": item.get("start_seconds"),
                "end_seconds": item.get("end_seconds"),
                "retrieval_score": item.get("retrieval_score"),
                "activity_cue_hits": item.get("activity_cue_hits", []),
            }
            for item in candidates
        ]
        prompt = (
            "Verify which retrieved activity segment best matches the target surgical "
            "phase using its observation summary and representative visual frames. "
            "Do not identify the instrument for a QA answer and do not use answer "
            "options. Local frames may be ambiguous, so consider the activity sequence, "
            "but do not claim an anatomical structure that is not visually supported. "
            "Return only JSON with selected_segment_id, confidence, and rationale. "
            "selected_segment_id must be one candidate ID, or null if none is plausible. "
            "confidence must be high, medium, or low.\nTarget phase: "
            + phase_label
            + "\nCandidates:\n"
            + json.dumps(catalog, ensure_ascii=False, separators=(",", ":"))
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for number, group in enumerate(evidence_groups, start=1):
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"Phase candidate evidence {number}: "
                        f"segment={group['segment_id']}, clip={group['clip_id']}"
                    ),
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
        selected = payload.get("selected_segment_id")
        selected_id = str(selected) if selected is not None else None
        if selected_id not in known_ids:
            selected_id = None
        confidence = str(payload.get("confidence", "low")).strip().lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "low"
        return {
            "selected_segment_id": selected_id,
            "confidence": confidence,
            "rationale": str(payload.get("rationale", "")),
        }

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

    def answer_phase_instrument(
        self,
        question: str,
        choices: Sequence[str],
        reader_input: dict[str, Any],
    ) -> tuple[str, str, list[str]]:
        """Rerank unknown-identity appearance tracks with their grounded frames."""
        labels = _choice_labels(choices)
        candidates = list(reader_input.get("candidate_tracks", []))
        known_track_ids = {str(item["track_id"]) for item in candidates}
        if not candidates:
            raise ValueError("Phase-instrument Reader requires candidate tracks")
        catalog = [
            {
                "track_id": item["track_id"],
                "graph_rank": item["graph_rank"],
                "observation_label": item["label"],
                "appearance_family": item["appearance_family"],
                "appearance_signature": item["appearance_signature"],
                "surface_forms": item["surface_forms"],
                "action_roles": item["action_roles"],
                "evidence_clip_ids": item["reader_clip_ids"],
                "option_matches": item.get("option_matches", []),
                "canonical_instrument": "unknown",
            }
            for item in candidates
        ]
        prompt = (
            "You are answering which instrument is visible at the onset of a surgical "
            "phase. The graph retrieved observation-level appearance tracks and frames "
            "near the phase boundary. Track labels are not medical instrument "
            "identifications, and tracks do not assert the same physical object over "
            "time. Independently compare the visual shape, jaws, shaft, tip, markings, "
            "and interaction pattern against every answer option. Do not choose an "
            "option merely because its wording resembles an observation label. "
            "Do not confuse a needle-like object with a needle holder: when an option "
            "names a holder or forceps, inspect the grasping track that manipulates the "
            "needle-like object. When options differ only by small versus regular size, "
            "compare relative jaw and shaft scale in the frames rather than defaulting "
            "to the unsized option. Option-match hints, when present, are lightweight "
            "prototype cues from option text; treat negative_hits as warnings and "
            "verify against the images before deciding. "
            "Graph rank is only a retrieval prior. Return only JSON with keys choice, "
            "selected_track_ids, and rationale. choice must be one of "
            f"{labels}; selected_track_ids must come from the catalog.\n"
            f"Target phase: {reader_input['phase_label']}\n"
            f"Question: {question}\nChoices:\n"
            + "\n".join(choices)
            + "\nAppearance-track catalog:\n"
            + json.dumps(catalog, ensure_ascii=False, separators=(",", ":"))
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for number, group in enumerate(reader_input["evidence_groups"], start=1):
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"Evidence clip {number}: {group['clip_id']}; "
                        f"tracks={group['track_ids']}"
                    ),
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
        payload = self._vision_json(content, max_tokens=384)
        choice = str(payload.get("choice", "")).strip().upper()
        if choice not in labels:
            raise RuntimeError(
                f"Reader returned invalid choice {choice!r}; expected one of {labels}"
            )
        raw_track_ids = payload.get("selected_track_ids", [])
        if isinstance(raw_track_ids, str):
            raw_track_ids = [raw_track_ids]
        if not isinstance(raw_track_ids, list):
            raw_track_ids = []
        selected_track_ids = []
        for track_id in raw_track_ids:
            value = str(track_id)
            if value in known_track_ids and value not in selected_track_ids:
                selected_track_ids.append(value)
        return choice, str(payload.get("rationale", "")), selected_track_ids

    def _text_response(self, prompt: str, *, max_tokens: int) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or ""

    def _vision_json(
        self, content: list[dict[str, Any]], *, max_tokens: int
    ) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
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
