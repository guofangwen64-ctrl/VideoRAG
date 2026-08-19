from __future__ import annotations

import base64
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .schemas import VgentClipPlan

DESCRIPTION_PROMPT_VERSION = "medical_clip_v1"


def select_even_full_clips(
    clips: Sequence[VgentClipPlan],
    count: int,
    *,
    frames_per_request: int = 64,
) -> list[VgentClipPlan]:
    """Select temporal coverage without admitting partial or incomplete clips."""
    if count <= 0 or frames_per_request <= 0:
        raise ValueError("count and frames_per_request must be positive")
    eligible = [
        clip
        for clip in clips
        if clip.sampled_frame_count == frames_per_request
        and len(clip.frame_paths) == frames_per_request
        and all(Path(path).is_file() for path in clip.frame_paths)
    ]
    if len(eligible) < count:
        raise ValueError(
            f"Need {count} complete {frames_per_request}-frame clips; "
            f"only {len(eligible)} are available"
        )
    if count == 1:
        return [eligible[len(eligible) // 2]]
    positions = [
        round(index * (len(eligible) - 1) / (count - 1)) for index in range(count)
    ]
    return [eligible[position] for position in positions]


class OpenAICompatibleClipDescriber:
    """Generate structured medical clip descriptions through a local vLLM endpoint."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key_env: str = "OPENAI_API_KEY",
        max_tokens: int = 1024,
        timeout_seconds: float = 600,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError(
                "Install model dependencies: pip install -e '.[models]'"
            ) from error
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"Set {api_key_env} before describing clips")
        self.client = OpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout_seconds
        )
        self.model = model
        self.max_tokens = max_tokens

    def describe(
        self, clip: VgentClipPlan, *, frames_per_request: int = 64
    ) -> dict[str, Any]:
        if len(clip.frame_paths) != frames_per_request:
            raise ValueError(
                f"Clip {clip.id} has {len(clip.frame_paths)} frames; "
                f"expected exactly {frames_per_request}"
            )
        prompt = (
            "You are analyzing one continuous segment of a medical or surgical video. "
            "The images are chronological and uniformly sampled at one frame per second. "
            "Describe only directly visible evidence. Do not infer a diagnosis, anatomy, "
            "instrument, action, or procedural phase when it is not visually supported. "
            "Use concise English terms and return exactly one JSON object with this schema: "
            '{"summary":"", "procedure_phase":"", "anatomy":[], "instruments":[], '
            '"entities":[{"name":"", "description":""}], '
            '"actions":[{"subject":"", "action":"", "target":""}], '
            '"findings":[], "state_changes":[], "uncertainties":[], '
            '"visual_evidence":[]}. '
            f"This segment covers {clip.start_seconds:.1f} to {clip.end_seconds:.1f} seconds."
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for frame_path in clip.frame_paths:
            encoded = base64.b64encode(Path(frame_path).read_bytes()).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                }
            )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=0,
            max_tokens=self.max_tokens,
        )
        text = response.choices[0].message.content or ""
        payload = _parse_json_object(text)
        required = {
            "summary",
            "procedure_phase",
            "anatomy",
            "instruments",
            "entities",
            "actions",
            "findings",
            "state_changes",
            "uncertainties",
            "visual_evidence",
        }
        missing = sorted(required - payload.keys())
        if missing:
            raise RuntimeError(f"Description is missing required keys: {missing}")
        return payload


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.replace("```json", "").replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError(f"Description did not contain a JSON object: {text}")
    try:
        payload = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Description returned invalid JSON: {text}") from error
    if not isinstance(payload, dict):
        raise TypeError("Description JSON must be an object")
    return payload
