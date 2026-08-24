from __future__ import annotations

import base64
import json
import math
import os
import re
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .schemas import VgentClipPlan

DESCRIPTION_PROMPT_VERSION = "medical_clip_observation_first_v10"

SUMMARY_FORBIDDEN_TERMS = (
    "possibly",
    "likely",
    "may",
    "blood",
    "bleeding",
    "surgical",
    "surgical field",
    "surgical work",
    "procedure",
    "suture",
    "suturing",
    "irrigation",
    "blood vessel",
    "inflamed",
    "dissection",
    "resection",
    "repair",
    "ligation",
)

OBSERVATION_FIRST_SYSTEM_PROMPT = """You are a literal visual transcription system.
The summary and observed_facts must contain only directly visible appearance,
motion, and spatial relationships. In the summary, never use possibly, likely,
may, blood, bleeding, surgical field, surgical work, procedure, suture,
surgical, suturing, irrigation, blood vessel, inflamed, dissection, resection, repair,
or ligation. Use generic appearance-first terms such as red fluid, clear fluid,
thread-like material, tubular structure, reddish tissue, and instrument.
Do not infer a medical action from generic instrument-tissue interaction.
medical_inferences must be [] unless distinctive visual evidence supports an
interpretation. Never use video titles, metadata, or neighboring-clip context.
Return only the requested JSON object. Before responding, silently check the
summary for every forbidden word and rewrite it if any is present.
The summary must be exactly one short sentence of at most 30 words in the form
of visible object, visible action, visible target, and visible appearance.
Do not describe the scene type, medical purpose, procedural context, skill,
precision, or delicacy. medical_inferences should usually be []."""

DESCRIPTION_PROMPT = """You are analyzing a short clip from a long medical procedure video.

Your task is to describe the clip based strictly on the provided visual frames.

Apply the observation-first rules before writing any JSON:
- The summary is a literal visual account, not a medical interpretation.
- Never write "possibly", "likely", or "may" in the summary.
- Do not use "blood", "bleeding", "surgical field", "surgical work",
  "procedure", "suture", "suturing", "irrigation", "blood vessel", or
  "inflamed" in the summary; describe visible color, shape, material, motion,
  and spatial relationships instead.
- If there is no distinctive visual evidence for a medical interpretation,
  return an empty medical_inferences array.
- Write the summary as exactly one sentence of at most 30 words. Mention only
  visible objects, actions, targets, colors, shapes, materials, and motion.
  Do not describe the scene type, purpose, precision, or procedural context.

IMPORTANT:
Separate direct visual observations from medical interpretations.
Do not present an inferred anatomical structure, surgical phase, diagnosis,
or procedure as a directly observed fact.

If something cannot be confidently identified, explicitly mark it as uncertain
instead of guessing.

Return ONLY valid JSON using the following schema:

{
  "summary": "...",

  "observed_facts": {
    "visible_anatomy": [],
    "visible_instruments": [],
    "visible_objects": [],
    "actions": [
      {
        "subject": "...",
        "action": "...",
        "target": "..."
      }
    ],
    "state_changes": [],
    "visual_evidence": []
  },

  "medical_inferences": [
    {
      "inference": "...",
      "basis": "...",
      "confidence": "high | medium | low"
    }
  ],

  "uncertainties": [
    {
      "item": "...",
      "reason": "..."
    }
  ]
}

Rules:

1. OBSERVED FACTS
Include only information directly supported by visible evidence.
Use generic terms when exact identity is unclear, such as:
- tissue
- tubular structure
- surgical instrument
- needle-like instrument
- thread-like material

Do NOT identify a structure as "meniscus", "heart", "vein", "tumor", etc.
unless distinctive visual evidence clearly supports that identification.

2. MEDICAL INFERENCES
Put interpretations here rather than in observed_facts.
Every inference must include its visual basis and confidence.
Do not add an inference merely to fill the example schema. Return [] when the
visible evidence is generic or insufficient.

3. UNCERTAINTIES
Explicitly record structures, instruments, actions, or phases that cannot
be reliably identified.
Do not guess simply to fill a field.

4. PROCEDURE PHASE
Do not infer a specific surgical phase from ambiguous local visual evidence.
If a phase is suggested, include it only under medical_inferences.

5. SUMMARY
The summary MUST contain only directly visible activity.
Avoid unsupported anatomical or procedural claims.

OBSERVATION-FIRST RULES:

1. The summary MUST contain only directly visible information.
   Never use "possibly", "likely", "may", or inferred medical terminology
   in the summary.

2. observed_facts MUST contain only visually verifiable facts.

3. Describe appearance before interpretation:
   - "red fluid" instead of "active bleeding"
   - "clear fluid" instead of "irrigation fluid"
   - "thread-like material" instead of "suture"
   - "tubular structure" instead of "blood vessel"
   - "reddish tissue" instead of "inflamed tissue"

4. Do not infer suturing, dissection, resection, repair, ligation,
   or other surgical actions from generic tissue manipulation alone.

5. A medical inference requires distinctive visual evidence.
   Generic evidence such as "instrument near tissue" is insufficient.

6. If evidence is insufficient, omit the inference rather than guessing.

7. Confidence:
   high = multiple distinctive visual cues
   medium = at least one distinctive cue, alternatives remain
   low = weak or contextual evidence only

8. Do not use external context, video title, dataset metadata,
   expected procedure type, or assumptions about neighboring clips."""


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


def select_full_clips_by_index(
    clips: Sequence[VgentClipPlan],
    indices: Sequence[int],
    *,
    frames_per_request: int = 64,
) -> list[VgentClipPlan]:
    """Select exact complete clips while preserving the requested order."""
    if not indices or frames_per_request <= 0:
        raise ValueError(
            "indices must not be empty and frames_per_request must be positive"
        )
    if len(set(indices)) != len(indices):
        raise ValueError("clip indices must be unique")
    by_index = {clip.clip_index: clip for clip in clips}
    selected = []
    for index in indices:
        clip = by_index.get(index)
        if clip is None:
            raise ValueError(f"Clip index {index} is not present in the manifest")
        if (
            clip.sampled_frame_count != frames_per_request
            or len(clip.frame_paths) != frames_per_request
            or any(not Path(path).is_file() for path in clip.frame_paths)
        ):
            raise ValueError(
                f"Clip index {index} is not a complete {frames_per_request}-frame clip"
            )
        selected.append(clip)
    return selected


def prepare_request_frame_paths(
    clip: VgentClipPlan,
    *,
    frames_per_request: int = 64,
) -> list[str]:
    """Return exact request length, padding only a valid partial tail clip."""
    source_paths = list(clip.frame_paths)
    if len(source_paths) != clip.sampled_frame_count:
        raise ValueError(
            f"Clip {clip.id} has {len(source_paths)} cached frames; "
            f"expected {clip.sampled_frame_count}"
        )
    if any(not Path(path).is_file() for path in source_paths):
        raise ValueError(f"Clip {clip.id} has missing cached frames")
    if len(source_paths) == frames_per_request:
        return source_paths
    if clip.is_partial and source_paths and len(source_paths) < frames_per_request:
        return source_paths + [source_paths[-1]] * (
            frames_per_request - len(source_paths)
        )
    raise ValueError(
        f"Clip {clip.id} has {len(source_paths)} frames; "
        f"expected exactly {frames_per_request}"
    )


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
        max_image_pixels: int = 200704,
        rewrite_summary_violations: bool = True,
        max_retries: int = 2,
        initial_retry_seconds: float = 10,
        max_retry_seconds: float = 120,
        response_format_json: bool = True,
        request_extra_body: dict[str, Any] | None = None,
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
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=0,
        )
        self.model = model
        self.max_tokens = max_tokens
        self.last_attempt_count = 0
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if initial_retry_seconds <= 0 or max_retry_seconds <= 0:
            raise ValueError("retry delays must be positive")
        self.max_retries = max_retries
        self.initial_retry_seconds = initial_retry_seconds
        self.max_retry_seconds = max_retry_seconds
        if max_image_pixels <= 0:
            raise ValueError("max_image_pixels must be positive")
        self.max_image_pixels = max_image_pixels
        self.rewrite_summary_violations = rewrite_summary_violations
        self.response_format_json = response_format_json
        self.request_extra_body = request_extra_body or {}

    def describe(
        self, clip: VgentClipPlan, *, frames_per_request: int = 64
    ) -> dict[str, Any]:
        self.last_attempt_count = 0
        request_paths = prepare_request_frame_paths(
            clip,
            frames_per_request=frames_per_request,
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": DESCRIPTION_PROMPT}]
        for frame_path in request_paths:
            encoded = _encode_resized_jpeg(frame_path, self.max_image_pixels)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                }
            )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": OBSERVATION_FIRST_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        payload = self._request_payload(messages)
        _validate_description_payload(payload)
        violations = find_summary_rule_violations(str(payload["summary"]))
        if violations and self.rewrite_summary_violations:
            rewrite_messages: list[dict[str, Any]] = [
                {"role": "system", "content": OBSERVATION_FIRST_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Rewrite only the summary and medical_inferences from the "
                        "following JSON without adding new visual information. The summary violates the observation-first "
                        "rules with these exact terms: "
                        f"{', '.join(violations)}. Remove those concepts and use "
                        "literal appearance words such as instrument, view, area, "
                        "red fluid, or clear fluid. Keep the summary to one sentence "
                        "of at most 30 words. Use medical_inferences=[] when the "
                        "evidence is only generic interaction. Return only this small "
                        'JSON shape: {"summary":"...","medical_inferences":[]}. '
                        "Do not repeat observed_facts or uncertainties.\n\nInput JSON:\n"
                        f"{json.dumps(payload, ensure_ascii=False)}"
                    ),
                },
            ]
            try:
                rewrite = self._request_payload(
                    rewrite_messages,
                    max_tokens=min(self.max_tokens, 384),
                )
                if "summary" not in rewrite or "medical_inferences" not in rewrite:
                    raise RuntimeError(
                        "Observation rewrite is missing summary or medical_inferences"
                    )
                payload = {
                    **payload,
                    "summary": rewrite["summary"],
                    "medical_inferences": rewrite["medical_inferences"],
                }
            except (RuntimeError, TypeError):
                # Preserve the already validated first response when only the
                # optional observation rewrite returns malformed JSON.
                pass
            _validate_description_payload(payload)
        return payload

    @staticmethod
    def _is_retryable_error(error: Exception) -> bool:
        status_code = getattr(error, "status_code", None)
        if status_code in {408, 409, 429, 500, 502, 503, 504}:
            return True
        return type(error).__name__ in {
            "APIConnectionError",
            "APITimeoutError",
            "TimeoutException",
        }

    def _retry_delay(self, error: Exception, retry_number: int) -> float:
        delay = min(
            self.initial_retry_seconds * (2**retry_number),
            self.max_retry_seconds,
        )
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)
        if headers:
            try:
                retry_after = float(headers.get("retry-after", 0))
            except (TypeError, ValueError):
                retry_after = 0
            if retry_after > 0:
                delay = min(max(delay, retry_after), self.max_retry_seconds)
        return delay

    def _request_payload(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if self.response_format_json:
            request["response_format"] = {"type": "json_object"}
        if self.request_extra_body:
            request["extra_body"] = self.request_extra_body
        for retry_number in range(self.max_retries + 1):
            self.last_attempt_count += 1
            try:
                response = self.client.chat.completions.create(**request)
                break
            except Exception as error:
                if retry_number >= self.max_retries or not self._is_retryable_error(
                    error
                ):
                    raise
                delay = self._retry_delay(error, retry_number)
                status_code = getattr(error, "status_code", "network")
                print(
                    f"Retryable API error HTTP {status_code}; retry "
                    f"{retry_number + 1}/{self.max_retries} in {delay:g}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
        text = response.choices[0].message.content or ""
        return _parse_json_object(text)


def _validate_description_payload(payload: dict[str, Any]) -> None:
    required = {
        "summary",
        "observed_facts",
        "medical_inferences",
        "uncertainties",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise RuntimeError(f"Description is missing required keys: {missing}")
    observed = payload["observed_facts"]
    if not isinstance(observed, dict):
        raise TypeError("observed_facts must be a JSON object")
    observed_required = {
        "visible_anatomy",
        "visible_instruments",
        "visible_objects",
        "actions",
        "state_changes",
        "visual_evidence",
    }
    observed_missing = sorted(observed_required - observed.keys())
    if observed_missing:
        raise RuntimeError(
            f"observed_facts is missing required keys: {observed_missing}"
        )


def find_summary_rule_violations(summary: str) -> list[str]:
    """Return forbidden terms present in a generated summary."""
    lowered = summary.lower()
    return [
        term
        for term in SUMMARY_FORBIDDEN_TERMS
        if re.search(rf"\b{re.escape(term)}\b", lowered)
    ]


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
    pixels = height * width
    if pixels > max_pixels:
        scale = math.sqrt(max_pixels / pixels)
        resized_width = max(1, int(width * scale))
        resized_height = max(1, int(height * scale))
        image = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError(f"Cannot encode frame: {path}")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


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
