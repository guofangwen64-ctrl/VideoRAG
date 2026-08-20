from __future__ import annotations

import base64
import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .schemas import VgentClipPlan

DESCRIPTION_PROMPT_VERSION = "medical_clip_observation_first_v4"

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
        if max_image_pixels <= 0:
            raise ValueError("max_image_pixels must be positive")
        self.max_image_pixels = max_image_pixels

    def describe(
        self, clip: VgentClipPlan, *, frames_per_request: int = 64
    ) -> dict[str, Any]:
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
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=0,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content or ""
        payload = _parse_json_object(text)
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
        return payload


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
