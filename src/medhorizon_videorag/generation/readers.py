"""Pluggable multiple-choice readers for retrieved video evidence."""
from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any, Protocol, Sequence

from medhorizon_videorag.core.schemas import ReaderAnswer


class VideoReader(Protocol):
    def answer(self, question: str, choices: Sequence[str], evidence: Sequence[dict[str, Any]]) -> ReaderAnswer: ...


def _choice_labels(choices: Sequence[str]) -> list[str]:
    labels = []
    for index, choice in enumerate(choices):
        match = re.match(r"\s*([A-Za-z0-9]+)[.):：]\s*", choice)
        labels.append(match.group(1).upper() if match else chr(ord("A") + index))
    return labels


class MockChoiceReader:
    """Offline smoke-test reader. It always chooses the first listed option."""

    def answer(self, question: str, choices: Sequence[str], evidence: Sequence[dict[str, Any]]) -> ReaderAnswer:
        labels = _choice_labels(choices)
        if not labels:
            raise ValueError("MedHorizon QA requires multiple-choice options")
        return ReaderAnswer(labels[0], "mock reader: no visual reasoning performed")


class OpenAICompatibleVisionReader:
    """OpenAI-compatible Chat Completions reader, usable with hosted or local VLMs."""

    def __init__(self, model: str, api_key_env: str = "OPENAI_API_KEY", base_url: str | None = None, max_tokens: int = 256) -> None:
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError("Install model dependencies: pip install -e '.[models]'") from error
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"Set {api_key_env} before using the OpenAI-compatible reader")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_tokens = max_tokens

    def answer(self, question: str, choices: Sequence[str], evidence: Sequence[dict[str, Any]]) -> ReaderAnswer:
        labels = _choice_labels(choices)
        prompt = (
            "You are a medical-video multiple-choice QA reader. Compare every listed option against the supplied "
            "frames and choose the best-supported option. Use only visible evidence: do not invent anatomy, surgical "
            "steps, instruments, or actions that cannot be seen. Return exactly one JSON object with keys choice and "
            "rationale. The rationale must be a short statement of visible evidence, not unsupported medical knowledge. choice must be one of "
            f"{labels}.\nQuestion: {question}\nChoices:\n" + "\n".join(choices) + "\nEvidence:"
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for item in evidence:
            content.append({"type": "text", "text": f"\nChunk {item['chunk_id']} ({item['start_seconds']:.1f}s–{item['end_seconds']:.1f}s):"})
            for frame_path in item.get("reader_frame_paths", []):
                encoded = base64.b64encode(Path(frame_path).read_bytes()).decode("ascii")
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})
        response = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": content}],
            temperature=0, max_tokens=self.max_tokens,
        )
        text = response.choices[0].message.content or ""
        try:
            payload = json.loads(re.search(r"\{.*\}", text, re.DOTALL).group())
            choice = str(payload["choice"]).strip().upper()
            rationale = str(payload.get("rationale", ""))
        except (AttributeError, json.JSONDecodeError, KeyError) as error:
            raise RuntimeError(f"Reader did not return the requested JSON: {text}") from error
        if choice not in labels:
            raise RuntimeError(f"Reader returned invalid choice {choice!r}; expected one of {labels}")
        return ReaderAnswer(choice, rationale)


def build_video_reader(config: dict[str, Any]) -> VideoReader:
    provider = config.get("provider", "mock")
    if provider in {"mock", "extractive"}:
        return MockChoiceReader()
    if provider in {"openai", "openai_compatible"}:
        model = config.get("model")
        if not model:
            raise ValueError("llm.model is required for the OpenAI-compatible reader")
        return OpenAICompatibleVisionReader(
            model=model, api_key_env=config.get("api_key_env", "OPENAI_API_KEY"),
            base_url=config.get("base_url"), max_tokens=int(config.get("max_tokens", 256)),
        )
    raise ValueError(f"Unknown llm.provider: {provider}")
