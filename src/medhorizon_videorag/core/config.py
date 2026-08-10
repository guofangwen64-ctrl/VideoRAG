from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    project: dict[str, Any] = field(default_factory=lambda: {"artifact_dir": "artifacts", "seed": 42})
    chunking: dict[str, Any] = field(default_factory=dict)
    vision: dict[str, Any] = field(default_factory=dict)
    retrieval: dict[str, Any] = field(default_factory=dict)
    llm: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    allowed = {name: raw.get(name, {}) for name in ExperimentConfig.__dataclass_fields__}
    return ExperimentConfig(**allowed)
