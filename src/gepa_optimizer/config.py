from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_NAME = "gepa.config.yaml"


@dataclass
class ProjectConfig:
    workflow: str = "workflow.yaml"
    dataset: str = "dataset.jsonl"
    metric: str = "metric.py"
    # Defaults target Google AI Studio's free tier (no card required) so a
    # fresh project is runnable without spending anything -- set GEMINI_API_KEY
    # and go. Swap to any other litellm model string (or a "callable:..." spec,
    # see llm.py) once you're ready to use a different provider.
    task_lm: str = "gemini/gemini-2.5-flash-lite"  # most generous free-tier RPM/day quota; called once per row
    # Also flash-lite, not a stronger model, by design: in testing, gemini-3.8-flash
    # (the officially current, non-deprecated model) returned persistent 503
    # "high demand" on every single reflection call on the free tier -- a real
    # infra/availability issue, not a code bug, logged in AGENTS.md. flash-lite
    # is the one actually proven to work end-to-end; swap this once 3.8-flash's
    # free-tier availability settles down, or any time you're on a paid tier.
    reflection_lm: str = "gemini/gemini-2.5-flash-lite"
    judge_lm: str | None = None
    max_metric_calls: int = 150
    val_fraction: float = 0.3
    seed: int = 0
    run_dir: str = "runs/latest"
    goal: str = ""
    criteria: str = ""

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG_NAME) -> "ProjectConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No config at {path}. Run `gepa-opt init` first, or pass --config."
            )
        data: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
        return cls(**data)

    def save(self, path: str | Path = DEFAULT_CONFIG_NAME) -> Path:
        path = Path(path)
        path.write_text(yaml.safe_dump(self.__dict__, sort_keys=False))
        return path
