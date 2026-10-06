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
    # None = auto: use the whole training set as the minibatch when it's
    # small enough (see runner.py's SMALL_DATASET_THRESHOLD) instead of
    # GEPA's own default (3) -- a 3-row sample makes an asymmetric-penalty
    # metric noisy (often zero positive-labeled rows in the sample at all),
    # while the full set every time is a stable, noise-free signal. Set an
    # explicit int to override either way.
    reflection_minibatch_size: int | None = None
    val_fraction: float = 0.3
    # Opt-in (0.0 = off): a held-out split GEPA never sees during search, used
    # only for a final seed-vs-best sanity check against overfitting to the
    # validation set. Worth turning on once you have enough rows that train/
    # val/test can each still be meaningful (a few dozen total, at least).
    test_fraction: float = 0.0
    seed: int = 0
    # How many rows to evaluate concurrently within one evaluate() call (not
    # across GEPA's own iterations, which stay sequential). Keep at 1 unless
    # you know your task_lm's rate limits can take it -- firing several
    # requests at once is exactly what triggered the free-tier 429s logged in
    # AGENTS.md. Safe to raise on a paid tier or a higher-limit provider.
    max_workers: int = 1
    # Crossover between two Pareto-frontier candidates (GEPA's "merge"
    # strategy), on top of reflective mutation -- lets a candidate that's
    # best on some examples and one that's best on others get combined
    # instead of only ever evolving one lineage at a time. Enabled by
    # default; set False to go back to reflection-only.
    use_merge: bool = True
    max_merge_invocations: int = 5
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
