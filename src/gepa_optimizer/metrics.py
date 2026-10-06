"""Scoring function contract + built-in templates + scaffolding.

A metric module must expose:

    def score(row: dict, trace: dict) -> tuple[float, str]:
        ...

- row: the dataset row (your input/reference fields).
- trace: the WorkflowRunner.run() output -- trace["final_output"] is the
  workflow's final answer for this row; trace["nodes"] has every node's
  system/user/output for deeper inspection.
- returns (score, feedback): score in [0, 1] (higher is better), feedback is a
  short human-readable string explaining *why* -- this feedback is what GEPA's
  reflection step actually reads to propose a better prompt, so vague feedback
  ("wrong") produces vague mutations. Be specific: what was expected, what was
  produced, what rule was violated.

This module also provides a handful of ready-made templates
(`TEMPLATES`) and `scaffold_metric()`, which drafts an editable metric.py
based on a stated goal/criteria so a human (or the agent operating this tool)
can review and adjust it before it's used -- this is the "propose a scoring
function, get it approved" step in the optimization workflow.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable, Protocol

ScoreFn = Callable[[dict[str, Any], dict[str, Any]], "tuple[float, str]"]


class MetricModule(Protocol):
    def score(self, row: dict[str, Any], trace: dict[str, Any]) -> tuple[float, str]: ...


def load_metric(path: str | Path) -> ScoreFn:
    path = Path(path)
    spec = importlib.util.spec_from_file_location("gepa_optimizer_metric", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load metric module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "score"):
        raise AttributeError(f"{path} must define a top-level `score(row, trace)` function")
    return module.score


TEMPLATES: dict[str, str] = {
    "exact_match": '''\
"""Scaffolded metric: exact_match.
Goal: {goal}
Criteria: {criteria}

Expects each dataset row to have a `reference` field holding the exact
expected final answer. Edit the comparison logic below to fit your task
(e.g. case-insensitive, strip punctuation) before approving.
"""


def score(row: dict, trace: dict) -> tuple[float, str]:
    expected = str(row.get("reference", "")).strip()
    actual = str(trace["final_output"]).strip()

    if actual == expected:
        return 1.0, f"Matched expected output exactly: {{expected!r}}"
    return 0.0, (
        f"Mismatch. Expected: {{expected!r}}. Got: {{actual!r}}. "
        "Diagnose whether the prompt is ambiguous about output format, "
        "missing a constraint, or causing the model to add extra text."
    )
''',
    "keyword_presence": '''\
"""Scaffolded metric: keyword_presence.
Goal: {goal}
Criteria: {criteria}

Expects each dataset row to have a `required_keywords` field: a list of
strings that must all appear (case-insensitive) in the final output. Edit
the keyword source / matching rule below before approving.
"""


def score(row: dict, trace: dict) -> tuple[float, str]:
    required = [str(k).lower() for k in row.get("required_keywords", [])]
    output_lower = str(trace["final_output"]).lower()

    if not required:
        return 0.0, "No required_keywords provided for this row -- metric cannot score it."

    missing = [k for k in required if k not in output_lower]
    found = [k for k in required if k in output_lower]
    fraction = len(found) / len(required)

    feedback = (
        f"Found {{len(found)}}/{{len(required)}} required keywords: {{found}}. "
        f"Missing: {{missing}}."
        if missing
        else f"All required keywords present: {{found}}."
    )
    return fraction, feedback
''',
    "llm_judge": '''\
"""Scaffolded metric: llm_judge.
Goal: {goal}
Criteria: {criteria}

Uses a judge LM call to rate the final output against the stated criteria on
a 0-1 scale with a written rationale. Requires a `judge_lm` callable to be
wired in by the caller (see run_optimization's --judge-lm flag) -- edit the
rubric prompt below to match your actual criteria before approving.
"""

RUBRIC = """You are grading an AI system's output against this goal and criteria.

Goal: {goal}
Criteria: {criteria}

Input given to the system:
{{input}}

System's output:
{{output}}

Score the output from 0.0 (completely fails the criteria) to 1.0 (fully
meets the criteria). Respond in exactly this format:

SCORE: <number between 0 and 1>
REASON: <one or two sentences, specific about what was right or wrong>
"""


def score(row: dict, trace: dict, judge_lm=None) -> tuple[float, str]:
    if judge_lm is None:
        raise RuntimeError("llm_judge metric requires judge_lm to be passed through")

    prompt = RUBRIC.format(input=row.get("input", ""), output=trace["final_output"])
    reply = judge_lm(prompt)

    score_val, reason = 0.0, f"Could not parse judge reply: {{reply!r}}"
    for line in reply.splitlines():
        line = line.strip()
        if line.upper().startswith("SCORE:"):
            try:
                score_val = max(0.0, min(1.0, float(line.split(":", 1)[1].strip())))
            except ValueError:
                pass
        elif line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    return score_val, reason
''',
}


def scaffold_metric(metric_type: str, goal: str, criteria: str, out_path: str | Path) -> Path:
    if metric_type not in TEMPLATES:
        raise ValueError(f"Unknown metric_type {metric_type!r}. Choose from: {list(TEMPLATES)}")
    text = TEMPLATES[metric_type].format(goal=goal, criteria=criteria)
    out_path = Path(out_path)
    out_path.write_text(text)
    return out_path
