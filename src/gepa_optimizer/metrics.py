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


def load_metric_module(path: str | Path):
    path = Path(path)
    spec = importlib.util.spec_from_file_location("gepa_optimizer_metric", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load metric module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "score"):
        raise AttributeError(f"{path} must define a top-level `score(row, trace)` function")
    return module


def load_metric(path: str | Path) -> ScoreFn:
    return load_metric_module(path).score


def classification_report(
    rows: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    classify_fn: Callable[[dict[str, Any], dict[str, Any]], str],
    positive_label: str,
    label_field: str = "label",
) -> dict[str, Any]:
    """Real confusion-matrix recall/precision/F1, computed post-hoc over a
    full set of (row, trace) pairs.

    This is NOT the per-example score GEPA optimizes against -- recall and
    precision are properties of a whole batch (they need every row's true
    label and every row's prediction to even be defined), not something any
    single `score(row, trace)` call can produce. This function is how you
    get the actual number after the fact: run it over the seed candidate
    and the best candidate's traces and compare.

    `classify_fn` should return the predicted label (same scheme as
    `label_field`'s values, e.g. "HALLUCINATED"/"GROUNDED") for one
    (row, trace) pair -- a metric module written for this should expose a
    `classify(row, trace) -> str` function its own `score()` also calls
    internally, so the two never disagree.
    """
    tp = fp = tn = fn = 0
    for row, trace in zip(rows, traces):
        expected = str(row[label_field]).strip().upper()
        predicted = classify_fn(row, trace)
        if expected == positive_label:
            if predicted == positive_label:
                tp += 1
            else:
                fn += 1
        else:
            if predicted == positive_label:
                fp += 1
            else:
                tn += 1

    recall = tp / (tp + fn) if (tp + fn) else None
    precision = tp / (tp + fp) if (tp + fp) else None
    f1 = (2 * precision * recall / (precision + recall)) if (recall and precision and (precision + recall)) else None

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "positive_label": positive_label,
    }


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
    "classification": '''\
"""Scaffolded metric: classification (binary, with recall/precision tradeoff).
Goal: {goal}
Criteria: {criteria}

For tasks like a hallucination judge, where you care about recall and
precision specifically -- not just raw accuracy. Recall/precision are
AGGREGATE numbers (they need the whole confusion matrix); no single
score(row, trace) call can compute them. What this CAN do per example is
weight the two kinds of mistake differently, which is what GEPA actually
optimizes against: a missed positive (false negative, hurts recall) is
penalized harder than a false alarm (false positive, hurts precision) by
default -- tune FN_PENALTY / FP_PENALTY below to match your real tradeoff
(lower FN_PENALTY = push harder for recall; raise FP_PENALTY = push harder
for precision).

Expects each dataset row to have a `label` field with POSITIVE_LABEL or
the negative class. `classify()` is exposed separately (not just inlined in
score()) so the reporting layer can call it after optimization to compute
the REAL recall/precision/F1 the winning prompt achieves -- see
metrics.classification_report() and runner.py.
"""

POSITIVE_LABEL = "HALLUCINATED"  # edit to your actual positive-class label
FN_PENALTY = 0.0   # score when a real positive is missed (false negative)
FP_PENALTY = 0.4   # score when a real negative is wrongly flagged (false positive)


def classify(row: dict, trace: dict) -> str:
    """Parse the workflow's final_output into a label. Edit this to match
    your actual output format -- this scaffold assumes the output mentions
    one of the two class names somewhere."""
    output = str(trace["final_output"]).upper()
    mentions_positive = POSITIVE_LABEL in output
    # TODO: replace "OTHER_LABEL" with your actual negative-class label
    mentions_negative = "OTHER_LABEL" in output
    if mentions_positive and mentions_negative:
        return "AMBIGUOUS"
    if mentions_positive:
        return POSITIVE_LABEL
    if mentions_negative:
        return "OTHER_LABEL"
    return "UNCLEAR"


def score(row: dict, trace: dict) -> tuple[float, str]:
    expected = row["label"].strip().upper()
    predicted = classify(row, trace)

    if predicted == expected:
        return 1.0, f"Correctly judged as {{expected}}."

    if expected == POSITIVE_LABEL:
        return FN_PENALTY, (
            f"Missed a real {{POSITIVE_LABEL}} (expected {{expected}}, got {{predicted}} from "
            f"{{trace['final_output']!r}}). Missing real positives is the costlier mistake here -- "
            "the prompt should be more willing to flag borderline cases."
        )
    return FP_PENALTY, (
        f"False alarm: judged {{predicted}} but expected {{expected}}, from "
        f"{{trace['final_output']!r}}. Too many false alarms erode trust; tighten the criteria "
        "for what actually counts as a positive."
    )
''',
    "composite": '''\
"""Scaffolded metric: composite (hard checks + qualitative judgment).
Goal: {goal}
Criteria: {criteria}

For criteria that mix a hard, checkable requirement (must mention an exact
phrase, must include specific terms) with a vaguer, qualitative one (must
"capture industry detail", must "sound professional") -- no single built-in
template does both, because they need different techniques. This scaffold
composes them: hard requirements gate first (fail immediately, score 0,
regardless of how good the rest looks -- a judge call can't be allowed to
paper over a missing required phrase), and only once those pass does a
judge_lm call score the qualitative part. Requires `judge_lm` to be wired in
(see gepa.config.yaml's judge_lm field) -- edit both the hard checks and the
rubric below to match your actual criteria.
"""

# Hard, checkable requirements -- edit this list. Each row can also carry
# its own `required_phrases` field if the exact phrase varies per example;
# this falls back to the list below when a row doesn't specify one.
DEFAULT_REQUIRED_PHRASES: list[str] = []

RUBRIC = """You are grading an AI system's output against this specific criterion
(the hard requirements have already been checked separately -- grade ONLY
the qualitative aspect below):

{criteria}

Output to grade:
{{output}}

Score from 0.0 (completely fails this criterion) to 1.0 (fully meets it).
Respond in exactly this format:

SCORE: <number between 0 and 1>
REASON: <one or two sentences, specific about what was right or wrong>
"""


def score(row: dict, trace: dict, judge_lm=None) -> tuple[float, str]:
    output = str(trace["final_output"])
    required = row.get("required_phrases", DEFAULT_REQUIRED_PHRASES)

    missing = [p for p in required if p.lower() not in output.lower()]
    if missing:
        return 0.0, (
            f"Missing required phrase(s) {{missing}} entirely -- hard fail regardless of "
            f"quality elsewhere. Output was: {{output!r}}"
        )

    if judge_lm is None:
        raise RuntimeError("composite metric requires judge_lm to be passed through")

    prompt = RUBRIC.format(output=output)
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

    return score_val, f"(required phrases present) {{reason}}"
''',
}


def scaffold_metric(metric_type: str, goal: str, criteria: str, out_path: str | Path) -> Path:
    if metric_type not in TEMPLATES:
        raise ValueError(f"Unknown metric_type {metric_type!r}. Choose from: {list(TEMPLATES)}")
    text = TEMPLATES[metric_type].format(goal=goal, criteria=criteria)
    out_path = Path(out_path)
    out_path.write_text(text)
    return out_path
