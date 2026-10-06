"""GEPAAdapter implementation wiring the generic WorkflowRunner + a metric
function into gepa's optimize() loop.

Swapping backends (e.g. to run the real workflow on Dify instead of this
package's local node-by-node interpreter) means subclassing WorkflowRunner
and overriding `call_llm` (or `run`, for full control) -- this adapter class
itself does not need to change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gepa.core.adapter import EvaluationBatch

from .metrics import ScoreFn
from .workflow import WorkflowRunner, WorkflowSpec


@dataclass
class Trajectory:
    row: dict[str, Any]
    trace: dict[str, Any]
    score: float
    feedback: str


class WorkflowGEPAAdapter:
    """Candidate: dict mapping optimizable node id -> prompt text."""

    # GEPAAdapter is a Protocol, not an ABC -- the engine still does
    # `self.adapter.propose_new_texts` unconditionally (gepa's reflective_mutation.py),
    # so a concrete adapter that doesn't define its own proposer must declare
    # this explicitly or every reflection call raises AttributeError.
    propose_new_texts = None

    def __init__(
        self,
        spec: WorkflowSpec,
        task_lm,
        score_fn: ScoreFn,
        judge_lm=None,
    ):
        self.spec = spec
        self.task_lm = task_lm
        self.score_fn = score_fn
        self.judge_lm = judge_lm

    def evaluate(
        self, batch: list[dict[str, Any]], candidate: dict[str, str], capture_traces: bool = False
    ) -> EvaluationBatch[Trajectory, dict[str, Any]]:
        runner = WorkflowRunner(self.spec, self.task_lm)
        outputs: list[dict[str, Any]] = []
        scores: list[float] = []
        trajectories: list[Trajectory] | None = [] if capture_traces else None

        for row in batch:
            try:
                trace = runner.run(candidate, row)
                if "judge_lm" in self.score_fn.__code__.co_varnames:
                    s, feedback = self.score_fn(row, trace, judge_lm=self.judge_lm)
                else:
                    s, feedback = self.score_fn(row, trace)
            except Exception as exc:  # per-example failure must not abort the run
                trace = {"final_output": None, "nodes": [], "context": {}, "error": str(exc)}
                s, feedback = 0.0, f"Execution error: {exc}"

            outputs.append(trace)
            scores.append(float(s))
            if capture_traces:
                trajectories.append(Trajectory(row=row, trace=trace, score=float(s), feedback=feedback))

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[Trajectory, dict[str, Any]],
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        reflective: dict[str, list[dict[str, Any]]] = {c: [] for c in components_to_update}

        for traj in eval_batch.trajectories or []:
            node_by_id = {n["id"]: n for n in traj.trace.get("nodes", [])}
            for component_id in components_to_update:
                node = node_by_id.get(component_id)
                record = {
                    "Inputs": {
                        "row": traj.row,
                        "rendered_prompt": (node or {}).get("user") or (node or {}).get("system"),
                    },
                    "Generated Outputs": traj.trace.get("final_output"),
                    "Feedback": f"score={traj.score:.3f}. {traj.feedback}",
                }
                reflective[component_id].append(record)

        return reflective
