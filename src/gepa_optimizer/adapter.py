"""GEPAAdapter implementation wiring the generic WorkflowRunner + a metric
function into gepa's optimize() loop.

Swapping backends (e.g. to run the real workflow on Dify instead of this
package's local node-by-node interpreter) means subclassing WorkflowRunner
and overriding `call_llm` (or `run`, for full control) -- this adapter class
itself does not need to change.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from gepa.core.adapter import EvaluationBatch

from .metrics import ScoreFn, recall_precision_objective_scores
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
        max_workers: int = 1,
        classify_fn=None,
        positive_label: str | None = None,
        label_field: str = "label",
    ):
        self.spec = spec
        self.task_lm = task_lm
        self.score_fn = score_fn
        self.judge_lm = judge_lm
        self.max_workers = max_workers
        # When both are set, evaluate() additionally reports per-row
        # recall/precision proxy objectives for GEPA's native multi-objective
        # Pareto tracking (see metrics.recall_precision_objective_scores).
        self.classify_fn = classify_fn
        self.positive_label = positive_label
        self.label_field = label_field

    def _evaluate_one(self, row: dict[str, Any]) -> tuple[dict[str, Any], float, str, Exception | None]:
        runner = WorkflowRunner(self.spec, self.task_lm)
        try:
            trace = runner.run(self._candidate, row)
            if "judge_lm" in self.score_fn.__code__.co_varnames:
                s, feedback = self.score_fn(row, trace, judge_lm=self.judge_lm)
            else:
                s, feedback = self.score_fn(row, trace)
            return trace, float(s), feedback, None
        except Exception as exc:  # per-example failure must not abort the run on its own
            trace = {"final_output": None, "nodes": [], "context": {}, "error": str(exc)}
            return trace, 0.0, f"Execution error: {exc}", exc

    def evaluate(
        self, batch: list[dict[str, Any]], candidate: dict[str, str], capture_traces: bool = False
    ) -> EvaluationBatch[Trajectory, dict[str, Any]]:
        # _evaluate_one reads self._candidate rather than taking it as a
        # parameter so it can be handed straight to ThreadPoolExecutor.map
        # without a per-row closure/partial allocation.
        self._candidate = candidate

        if self.max_workers > 1 and len(batch) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                results = list(pool.map(self._evaluate_one, batch))
        else:
            results = [self._evaluate_one(row) for row in batch]

        outputs: list[dict[str, Any]] = []
        scores: list[float] = []
        trajectories: list[Trajectory] | None = [] if capture_traces else None
        objective_scores: list[dict[str, float]] | None = [] if self.classify_fn else None
        exceptions: list[Exception] = []

        for row, (trace, s, feedback, exc) in zip(batch, results):
            outputs.append(trace)
            scores.append(s)
            if exc is not None:
                exceptions.append(exc)
            if capture_traces:
                trajectories.append(Trajectory(row=row, trace=trace, score=s, feedback=feedback))
            if self.classify_fn:
                expected = str(row[self.label_field]).strip().upper()
                predicted = self.classify_fn(row, trace)
                objective_scores.append(
                    recall_precision_objective_scores(expected, predicted, self.positive_label)
                )

        if batch and len(exceptions) == len(batch):
            # Every single row raised a real exception (not a legitimately
            # low score) -- this is a config/dependency problem, not a bad
            # prompt, and silently reporting it as score=0.0 across the
            # board would look identical to "the prompt is just wrong" in
            # the run log. Surface it loudly instead. Per GEPAAdapter's own
            # contract, this is the sanctioned case to raise in (reserved
            # for unrecoverable, systemic failures); gepa.optimize()'s
            # raise_on_exception policy governs what happens from here.
            raise RuntimeError(
                f"All {len(batch)} example(s) in this batch raised an exception -- likely a "
                f"config/dependency problem (bad task_lm, missing package, bad credentials), "
                f"not a bad prompt. First error: {exceptions[0]!r}"
            )

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories, objective_scores=objective_scores)

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
