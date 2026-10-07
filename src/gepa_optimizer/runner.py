from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import gepa

from .adapter import WorkflowGEPAAdapter
from .config import ProjectConfig
from .dataset import load_dataset, split_dataset
from .llm import resolve_lm
from .metrics import classification_report, load_metric_module
from .report import best_score, write_report
from .workflow import WorkflowSpec

# Below this many training rows, just use the whole training set as the
# reflection minibatch every time (see AGENTS.md's bug log for why: GEPA's
# own default of 3 makes a per-example proxy metric noisy or even undefined
# on a small random sample -- e.g. a 3-row sample of a recall-sensitive
# metric frequently contains zero positive-labeled rows at all). Above this,
# evaluating every candidate against the whole set on every reflection step
# gets expensive, so fall back to GEPA's own sampling default.
SMALL_DATASET_THRESHOLD = 25


class ProgressLogger:
    """A GEPACallback (duck-typed Protocol -- implement only the methods
    needed, per gepa's own callbacks.py) that writes one clean, regular
    line per iteration to both stdout and run_dir/progress.log.

    gepa's own run_log.txt is comprehensive but low-level (raw provider
    errors, retries, every reflection attempt) -- genuinely useful for
    debugging, not for "is this still going and how's it doing" at a
    glance. This is the short structured line meant for that: a human
    watching the terminal, or an agent polling progress.log while the CLI
    call runs, can answer "where are we" without parsing the verbose log.
    """

    def __init__(self, path: Path, max_metric_calls: int):
        self.path = path
        self.max_metric_calls = max_metric_calls

    def _log(self, line: str) -> None:
        with self.path.open("a") as f:
            f.write(line + "\n")
        print(line)

    def on_optimization_start(self, event) -> None:
        self.path.write_text("")  # fresh file per run
        self._log(
            f"[progress] starting: {event['trainset_size']} trainset / {event['valset_size']} valset "
            f"row(s), budget {self.max_metric_calls} metric calls"
        )

    def on_iteration_end(self, event) -> None:
        state = event["state"]
        scores = state.program_full_scores_val_set
        best = max(scores) if scores else None
        best_str = f"{best:.4f}" if best is not None else "n/a"
        status = "accepted a new candidate" if event["proposal_accepted"] else "no improvement"
        self._log(
            f"[progress] iteration {event['iteration']}: {status}. "
            f"{state.total_num_evals}/{self.max_metric_calls} metric calls used, "
            f"{len(state.program_candidates)} candidate(s) so far, best val score {best_str}."
        )

    def on_optimization_end(self, event) -> None:
        self._log(
            f"[progress] done: {event['total_iterations']} iteration(s), "
            f"{event['total_metric_calls']} metric call(s) used."
        )


def _classification_capability(metric_module) -> tuple[Any, str, str] | None:
    """(classify_fn, positive_label, label_field) if the metric module opts
    in by exposing `classify` + `POSITIVE_LABEL`, else None."""
    if not (hasattr(metric_module, "classify") and hasattr(metric_module, "POSITIVE_LABEL")):
        return None
    return metric_module.classify, metric_module.POSITIVE_LABEL, getattr(metric_module, "LABEL_FIELD", "label")


def _classification_stats(
    adapter: WorkflowGEPAAdapter,
    classify_fn,
    positive_label: str,
    label_field: str,
    eval_rows: list[dict[str, Any]],
    candidate: dict[str, str],
) -> dict[str, Any]:
    """Re-evaluate `candidate` over `eval_rows` and compute the real
    confusion-matrix recall/precision/F1 -- not the per-example proxy score
    (or proxy objective) GEPA optimized against, but the aggregate number
    those were only ever a stand-in for."""
    eval_batch = adapter.evaluate(eval_rows, candidate, capture_traces=True)
    traces = [t.trace for t in eval_batch.trajectories]
    return classification_report(eval_rows, traces, classify_fn, positive_label, label_field=label_field)


def _held_out_test_stats(
    adapter: WorkflowGEPAAdapter, testset: list[dict[str, Any]], candidate: dict[str, str]
) -> dict[str, Any]:
    """Plain mean score over a split GEPA never saw during search -- the
    actual sanity check against overfitting to the validation set, distinct
    from (and simpler than) the classification-specific stats above."""
    eval_batch = adapter.evaluate(testset, candidate, capture_traces=False)
    return {"mean_score": sum(eval_batch.scores) / len(eval_batch.scores), "n": len(eval_batch.scores)}


def _build_adapter(config: ProjectConfig, project_root: Path, metric_module) -> tuple[WorkflowGEPAAdapter, str]:
    spec = WorkflowSpec.from_yaml(project_root / config.workflow)
    task_lm = resolve_lm(config.task_lm, project_root=project_root)
    judge_lm = resolve_lm(config.judge_lm, project_root=project_root) if config.judge_lm else None

    adapter_kwargs: dict[str, Any] = {"max_workers": config.max_workers}
    frontier_type = "instance"
    classification_capability = _classification_capability(metric_module)
    if classification_capability is not None:
        classify_fn, positive_label, label_field = classification_capability
        adapter_kwargs.update(classify_fn=classify_fn, positive_label=positive_label, label_field=label_field)
        # "hybrid" keeps the usual per-instance Pareto tracking AND adds a
        # per-objective frontier for recall_proxy/precision_proxy, so a
        # recall-leaning candidate and a precision-leaning candidate can both
        # survive on the frontier instead of being collapsed into one scalar
        # via FN_PENALTY/FP_PENALTY alone. Verified against the real engine
        # (see AGENTS.md) -- not just assumed to work.
        frontier_type = "hybrid"

    adapter = WorkflowGEPAAdapter(
        spec=spec, task_lm=task_lm, score_fn=metric_module.score, judge_lm=judge_lm, **adapter_kwargs
    )
    return adapter, frontier_type


def estimate_run(
    config: ProjectConfig, project_root: str | Path = ".", max_metric_calls: int | None = None
) -> dict[str, Any]:
    """A rough, structural estimate of what a real `optimize()` run will
    cost in LM calls and wall-clock time -- pure arithmetic over the
    budget and dataset shape, no network calls and no pricing database to
    keep in sync with live provider rates (which would be its own ongoing
    maintenance burden and still only approximate). Meant to be shown to
    the user *before* committing to a real run; deliberately presented as
    a rough range, not a guarantee -- GEPA's real call count also depends
    on acceptance rate and how many merges actually trigger, which aren't
    knowable in advance.
    """
    project_root = Path(project_root)
    budget = max_metric_calls if max_metric_calls is not None else config.max_metric_calls
    rows = load_dataset(project_root / config.dataset)
    trainset, valset, testset = split_dataset(
        rows, val_fraction=config.val_fraction, test_fraction=config.test_fraction, seed=config.seed
    )

    if config.reflection_minibatch_size is not None:
        minibatch = config.reflection_minibatch_size
    elif len(trainset) <= SMALL_DATASET_THRESHOLD:
        minibatch = max(len(trainset), 1)
    else:
        minibatch = 3  # gepa's own default

    # Rough ceiling, not exact: each iteration risks one reflection_lm call;
    # max_metric_calls itself already counts every task_lm/adapter.evaluate()
    # call directly, by definition.
    est_iterations = max(1, budget // max(minibatch, 1))
    est_task_calls = budget
    est_reflection_calls = est_iterations
    est_merge_calls = config.max_merge_invocations if config.use_merge else 0
    total_calls = est_task_calls + est_reflection_calls + est_merge_calls

    seconds_per_call_low, seconds_per_call_high = 1.5, 5.0
    # Rows within a batch run concurrently above max_workers=1, so wall-clock
    # time per batch drops roughly in proportion (not exactly -- rate limits,
    # provider-side queuing, etc. aren't modeled here).
    concurrency = max(1, min(config.max_workers, minibatch or 1))
    est_seconds_low = (total_calls * seconds_per_call_low) / concurrency
    est_seconds_high = (total_calls * seconds_per_call_high) / concurrency

    return {
        "budget": budget,
        "trainset_size": len(trainset),
        "valset_size": len(valset),
        "testset_size": len(testset),
        "minibatch_size": minibatch,
        "est_iterations": est_iterations,
        "est_task_calls": est_task_calls,
        "est_reflection_calls": est_reflection_calls,
        "est_merge_calls": est_merge_calls,
        "est_total_calls": total_calls,
        "est_minutes_low": round(est_seconds_low / 60, 1),
        "est_minutes_high": round(est_seconds_high / 60, 1),
    }


def validate_metric(config: ProjectConfig, project_root: str | Path = ".", n_rows: int = 2) -> dict[str, Any]:
    """Run the metric against a couple of real dataset rows *before*
    committing to a full optimize() run. Several of this project's own real
    bugs (wrong field names, missing deps, a broken task_lm) only surfaced
    mid-run, after burning real API budget/quota -- this catches the same
    class of problem for ~n_rows worth of calls instead.

    Raises (propagated from adapter.evaluate()'s own loud-failure check) if
    every sampled row fails with a real exception. Returns a summary dict
    either way when it doesn't raise; `n_failed > 0` means at least one
    sampled row failed even though not all of them did -- worth surfacing,
    not necessarily fatal (could be one flaky network call).
    """
    project_root = Path(project_root)
    rows = load_dataset(project_root / config.dataset)
    if not rows:
        raise ValueError(f"{project_root / config.dataset} is empty -- nothing to validate against.")

    metric_module = load_metric_module(project_root / config.metric)
    spec = WorkflowSpec.from_yaml(project_root / config.workflow)
    adapter, _ = _build_adapter(config, project_root, metric_module)

    sample = rows[:n_rows]
    eval_batch = adapter.evaluate(sample, spec.seed_candidate(), capture_traces=True)

    failures = [
        (row, out.get("error")) for row, out in zip(sample, eval_batch.outputs) if isinstance(out, dict) and out.get("error")
    ]
    return {"n_checked": len(sample), "n_failed": len(failures), "scores": eval_batch.scores, "failures": failures}


def _resolve_run_dir(config: ProjectConfig, project_root: Path, resume: bool | None) -> Path:
    run_dir = project_root / config.run_dir
    state_file = run_dir / "gepa_state.bin"

    if not state_file.exists():
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    if resume is None:
        raise RuntimeError(
            f"{state_file} already exists from a previous run. gepa.optimize() resumes from it "
            f"automatically if left alone -- which silently served stale results from before a bug "
            f"fix earlier in this project's own history (see AGENTS.md). Pass resume=True to actually "
            f"continue that run, or resume=False to archive it and start fresh "
            f"(gepa-opt optimize --resume or --fresh)."
        )
    if resume is False:
        archive_dir = run_dir.parent / f"{run_dir.name}_archived_{int(time.time())}"
        run_dir.rename(archive_dir)
        print(f"Archived previous run to {archive_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir


def run_optimization(
    config: ProjectConfig,
    project_root: str | Path = ".",
    resume: bool | None = None,
    skip_validate: bool = False,
) -> "gepa.GEPAResult":
    project_root = Path(project_root)

    metric_module = load_metric_module(project_root / config.metric)

    if not skip_validate:
        validation = validate_metric(config, project_root=project_root)
        if validation["n_failed"] > 0:
            print(
                f"WARNING: {validation['n_failed']}/{validation['n_checked']} pre-flight validation "
                f"row(s) failed: {validation['failures']}. Proceeding anyway -- pass skip_validate=True "
                f"(or --skip-validate) to silence this once you've confirmed it's expected."
            )

    run_dir = _resolve_run_dir(config, project_root, resume)

    spec = WorkflowSpec.from_yaml(project_root / config.workflow)
    rows = load_dataset(project_root / config.dataset)
    trainset, valset, testset = split_dataset(
        rows, val_fraction=config.val_fraction, test_fraction=config.test_fraction, seed=config.seed
    )

    classification_capability = _classification_capability(metric_module)
    reflection_lm = resolve_lm(config.reflection_lm, project_root=project_root)
    adapter, frontier_type = _build_adapter(config, project_root, metric_module)

    if config.reflection_minibatch_size is not None:
        minibatch_size = config.reflection_minibatch_size
    elif len(trainset) <= SMALL_DATASET_THRESHOLD:
        minibatch_size = len(trainset)  # every reflection step sees the whole training set, not a noisy sample
    else:
        minibatch_size = None  # GEPA's own default (3)

    result = gepa.optimize(
        seed_candidate=spec.seed_candidate(),
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=minibatch_size,
        frontier_type=frontier_type,
        use_merge=config.use_merge,
        max_merge_invocations=config.max_merge_invocations,
        max_metric_calls=config.max_metric_calls,
        seed=config.seed,
        run_dir=str(run_dir),  # gepa==0.1.4 writes gepa_state.bin here (candidate pool, for resuming);
        # we separately dump our own JSON/HTML views of it below -- that's the human-readable "cached runs" artifact
        track_best_outputs=True,
        display_progress_bar=True,
        callbacks=[ProgressLogger(run_dir / "progress.log", config.max_metric_calls)],
    )

    (run_dir / "result.json").write_text(json.dumps(result.to_dict(), indent=2))
    (run_dir / "best_candidate.json").write_text(
        json.dumps({"best_candidate": result.best_candidate, "best_score": best_score(result)}, indent=2)
    )
    (run_dir / "candidate_tree.html").write_text(result.candidate_tree_html())

    classification_stats = None
    if classification_capability is not None:
        classify_fn, positive_label, label_field = classification_capability
        # Prefer the held-out test set when there is one (never seen during
        # search, the more rigorous number); fall back to the valset GEPA
        # already tracked scores against otherwise.
        eval_rows = testset or valset
        seed_stats = _classification_stats(adapter, classify_fn, positive_label, label_field, eval_rows, result.candidates[0])
        best_stats = _classification_stats(adapter, classify_fn, positive_label, label_field, eval_rows, result.best_candidate)
        classification_stats = {"seed": seed_stats, "best": best_stats, "evaluated_on": "test" if testset else "val"}
        (run_dir / "classification_report.json").write_text(json.dumps(classification_stats, indent=2))

    held_out_stats = None
    if testset:
        held_out_stats = {
            "seed": _held_out_test_stats(adapter, testset, result.candidates[0]),
            "best": _held_out_test_stats(adapter, testset, result.best_candidate),
        }
        (run_dir / "held_out_test.json").write_text(json.dumps(held_out_stats, indent=2))

    write_report(
        result,
        run_dir,
        config_summary={"goal": config.goal, "criteria": config.criteria},
        classification_stats=classification_stats,
        held_out_stats=held_out_stats,
    )
    return result
