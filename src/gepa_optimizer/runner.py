from __future__ import annotations

import json
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


def run_optimization(config: ProjectConfig, project_root: str | Path = ".") -> "gepa.GEPAResult":
    project_root = Path(project_root)

    spec = WorkflowSpec.from_yaml(project_root / config.workflow)
    rows = load_dataset(project_root / config.dataset)
    trainset, valset, testset = split_dataset(
        rows, val_fraction=config.val_fraction, test_fraction=config.test_fraction, seed=config.seed
    )

    metric_module = load_metric_module(project_root / config.metric)
    score_fn = metric_module.score
    classification_capability = _classification_capability(metric_module)

    task_lm = resolve_lm(config.task_lm, project_root=project_root)
    reflection_lm = resolve_lm(config.reflection_lm, project_root=project_root)
    judge_lm = resolve_lm(config.judge_lm, project_root=project_root) if config.judge_lm else None

    adapter_kwargs: dict[str, Any] = {"max_workers": config.max_workers}
    frontier_type = "instance"
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

    adapter = WorkflowGEPAAdapter(spec=spec, task_lm=task_lm, score_fn=score_fn, judge_lm=judge_lm, **adapter_kwargs)

    if config.reflection_minibatch_size is not None:
        minibatch_size = config.reflection_minibatch_size
    elif len(trainset) <= SMALL_DATASET_THRESHOLD:
        minibatch_size = len(trainset)  # every reflection step sees the whole training set, not a noisy sample
    else:
        minibatch_size = None  # GEPA's own default (3)

    run_dir = project_root / config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    result = gepa.optimize(
        seed_candidate=spec.seed_candidate(),
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=minibatch_size,
        frontier_type=frontier_type,
        max_metric_calls=config.max_metric_calls,
        seed=config.seed,
        run_dir=str(run_dir),  # gepa==0.1.4 writes gepa_state.bin here (candidate pool, for resuming);
        # we separately dump our own JSON/HTML views of it below -- that's the human-readable "cached runs" artifact
        track_best_outputs=True,
        display_progress_bar=True,
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
