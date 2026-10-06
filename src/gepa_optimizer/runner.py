from __future__ import annotations

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


def _classification_stats_if_available(
    adapter: WorkflowGEPAAdapter, metric_module, valset: list[dict[str, Any]], candidate: dict[str, str]
) -> dict[str, Any] | None:
    """If the metric module opts in (exposes `classify` + `POSITIVE_LABEL`),
    re-evaluate `candidate` over `valset` and compute the real confusion-matrix
    recall/precision/F1 -- not the per-example proxy score GEPA optimized
    against, but the aggregate number that score was only ever a stand-in for.
    """
    if not (hasattr(metric_module, "classify") and hasattr(metric_module, "POSITIVE_LABEL")):
        return None
    eval_batch = adapter.evaluate(valset, candidate, capture_traces=True)
    traces = [t.trace for t in eval_batch.trajectories]
    return classification_report(
        valset, traces, metric_module.classify, metric_module.POSITIVE_LABEL,
        label_field=getattr(metric_module, "LABEL_FIELD", "label"),
    )


def run_optimization(config: ProjectConfig, project_root: str | Path = ".") -> "gepa.GEPAResult":
    project_root = Path(project_root)

    spec = WorkflowSpec.from_yaml(project_root / config.workflow)
    rows = load_dataset(project_root / config.dataset)
    trainset, valset = split_dataset(rows, val_fraction=config.val_fraction, seed=config.seed)

    metric_module = load_metric_module(project_root / config.metric)
    score_fn = metric_module.score

    task_lm = resolve_lm(config.task_lm, project_root=project_root)
    reflection_lm = resolve_lm(config.reflection_lm, project_root=project_root)
    judge_lm = resolve_lm(config.judge_lm, project_root=project_root) if config.judge_lm else None

    adapter = WorkflowGEPAAdapter(spec=spec, task_lm=task_lm, score_fn=score_fn, judge_lm=judge_lm)

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
        max_metric_calls=config.max_metric_calls,
        seed=config.seed,
        run_dir=str(run_dir),  # gepa==0.1.4 writes gepa_state.bin here (candidate pool, for resuming);
        # we separately dump our own JSON/HTML views of it below -- that's the human-readable "cached runs" artifact
        track_best_outputs=True,
        display_progress_bar=True,
    )

    import json

    (run_dir / "result.json").write_text(json.dumps(result.to_dict(), indent=2))
    (run_dir / "best_candidate.json").write_text(
        json.dumps({"best_candidate": result.best_candidate, "best_score": best_score(result)}, indent=2)
    )
    (run_dir / "candidate_tree.html").write_text(result.candidate_tree_html())

    seed_stats = _classification_stats_if_available(adapter, metric_module, valset, result.candidates[0])
    best_stats = _classification_stats_if_available(adapter, metric_module, valset, result.best_candidate)
    if best_stats is not None:
        (run_dir / "classification_report.json").write_text(
            json.dumps({"seed": seed_stats, "best": best_stats}, indent=2)
        )

    write_report(
        result,
        run_dir,
        config_summary={"goal": config.goal, "criteria": config.criteria},
        classification_stats={"seed": seed_stats, "best": best_stats} if best_stats is not None else None,
    )
    return result
