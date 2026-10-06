from __future__ import annotations

from pathlib import Path
from typing import Any

import gepa

from .adapter import WorkflowGEPAAdapter
from .config import ProjectConfig
from .dataset import load_dataset, split_dataset
from .llm import resolve_lm
from .metrics import load_metric
from .report import write_report
from .workflow import WorkflowSpec


def run_optimization(config: ProjectConfig, project_root: str | Path = ".") -> "gepa.GEPAResult":
    project_root = Path(project_root)

    spec = WorkflowSpec.from_yaml(project_root / config.workflow)
    rows = load_dataset(project_root / config.dataset)
    trainset, valset = split_dataset(rows, val_fraction=config.val_fraction, seed=config.seed)

    score_fn = load_metric(project_root / config.metric)

    task_lm = resolve_lm(config.task_lm, project_root=project_root)
    reflection_lm = resolve_lm(config.reflection_lm, project_root=project_root)
    judge_lm = resolve_lm(config.judge_lm, project_root=project_root) if config.judge_lm else None

    adapter = WorkflowGEPAAdapter(spec=spec, task_lm=task_lm, score_fn=score_fn, judge_lm=judge_lm)

    run_dir = project_root / config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    result = gepa.optimize(
        seed_candidate=spec.seed_candidate(),
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=reflection_lm,
        max_metric_calls=config.max_metric_calls,
        seed=config.seed,
        run_dir=str(run_dir),
        write_agent_state=True,  # keeps every candidate/trace/score on disk -- the "cached runs" the user asked for
        track_best_outputs=True,
        display_progress_bar=True,
    )

    import json

    (run_dir / "result.json").write_text(json.dumps(result.to_dict(), indent=2))
    (run_dir / "best_candidate.json").write_text(
        json.dumps({"best_candidate": result.best_candidate, "best_score": result.best_score}, indent=2)
    )
    (run_dir / "candidate_tree.html").write_text(result.candidate_tree_html())
    write_report(result, run_dir, config_summary={"goal": config.goal, "criteria": config.criteria})
    return result
