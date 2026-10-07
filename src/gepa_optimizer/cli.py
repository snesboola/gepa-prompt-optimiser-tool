from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from .config import DEFAULT_CONFIG_NAME, ProjectConfig
from .dataset import inspect_dataset, load_dataset
from .metrics import TEMPLATE_EXPECTED_FIELDS, TEMPLATES, scaffold_metric

WORKFLOW_TEMPLATE = """\
# Generic workflow spec optimized by gepa-optimizer.
# Each node with `optimize: true` is a text component GEPA is allowed to evolve.
# Templates use $var (string.Template) substitution: dataset row fields are
# available directly ($input, $reference, ...); earlier nodes' outputs become
# available under their own id ($node_id).

nodes:
  - id: system_prompt
    type: llm
    optimize: system
    system: |
      You are a helpful assistant.
    user: |
      $input

final_output: system_prompt
"""


def cmd_init(args: argparse.Namespace) -> None:
    project_root = Path(args.path)
    project_root.mkdir(parents=True, exist_ok=True)

    config_path = project_root / DEFAULT_CONFIG_NAME
    if config_path.exists() and not args.force:
        print(f"{config_path} already exists. Use --force to overwrite.", file=sys.stderr)
        sys.exit(1)

    config = ProjectConfig(goal=args.goal or "", criteria=args.criteria or "")
    config.save(config_path)

    workflow_path = project_root / config.workflow
    if not workflow_path.exists() or args.force:
        workflow_path.write_text(WORKFLOW_TEMPLATE)

    dataset_path = project_root / config.dataset
    if not dataset_path.exists():
        dataset_path.write_text(
            '{"input": "example input", "reference": "example expected output"}\n'
        )

    print(f"Initialized project at {project_root}")
    print(f"  config:   {config_path}")
    print(f"  workflow: {workflow_path}  (edit: mark the node(s) you want optimized)")
    print(f"  dataset:  {dataset_path}  (replace with your real rows)")
    print("Next: gepa-opt suggest-metric --type <exact_match|keyword_presence|llm_judge>")


def cmd_inspect_dataset(args: argparse.Namespace) -> None:
    project_root = Path(args.path)
    config = ProjectConfig.load(project_root / DEFAULT_CONFIG_NAME)
    rows = load_dataset(project_root / config.dataset)
    info = inspect_dataset(rows)

    print(f"{info['n_rows']} row(s), columns: {info['columns']}\n")
    for col, stats in info["column_stats"].items():
        tag = " <- likely categorical (label/verdict/category?)" if stats["likely_categorical"] else ""
        print(f"  {col}: {stats['n_distinct']} distinct value(s), e.g. {stats['examples']}{tag}")
    print("\nSample row(s):")
    for row in info["sample"]:
        print(f"  {row}")


def cmd_suggest_metric(args: argparse.Namespace) -> None:
    project_root = Path(args.path)
    config = ProjectConfig.load(project_root / DEFAULT_CONFIG_NAME)

    goal = args.goal or config.goal
    criteria = args.criteria or config.criteria
    if not goal:
        print("No --goal given and none stored in config. Provide --goal.", file=sys.stderr)
        sys.exit(1)

    out_path = project_root / config.metric
    if out_path.exists() and not args.force:
        print(f"{out_path} already exists. Use --force to overwrite.", file=sys.stderr)
        sys.exit(1)

    # Check the chosen template's assumed field names against what the real
    # dataset actually has, *before* scaffolding -- a silent mismatch here
    # means a KeyError (or worse, a quietly-always-empty fallback) later,
    # not when it's cheap to catch.
    try:
        rows = load_dataset(project_root / config.dataset)
        info = inspect_dataset(rows)
        print(f"Dataset columns: {info['columns']}")
        missing = [f for f in TEMPLATE_EXPECTED_FIELDS.get(args.type, []) if f not in info["columns"]]
        if missing:
            candidates = [
                col for col, stats in info["column_stats"].items() if stats["likely_categorical"]
            ]
            suggestion = (
                f" Column(s) that look like they might be it (few distinct values): {candidates} -- "
                f"consider setting LABEL_FIELD (or the equivalent constant) to one of those in metric.py."
                if candidates
                else " No column looked like an obvious candidate (checked for low-distinct-value columns) -- "
                "double check your dataset actually has the field this template needs."
            )
            print(
                f"WARNING: the {args.type!r} template's scaffold expects a {missing} field, "
                f"but your dataset's columns are {info['columns']} -- none match.{suggestion}",
                file=sys.stderr,
            )
    except FileNotFoundError:
        pass  # dataset not written yet (e.g. mid-scaffolding a brand-new project) -- nothing to check against

    scaffold_metric(args.type, goal, criteria, out_path)

    if args.goal or args.criteria:
        config.goal = goal
        config.criteria = criteria
        config.save(project_root / DEFAULT_CONFIG_NAME)

    print(f"Drafted a {args.type!r} metric at {out_path}.")
    print("Review it, edit the scoring logic to match your real criteria, then approve by running optimize.")
    print(f"(available templates: {list(TEMPLATES)})")


def cmd_status(args: argparse.Namespace) -> None:
    """Where this project actually stands right now -- lets anyone (the
    user or whichever agent picks this back up) re-orient after a break
    without re-reading the whole conversation from scratch."""
    project_root = Path(args.path)
    config_path = project_root / DEFAULT_CONFIG_NAME

    print(f"Project: {project_root.resolve()}\n")

    if not config_path.exists():
        print(f"No {DEFAULT_CONFIG_NAME} yet. Next: gepa-opt init --goal \"...\" --criteria \"...\"")
        return

    config = ProjectConfig.load(config_path)
    print(f"gepa.config.yaml: present")
    print(f"  goal:     {config.goal or '(not set)'}")
    print(f"  criteria: {config.criteria or '(not set)'}")

    workflow_path = project_root / config.workflow
    if workflow_path.exists():
        edited = workflow_path.read_text() != WORKFLOW_TEMPLATE
        print(f"workflow.yaml: present, {'edited' if edited else 'UNEDITED -- still the default scaffold'}")
    else:
        print("workflow.yaml: MISSING")

    dataset_path = project_root / config.dataset
    if dataset_path.exists():
        try:
            n_rows = len(load_dataset(dataset_path))
            print(f"dataset ({dataset_path.name}): present, {n_rows} row(s)")
        except Exception as exc:
            print(f"dataset ({dataset_path.name}): present but failed to parse: {exc}")
    else:
        print(f"dataset ({dataset_path.name}): MISSING")

    metric_path = project_root / config.metric
    if metric_path.exists():
        content = metric_path.read_text()
        unedited_markers = ["TODO: replace", "OTHER_LABEL"]
        looks_unedited = any(marker in content for marker in unedited_markers)
        note = " -- still has unedited scaffold markers (e.g. TODO/OTHER_LABEL), probably not reviewed yet" if looks_unedited else ""
        print(f"metric.py: present{note}")
    else:
        print("metric.py: MISSING -- next: gepa-opt inspect-dataset, then suggest-metric")

    run_dir = project_root / config.run_dir
    report_path = run_dir / "report.md"
    progress_path = run_dir / "progress.log"
    if report_path.exists():
        import datetime
        mtime = datetime.datetime.fromtimestamp(report_path.stat().st_mtime)
        best_path = run_dir / "best_candidate.json"
        best_note = ""
        if best_path.exists():
            import json

            best = json.loads(best_path.read_text())
            best_note = f", best score {best.get('best_score'):.4f}" if "best_score" in best else ""
        print(f"\nLast run: completed {mtime:%Y-%m-%d %H:%M}{best_note} -- {report_path}")
    elif progress_path.exists():
        lines = progress_path.read_text().strip().splitlines()
        print(f"\nRun in progress or interrupted -- last line: {lines[-1] if lines else '(empty)'}")
        print(f"  ({progress_path})")
    else:
        print("\nNo run yet. Next: gepa-opt validate, then gepa-opt optimize")


def cmd_estimate(args: argparse.Namespace) -> None:
    from .runner import estimate_run  # deferred: gepa import is heavy and optional until needed

    project_root = Path(args.path)
    config = ProjectConfig.load(project_root / DEFAULT_CONFIG_NAME)

    est = estimate_run(config, project_root=project_root, max_metric_calls=args.max_metric_calls)
    print(f"Rough estimate for a budget of {est['budget']} metric calls (not a guarantee):")
    print(f"  dataset: {est['trainset_size']} train / {est['valset_size']} val / {est['testset_size']} test row(s)")
    print(f"  ~{est['est_task_calls']} task_lm call(s) + ~{est['est_reflection_calls']} reflection_lm call(s)", end="")
    if est["est_merge_calls"]:
        print(f" + up to {est['est_merge_calls']} merge call(s)", end="")
    print(f" = ~{est['est_total_calls']} total LM call(s)")
    print(f"  ~{est['est_minutes_low']}-{est['est_minutes_high']} minute(s) (assuming 1.5-5s per call)")


def cmd_validate(args: argparse.Namespace) -> None:
    from .runner import validate_metric  # deferred: gepa import is heavy and optional until needed

    project_root = Path(args.path)
    config = ProjectConfig.load(project_root / DEFAULT_CONFIG_NAME)

    print(f"Validating metric against {args.n_rows} real dataset row(s) (before spending a real budget)...")
    result = validate_metric(config, project_root=project_root, n_rows=args.n_rows)
    print(f"Checked {result['n_checked']}, failed {result['n_failed']}. Scores: {result['scores']}")
    if result["n_failed"] > 0:
        print(f"Failures: {result['failures']}", file=sys.stderr)
        sys.exit(1)
    print("OK -- looks safe to run `gepa-opt optimize`.")


def cmd_optimize(args: argparse.Namespace) -> None:
    from .runner import run_optimization  # deferred: gepa import is heavy and optional until needed

    project_root = Path(args.path)
    config = ProjectConfig.load(project_root / DEFAULT_CONFIG_NAME)

    if args.max_metric_calls is not None:
        config.max_metric_calls = args.max_metric_calls

    resume = True if args.resume else (False if args.fresh else None)
    state_file = project_root / config.run_dir / "gepa_state.bin"
    if resume is None and state_file.exists():
        print(
            f"{state_file} exists from a previous run. Re-run with --resume to continue it, "
            f"or --fresh to archive it and start over.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Running GEPA optimization (budget: {config.max_metric_calls} metric calls)...")
    result = run_optimization(config, project_root=project_root, resume=resume, skip_validate=args.skip_validate)
    from .report import best_score

    print(f"\nDone. Best score: {best_score(result):.4f} (candidate {result.best_idx} of {result.num_candidates})")
    print(f"Report: {project_root / config.run_dir / 'report.md'}")


def cmd_report(args: argparse.Namespace) -> None:
    project_root = Path(args.path)
    config = ProjectConfig.load(project_root / DEFAULT_CONFIG_NAME)
    report_path = project_root / config.run_dir / "report.md"
    if not report_path.exists():
        print(f"No report found at {report_path}. Run `gepa-opt optimize` first.", file=sys.stderr)
        sys.exit(1)
    print(report_path.read_text())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="gepa-opt", description="Agent-agnostic GEPA prompt optimizer.")
    parser.add_argument("--path", default=".", help="Project directory (default: current directory)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Scaffold a new optimization project")
    p_init.add_argument("--goal", help="What you're optimizing for, in plain language")
    p_init.add_argument("--criteria", help="Specific criteria the output must meet")
    p_init.add_argument("--force", action="store_true")
    p_init.set_defaults(func=cmd_init)

    p_inspect = sub.add_parser("inspect-dataset", help="Show the dataset's real columns, value samples, and sample rows")
    p_inspect.set_defaults(func=cmd_inspect_dataset)

    p_status = sub.add_parser("status", help="Where this project stands: what's wired up, what's left")
    p_status.set_defaults(func=cmd_status)

    p_metric = sub.add_parser("suggest-metric", help="Scaffold a scoring function for review/approval")
    p_metric.add_argument("--type", default="exact_match", choices=list(TEMPLATES))
    p_metric.add_argument("--goal")
    p_metric.add_argument("--criteria")
    p_metric.add_argument("--force", action="store_true")
    p_metric.set_defaults(func=cmd_suggest_metric)

    p_estimate = sub.add_parser("estimate", help="Rough LM-call-count and time estimate before committing to a real run")
    p_estimate.add_argument("--max-metric-calls", type=int, default=None)
    p_estimate.set_defaults(func=cmd_estimate)

    p_validate = sub.add_parser("validate", help="Dry-run the metric against a couple of real rows before spending budget")
    p_validate.add_argument("--n-rows", type=int, default=2)
    p_validate.set_defaults(func=cmd_validate)

    p_opt = sub.add_parser("optimize", help="Run the GEPA loop")
    p_opt.add_argument("--max-metric-calls", type=int, default=None)
    p_opt.add_argument("--resume", action="store_true", help="Continue a previous run found in run_dir")
    p_opt.add_argument("--fresh", action="store_true", help="Archive a previous run in run_dir and start over")
    p_opt.add_argument("--skip-validate", action="store_true", help="Skip the automatic pre-flight metric check")
    p_opt.set_defaults(func=cmd_optimize)

    p_report = sub.add_parser("report", help="Print the summary report from the last run")
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    load_dotenv(Path(args.path) / ".env")  # e.g. GEMINI_API_KEY -- see .env.example
    args.func(args)


if __name__ == "__main__":
    main()
