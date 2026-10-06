from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import DEFAULT_CONFIG_NAME, ProjectConfig
from .metrics import TEMPLATES, scaffold_metric

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

    scaffold_metric(args.type, goal, criteria, out_path)

    if args.goal or args.criteria:
        config.goal = goal
        config.criteria = criteria
        config.save(project_root / DEFAULT_CONFIG_NAME)

    print(f"Drafted a {args.type!r} metric at {out_path}.")
    print("Review it, edit the scoring logic to match your real criteria, then approve by running optimize.")
    print(f"(available templates: {list(TEMPLATES)})")


def cmd_optimize(args: argparse.Namespace) -> None:
    from .runner import run_optimization  # deferred: gepa import is heavy and optional until needed

    project_root = Path(args.path)
    config = ProjectConfig.load(project_root / DEFAULT_CONFIG_NAME)

    if args.max_metric_calls is not None:
        config.max_metric_calls = args.max_metric_calls

    print(f"Running GEPA optimization (budget: {config.max_metric_calls} metric calls)...")
    result = run_optimization(config, project_root=project_root)
    print(f"\nDone. Best score: {result.best_score:.4f} (candidate {result.best_idx} of {result.num_candidates})")
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

    p_metric = sub.add_parser("suggest-metric", help="Scaffold a scoring function for review/approval")
    p_metric.add_argument("--type", default="exact_match", choices=list(TEMPLATES))
    p_metric.add_argument("--goal")
    p_metric.add_argument("--criteria")
    p_metric.add_argument("--force", action="store_true")
    p_metric.set_defaults(func=cmd_suggest_metric)

    p_opt = sub.add_parser("optimize", help="Run the GEPA loop")
    p_opt.add_argument("--max-metric-calls", type=int, default=None)
    p_opt.set_defaults(func=cmd_optimize)

    p_report = sub.add_parser("report", help="Print the summary report from the last run")
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
