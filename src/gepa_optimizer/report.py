"""Builds the human-readable summary deliverable: best prompt(s), how the
score moved across every candidate tried, and where to find the full cached
run (every candidate/score/trace lives under run_dir because
run_optimization() passes write_agent_state=True to gepa.optimize()).
"""

from __future__ import annotations

import json
from pathlib import Path

from gepa import GEPAResult


def build_report(result: GEPAResult, run_dir: str | Path, config_summary: dict) -> str:
    run_dir = Path(run_dir)
    seed_candidate = result.candidates[0]
    best_candidate = result.best_candidate
    best_idx = result.best_idx

    lines: list[str] = []
    lines.append("# GEPA Optimization Report\n")

    if config_summary.get("goal"):
        lines.append(f"**Goal:** {config_summary['goal']}\n")
    if config_summary.get("criteria"):
        lines.append(f"**Criteria:** {config_summary['criteria']}\n")

    lines.append(f"**Candidates explored:** {result.num_candidates}")
    lines.append(f"**Metric calls used:** {result.total_evals}")
    lines.append(f"**Best candidate index:** {best_idx} (score: {result.best_score:.4f})\n")

    lines.append("## Score trajectory\n")
    lines.append("| Candidate | Parent(s) | Val Score |")
    lines.append("|---|---|---|")
    for i, score in enumerate(result.val_aggregate_scores):
        parents = result.parents[i] if i < len(result.parents) else []
        marker = " **<- best**" if i == best_idx else ""
        lines.append(f"| {i} | {parents} | {score:.4f}{marker} |")
    lines.append("")

    lines.append("## Best prompt(s) vs. seed\n")
    for component, text in best_candidate.items():
        seed_text = seed_candidate.get(component, "")
        lines.append(f"### `{component}`\n")
        if text == seed_text:
            lines.append("_Unchanged from seed._\n")
        else:
            lines.append("**Seed:**")
            lines.append(f"```\n{seed_text}\n```")
            lines.append("**Evolved:**")
            lines.append(f"```\n{text}\n```\n")

    frontier = result.per_val_instance_best_candidates
    lines.append("## Pareto frontier\n")
    lines.append(
        f"{len(set().union(*frontier.values()) if frontier else set())} distinct candidate(s) "
        f"are best-on-at-least-one validation example across {len(frontier)} example(s)."
    )
    lines.append("")

    lines.append("## Cached run data\n")
    lines.append(f"- Full candidate pool + scores: `{run_dir / 'result.json'}`")
    lines.append(f"- Best candidate only: `{run_dir / 'best_candidate.json'}`")
    lines.append(f"- Interactive lineage tree: `{run_dir / 'candidate_tree.html'}`")
    if (run_dir / "iterations").exists():
        lines.append(f"- Per-iteration traces/components: `{run_dir / 'iterations'}/`")

    return "\n".join(lines)


def write_report(result: GEPAResult, run_dir: str | Path, config_summary: dict) -> Path:
    run_dir = Path(run_dir)
    report_path = run_dir / "report.md"
    report_path.write_text(build_report(result, run_dir, config_summary))
    return report_path
