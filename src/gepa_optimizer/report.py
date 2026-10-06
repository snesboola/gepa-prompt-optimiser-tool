"""Builds the human-readable summary deliverable: best prompt(s), how the
score moved across every candidate tried, and where to find the full cached
run (result.json + best_candidate.json + candidate_tree.html, written by
runner.run_optimization() right after gepa.optimize() returns).
"""

from __future__ import annotations

import json
from pathlib import Path

from gepa import GEPAResult


def best_score(result: GEPAResult) -> float:
    """gepa==0.1.4's GEPAResult has no .best_score property (that's an
    unreleased main-branch addition) -- derive it the same way its own
    docstring example does."""
    return result.val_aggregate_scores[result.best_idx]


def total_metric_calls(result: GEPAResult) -> int:
    """Likewise, .total_evals doesn't exist yet in 0.1.4."""
    if result.total_metric_calls is not None:
        return result.total_metric_calls
    return sum(result.discovery_eval_counts)


def _fmt_pct(x: float | None) -> str:
    return f"{x:.1%}" if x is not None else "n/a (no positive/negative examples in valset)"


def build_report(
    result: GEPAResult,
    run_dir: str | Path,
    config_summary: dict,
    classification_stats: dict | None = None,
) -> str:
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
    lines.append(f"**Metric calls used:** {total_metric_calls(result)}")
    lines.append(f"**Best candidate index:** {best_idx} (score: {best_score(result):.4f})\n")

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

    if classification_stats is not None:
        seed_stats = classification_stats.get("seed")
        best_stats = classification_stats.get("best")
        lines.append("## Real recall / precision (not the per-example proxy score)\n")
        lines.append(
            "GEPA optimized against a per-example proxy score (asymmetric penalties for "
            "false negatives vs. false positives) because recall/precision can't be computed "
            "from a single example. These are the real, aggregate confusion-matrix numbers for "
            "the seed vs. the winning prompt, on the validation set:\n"
        )
        lines.append("| | Seed | Best |")
        lines.append("|---|---|---|")
        for label, key in [("Recall", "recall"), ("Precision", "precision"), ("F1", "f1")]:
            seed_val = seed_stats.get(key) if seed_stats else None
            best_val = best_stats.get(key) if best_stats else None
            lines.append(f"| {label} | {_fmt_pct(seed_val)} | {_fmt_pct(best_val)} |")
        if best_stats:
            lines.append(
                f"\nConfusion matrix (best): TP={best_stats['tp']}, FP={best_stats['fp']}, "
                f"TN={best_stats['tn']}, FN={best_stats['fn']} "
                f"(positive class: `{best_stats['positive_label']}`)\n"
            )

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
    if (run_dir / "run_log.txt").exists():
        lines.append(
            f"- Full run log (every iteration, every reflection attempt, retries/errors verbatim): "
            f"`{run_dir / 'run_log.txt'}`"
        )
    if (run_dir / "iterations").exists():
        lines.append(f"- Per-iteration traces/components: `{run_dir / 'iterations'}/`")
    if (run_dir / "classification_report.json").exists():
        lines.append(f"- Full recall/precision/F1 breakdown (seed vs. best): `{run_dir / 'classification_report.json'}`")

    return "\n".join(lines)


def write_report(
    result: GEPAResult,
    run_dir: str | Path,
    config_summary: dict,
    classification_stats: dict | None = None,
) -> Path:
    run_dir = Path(run_dir)
    report_path = run_dir / "report.md"
    report_path.write_text(build_report(result, run_dir, config_summary, classification_stats=classification_stats))
    return report_path
