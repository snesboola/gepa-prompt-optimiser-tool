"""Builds the human-readable summary deliverable: best prompt(s), how the
score moved across every candidate tried, and where to find the full cached
run (result.json + best_candidate.json + candidate_tree.html, written by
runner.run_optimization() right after gepa.optimize() returns).
"""

from __future__ import annotations

import difflib
import html
import json
import re
from pathlib import Path

from gepa import GEPAResult


def word_diff_markdown(old: str, new: str) -> str:
    """Word-level diff rendered as markdown (~~removed~~, **added**, plain
    unchanged) -- so "what changed" between two prompts is readable at a
    glance instead of eyeballing two full text blocks."""
    old_words, new_words = old.split(), new.split()
    matcher = difflib.SequenceMatcher(a=old_words, b=new_words, autojunk=False)
    parts: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            parts.append(" ".join(old_words[i1:i2]))
        elif tag in ("delete", "replace"):
            if old_words[i1:i2]:
                parts.append(f"~~{' '.join(old_words[i1:i2])}~~")
            if tag == "replace" and new_words[j1:j2]:
                parts.append(f"**{' '.join(new_words[j1:j2])}**")
        elif tag == "insert":
            if new_words[j1:j2]:
                parts.append(f"**{' '.join(new_words[j1:j2])}**")
    return " ".join(p for p in parts if p)


_DETAILS_SUMMARY_RE = re.compile(r"^<details><summary>(.*)</summary>$")


def _inline_html(text: str) -> str:
    """Escape, then apply the small set of inline markers build_report()
    actually produces -- escaping first is safe since **, ~~, ` aren't
    touched by html.escape(), and it means any literal HTML-looking text
    inside a prompt doesn't get interpreted as real markup."""
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"~~(.+?)~~", r"<del>\1</del>", text)
    text = re.sub(r"`([^`]+?)`", r"<code>\1</code>", text)
    return text


def _render_table_html(rows: list[str]) -> str:
    def split_row(r: str) -> list[str]:
        return [c.strip() for c in r.strip().strip("|").split("|")]

    header = split_row(rows[0])
    data_rows = [split_row(r) for r in rows[2:]] if len(rows) > 2 else []  # rows[1] is the |---|---| separator
    out = ["<table>", "<tr>" + "".join(f"<th>{_inline_html(h)}</th>" for h in header) + "</tr>"]
    for r in data_rows:
        out.append("<tr>" + "".join(f"<td>{_inline_html(c)}</td>" for c in r) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


def markdown_to_html_body(markdown_text: str) -> str:
    """Narrow, purpose-built converter for exactly the markdown constructs
    build_report() produces (headers, **bold**, ~~strike~~, `code`, fenced
    code blocks, pipe tables, and literal <details><summary>/</details>
    passthrough) -- not a general markdown parser. Deliberately not a
    second copy of the report's actual content/structure logic: this only
    ever transforms build_report()'s already-assembled markdown string, so
    there's one source of truth for what the report says, not two that can
    drift apart.
    """
    lines = markdown_text.split("\n")
    out: list[str] = []
    table_buffer: list[str] = []
    i = 0

    def flush_table() -> None:
        if table_buffer:
            out.append(_render_table_html(table_buffer))
            table_buffer.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_table()
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1
            out.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
            continue

        if stripped.startswith("|"):
            table_buffer.append(stripped)
            i += 1
            continue
        flush_table()

        details_match = _DETAILS_SUMMARY_RE.match(stripped)
        if details_match:
            out.append(f"<details><summary>{_inline_html(details_match.group(1))}</summary>")
        elif stripped in ("</details>",):
            out.append(stripped)
        elif stripped.startswith("### "):
            out.append(f"<h3>{_inline_html(stripped[4:])}</h3>")
        elif stripped.startswith("## "):
            out.append(f"<h2>{_inline_html(stripped[3:])}</h2>")
        elif stripped.startswith("# "):
            out.append(f"<h1>{_inline_html(stripped[2:])}</h1>")
        elif stripped == "":
            pass  # paragraph break -- nothing to emit, blocks are already separate elements
        else:
            out.append(f"<p>{_inline_html(stripped)}</p>")
        i += 1

    flush_table()
    return "\n".join(out)


_HTML_PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>GEPA Optimization Report</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 860px;
          margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }}
  h1, h2, h3 {{ margin-top: 2rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  th, td {{ border: 1px solid #8884; padding: 0.4rem 0.7rem; text-align: left; }}
  th {{ background: #8881; }}
  pre {{ background: #8881; padding: 0.8rem; border-radius: 6px; overflow-x: auto; }}
  code {{ background: #8881; padding: 0.1rem 0.3rem; border-radius: 4px; }}
  pre code {{ background: none; padding: 0; }}
  del {{ opacity: 0.6; }}
  details {{ border: 1px solid #8884; border-radius: 6px; padding: 0.5rem 0.8rem; margin: 0.6rem 0; }}
  summary {{ cursor: pointer; font-weight: 600; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def build_report_html(
    result: GEPAResult,
    run_dir: str | Path,
    config_summary: dict,
    classification_stats: dict | None = None,
    held_out_stats: dict | None = None,
) -> str:
    """The same report as build_report(), as a self-contained static HTML
    file -- no external dependencies, no network calls, no dependency on
    any particular coding agent's publishing mechanism. Any harness can
    generate this (it's just a file write, like candidate_tree.html
    already is); opening it just needs a browser.
    """
    markdown_text = build_report(
        result, run_dir, config_summary, classification_stats=classification_stats, held_out_stats=held_out_stats
    )
    return _HTML_PAGE_TEMPLATE.format(body=markdown_to_html_body(markdown_text))


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
    held_out_stats: dict | None = None,
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
            lines.append("(Unchanged from seed.)\n")
        else:
            lines.append(f"**What changed:** {word_diff_markdown(seed_text, text)}\n")
            lines.append("<details><summary>Full text (seed vs. evolved)</summary>\n")
            lines.append("**Seed:**")
            lines.append(f"```\n{seed_text}\n```")
            lines.append("**Evolved:**")
            lines.append(f"```\n{text}\n```")
            lines.append("</details>\n")

    lines.append("## All candidates tried\n")
    lines.append(
        "Every prompt GEPA actually tried, not just the seed and the winner -- "
        "this is the full \"different approaches\" record.\n"
    )
    for idx, candidate in enumerate(result.candidates):
        parents = [p for p in (result.parents[idx] if idx < len(result.parents) else []) if p is not None]
        score = result.val_aggregate_scores[idx] if idx < len(result.val_aggregate_scores) else None
        role = " **(best)**" if idx == best_idx else (" (seed)" if idx == 0 else "")
        score_str = f"{score:.4f}" if score is not None else "n/a"
        lines.append(f"<details><summary>Candidate {idx}{role} -- score {score_str}, parent(s) {parents or 'none'}</summary>\n")
        if len(parents) == 1:
            parent_candidate = result.candidates[parents[0]]
            for component, text in candidate.items():
                parent_text = parent_candidate.get(component, "")
                if text == parent_text:
                    continue
                lines.append(f"`{component}`: {word_diff_markdown(parent_text, text)}\n")
        else:
            # seed (no parent) or a merge result (two parents) -- a diff
            # against one parent alone would be misleading, show full text
            for component, text in candidate.items():
                lines.append(f"**`{component}`:**")
                lines.append(f"```\n{text}\n```")
        lines.append("</details>\n")

    if classification_stats is not None:
        seed_stats = classification_stats.get("seed")
        best_stats = classification_stats.get("best")
        evaluated_on = classification_stats.get("evaluated_on", "val")
        set_label = "held-out test set (never seen during search)" if evaluated_on == "test" else "validation set"
        lines.append("## Real recall / precision (not the per-example proxy score)\n")
        lines.append(
            "GEPA optimized against a per-example proxy (either asymmetric FN/FP penalties, or "
            "the recall_proxy/precision_proxy objectives tracked via GEPA's own multi-objective "
            "Pareto frontier below) because recall/precision can't be computed from a single "
            f"example. These are the real, aggregate confusion-matrix numbers, on the {set_label}:\n"
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

    if held_out_stats is not None:
        seed_ho = held_out_stats.get("seed", {})
        best_ho = held_out_stats.get("best", {})
        lines.append("## Held-out test performance\n")
        lines.append(
            f"Mean score on {best_ho.get('n', '?')} rows GEPA never saw during search (not "
            "trainset, not valset) -- the actual check against overfitting to the validation set:\n"
        )
        lines.append("| | Seed | Best |")
        lines.append("|---|---|---|")
        lines.append(
            f"| Mean score | {seed_ho.get('mean_score', 0):.4f} | {best_ho.get('mean_score', 0):.4f} |"
        )
        lines.append("")

    frontier = result.per_val_instance_best_candidates
    lines.append("## Pareto frontier\n")
    lines.append(
        f"{len(set().union(*frontier.values()) if frontier else set())} distinct candidate(s) "
        f"are best-on-at-least-one validation example across {len(frontier)} example(s)."
    )
    if result.objective_pareto_front:
        lines.append("\nPer-objective frontier (GEPA's native multi-objective tracking):\n")
        for obj_name, obj_score in result.objective_pareto_front.items():
            n_best = len((result.per_objective_best_candidates or {}).get(obj_name, []))
            lines.append(f"- `{obj_name}`: best aggregate = {obj_score:.4f}, {n_best} candidate(s) tied for best")
    lines.append("")

    lines.append("## Cached run data\n")
    lines.append(f"- Full candidate pool + scores: `{run_dir / 'result.json'}`")
    lines.append(f"- Best candidate only: `{run_dir / 'best_candidate.json'}`")
    lines.append(f"- This same report as a styled, self-contained HTML page: `{run_dir / 'report.html'}`")
    lines.append(f"- Interactive lineage tree: `{run_dir / 'candidate_tree.html'}`")
    if (run_dir / "run_log.txt").exists():
        lines.append(
            f"- Full run log (every iteration, every reflection attempt, retries/errors verbatim): "
            f"`{run_dir / 'run_log.txt'}`"
        )
    if (run_dir / "progress.log").exists():
        lines.append(
            f"- Short, regular progress line per iteration (metric calls used, best score so far) -- "
            f"the thing to tail for \"is this still going\": `{run_dir / 'progress.log'}`"
        )
    if (run_dir / "iterations").exists():
        lines.append(f"- Per-iteration traces/components: `{run_dir / 'iterations'}/`")
    if (run_dir / "classification_report.json").exists():
        lines.append(f"- Full recall/precision/F1 breakdown (seed vs. best): `{run_dir / 'classification_report.json'}`")
    if (run_dir / "held_out_test.json").exists():
        lines.append(f"- Held-out test scores (seed vs. best): `{run_dir / 'held_out_test.json'}`")

    return "\n".join(lines)


def write_report(
    result: GEPAResult,
    run_dir: str | Path,
    config_summary: dict,
    classification_stats: dict | None = None,
    held_out_stats: dict | None = None,
) -> Path:
    run_dir = Path(run_dir)
    markdown_text = build_report(
        result, run_dir, config_summary, classification_stats=classification_stats, held_out_stats=held_out_stats
    )
    report_path = run_dir / "report.md"
    report_path.write_text(markdown_text)
    (run_dir / "report.html").write_text(_HTML_PAGE_TEMPLATE.format(body=markdown_to_html_body(markdown_text)))
    return report_path
