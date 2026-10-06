---
name: gepa-optimizer
description: >
  Use when the user wants to optimize a prompt that lives inside a larger
  workflow or agent system, against a real dataset, using GEPA (reflective,
  Pareto-based prompt evolution). Triggers: "optimize this prompt with GEPA",
  "run GEPA on my workflow", "tune my agent's prompt against my dataset",
  "evolve this system prompt", "improve this prompt using my examples",
  "set up a GEPA optimization". Drives the `gepa-opt` CLI in this repo
  through the full journey: clarify the goal/target/constraints, draft and
  get approval on a scoring function, run the optimization loop, and hand
  back the best prompt with full run history and a readable summary.
---

# GEPA Optimizer: guided journey

This skill is the conversational front end to the `gepa_optimizer` Python
package in this repo (`gepa-opt` CLI). The package is agent-agnostic by
design — it just needs a workflow file, a dataset, and a scoring function.
This skill is what makes *using* it feel like a conversation instead of
hand-writing YAML and flags. Follow these phases in order; do not skip the
approval gates.

## Prerequisites

The user has (or will point you at) a project folder containing:
- a dataset of examples (JSONL/JSON/CSV),
- the prompt or workflow they want optimized (a plain prompt file, or an
  existing `workflow.yaml`, or something else — e.g. a Dify export — that
  you'll help translate; see "Non-native workflow sources" below).

If the folder has no `gepa.config.yaml` yet, this is a new project.

## Phase 1 — Understand the ask

Ask (conversationally, not as a form) for whatever isn't already obvious
from context:
- **Goal**: what "better" means for this prompt, in plain language.
- **Target**: which part of the workflow should actually change. If there's
  more than one LLM call in the system, get the user to point at the one
  (or the few) they want evolved — everything else stays fixed but still
  executes, so context flows correctly.
- **Criteria**: concrete, checkable conditions (format, required content,
  tone, length, must/must-not statements).
- **Constraints**: hard limits that should never be violated regardless of
  score (e.g. "never change the output JSON schema", "must stay under 200
  tokens", "must not drop the disclaimer sentence"). These need to show up
  as hard-fail conditions in the metric (score 0 + explicit feedback), not
  just as soft preferences, or GEPA may trade them away for a higher score
  elsewhere.

Don't block on perfect answers — reasonable defaults plus a quick confirmation
beat a long intake form.

## Phase 2 — Wire up the project

If `gepa.config.yaml` doesn't exist yet:

```bash
gepa-opt init --goal "<goal>" --criteria "<criteria>"
```

Then edit the generated `workflow.yaml` yourself (don't just hand it to the
user unedited) to actually represent their system: one node per LLM call,
in order, with the target node(s) marked `optimize: true` and everything
else `optimize: false`. Reuse the user's real prompt text as the seed — the
whole point is evolving *their* prompt, not a placeholder.

Point `dataset` at their real file. If it's CSV/JSON instead of JSONL, the
loader already handles that — just update `dataset:` in `gepa.config.yaml`.

### Non-native workflow sources

If the user's "workflow" is actually an export from another platform (e.g. a
Dify DSL YAML) rather than something you can run directly: for this generic
branch, translate it by hand — read the platform file, identify the node(s)
containing the target prompt(s) and how data flows between them, and
represent that same shape in `workflow.yaml`. Tell the user plainly that this
runs the prompt logic standalone (direct LLM calls), not inside their actual
platform runtime (no tools/retrieval/side-effects from other nodes) — see
`docs/adapters.md` if they want the real thing wired up instead of this
translation.

## Phase 3 — Propose a scoring function, then stop for approval

This is a hard gate. Never run optimization against a metric the user hasn't
seen.

1. Look at a sample of the real dataset and the stated criteria/constraints.
2. Pick the closest starting template (`exact_match`, `keyword_presence`, or
   `llm_judge`) and scaffold it:
   ```bash
   gepa-opt suggest-metric --type <type>
   ```
3. **Then rewrite the generated `metric.py` yourself** so the comparison
   logic actually matches the stated criteria and constraints — the
   scaffold is a starting shape, not a finished metric. Encode every hard
   constraint as an explicit check that forces score to 0 with feedback
   naming which constraint broke.
4. Make sure the feedback strings are specific (what was expected, what was
   produced, which rule it violated) — vague feedback produces vague
   mutations; this is the single biggest lever on optimization quality.
5. Show the user the metric (or a summary of its logic) and ask for explicit
   approval or changes before proceeding. If they ask for changes, edit and
   re-confirm.

## Phase 4 — Run it

```bash
gepa-opt optimize --max-metric-calls <budget>
```

Default budget (150) is reasonable for a first run on a small-to-medium
dataset; scale down for a quick smoke test (e.g. 20-30) or up if the user
wants a thorough search. Mention the budget/cost tradeoff to the user before
running anything large. Requires `ANTHROPIC_API_KEY` (or whatever provider
`task_lm`/`reflection_lm` in `gepa.config.yaml` need) to be set — check for it
and ask if missing, don't just fail silently into a wall of tracebacks.

## Phase 5 — Deliver results

Everything lives under `runs/latest/` (per `run_dir` in config):
- `report.md` — read this and present its contents conversationally (best
  prompt vs. seed, score trajectory, Pareto frontier size).
- `best_candidate.json` — the winning prompt(s) alone.
- `result.json` — every candidate tried, with lineage and scores (this is
  the "all the different approaches tried" the user asked for).
- `candidate_tree.html` — interactive lineage view.

Give the user:
1. The best prompt, clearly, ready to paste back into their real system.
2. A short narrative of what changed and why (pull this from the top
   candidates' score deltas and feedback, not just "it got better").
3. Where the full cache lives, so they can dig into any candidate that
   didn't win.

If this session has artifact publishing available, consider turning
`report.md` (plus the lineage tree) into a published summary page rather
than a wall of markdown in chat — check the relevant skill for how to do
that well before building one.

## Notes for maintainers of this skill

This file is the Claude-Code-specific trigger wrapper. The actual
intelligence — reading a dataset, drafting a metric, deciding what "good"
feedback looks like — is just good judgment applied to the `gepa-opt` CLI's
inputs/outputs, not anything wired into Claude Code's tool-use loop
specifically. Porting this journey to a different agent harness means
rewriting this playbook in whatever instruction format that harness uses;
the underlying package (`src/gepa_optimizer/`) does not change.
