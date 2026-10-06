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

## Phase 0 — Understand the whole workflow first

Before asking the user anything or touching any file, read the *entire*
workflow/agent definition they handed you — not just the node that looks
like the target prompt. This matters most for a platform export (e.g. a
Dify DSL YAML), which encodes a full graph most users won't have described
to you otherwise:

- Read every node (`workflow.graph.nodes`) and every edge between them, the
  app's own `name`/`description` metadata, and any conditional branches,
  retrieval/tool nodes, or code nodes.
- Build an actual mental model: what is this workflow *for* (the end-user
  problem it solves), what role does each node play in solving it, and how
  does data actually flow from entry to final output?
- Specifically work out what feeds *into* the likely target node and what
  consumes its output downstream — a node three steps later expecting
  strict JSON from the target node is a constraint the user may not think
  to mention, but GEPA breaking it silently would make the "optimized"
  prompt useless in production.

State this understanding back in a few plain sentences before proceeding —
e.g. "this looks like a support-ticket triage system: an intent classifier
picks a queue, a retrieval node pulls relevant docs, and a reply-drafting
node (your likely target) writes the response using both." Ask the user to
confirm or correct it. This is cheap to get wrong here and expensive to
discover after burning optimization budget on a misread workflow — treat it
as seriously as the Phase 3 approval gate, even though it's lighter-weight.

Carry this understanding forward: it should sharpen the questions in Phase
1, catch downstream-format constraints for Phase 2's translation and Phase
3's metric, and give you the vocabulary to explain results in Phase 5 in
terms of what the workflow is actually trying to do.

For a plain prompt file or an already-generic `workflow.yaml` with no
surrounding platform graph, this phase is quick — there's less to map out,
but still skim the whole file rather than jumping straight to the prompt
text.

## Phase 1 — Understand the ask

Ask (conversationally, not as a form) for whatever isn't already obvious
from context — informed by what Phase 0 told you about the workflow, not as
a generic intake form:
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
branch, translate it by hand using the full mental model you built in Phase
0, not just the target node in isolation. Represent that same shape in
`workflow.yaml` — fixed upstream nodes feeding the target with real
variable names, the target itself, and (if relevant to scoring) a
downstream node that shows what format its output actually needs to be in.
Tell the user plainly that this runs the prompt logic standalone (direct
LLM calls), not inside their actual platform runtime (no tools/retrieval/
side-effects from other nodes) — see `docs/adapters.md` if they want the
real thing wired up instead of this translation.

## Phase 3 — Propose a scoring function, then stop for approval

This is a hard gate. Never run optimization against a metric the user hasn't
seen.

1. Actually inspect the dataset before assuming anything about its
   structure — don't guess column names:
   ```bash
   gepa-opt inspect-dataset
   ```
   This prints every real column, a cheap distinct-value-count heuristic
   flagging some as "likely categorical," and a few sample rows. Treat
   that heuristic as a weak hint, not an answer — it has no idea what a
   column *means*, only how many distinct values it happens to have, and
   it's already known to misfire (it flagged a repeated context passage as
   "likely categorical" in `examples/hallucination_judge_demo` just
   because the demo dataset was small enough that a few context strings
   repeated). You're the one who should actually read the column names and
   sample values and reason about which one is the label/reference/
   whatever-field, the same way you'd read any other data — that's exactly
   the kind of judgment call a fixed statistical threshold can't make but
   you can (recognizing "verdict" or "is_fraud" or "ground_truth" as a
   label column from its name and values, independent of how many distinct
   values it has). Cross-reference this with the stated criteria/
   constraints and what Phase 0 told you about what consumes this node's
   output — a downstream format requirement is a hard constraint even if
   the user never said it explicitly.
2. Pick the closest starting template and scaffold it:
   ```bash
   gepa-opt suggest-metric --type <type>
   ```
   - `exact_match` / `keyword_presence`: a known-correct answer exists.
   - `llm_judge`: open-ended output, no single correct answer, scored
     against stated criteria via a separate judge call.
   - `classification`: the goal is phrased in terms of recall/precision
     (e.g. "catch most real hallucinations without too many false alarms"),
     not raw accuracy. Say so explicitly to the user if you choose this one
     instead of `exact_match`: GEPA scores one row at a time, so recall/
     precision (aggregate, whole-dataset numbers) can't be the literal
     per-example signal — this template scores each row with *asymmetric*
     penalties (missing a positive costs more than a false alarm, by
     default) as the closest thing GEPA can act on per example, and the
     run separately reports the *real* recall/precision/F1 for the seed vs.
     the winning prompt in `report.md`'s "Real recall / precision" section
     (via `metrics.classification_report()`, wired in `runner.py`) — that's
     where the user sees the actual number, not the proxy.
   - `composite`: criteria mix a hard, checkable requirement (must mention
     an exact phrase, must include specific terms) with a vaguer
     qualitative one (must "capture industry detail", must "sound
     professional") *in the same output*. Don't pick just one template for
     this — the hard part needs an exact check, the soft part needs a
     judge call, and they need to combine so the hard check can't be
     traded away for a nicer-sounding response. `composite` does exactly
     that: fails immediately (score 0, judge never even consulted) if a
     required phrase is missing, and only once that gate passes does
     `judge_lm` score the qualitative part.
3. **Then rewrite the generated `metric.py` yourself** so the comparison
   logic actually matches the stated criteria and constraints — the
   scaffold is a starting shape, not a finished metric. Encode every hard
   constraint as an explicit check that forces score to 0 with feedback
   naming which constraint broke. This includes field names:
   `classification`/`exact_match`/`keyword_presence` scaffold a real,
   used constant (`LABEL_FIELD`/`REFERENCE_FIELD`/`REQUIRED_KEYWORDS_FIELD`)
   defaulting to a guessed name ("label", "reference", ...) — set it to
   whatever you determined the real column is called in step 1, from your
   own reading of the data, not from the tool's categorical-heuristic
   suggestion alone. `suggest-metric` will warn if the default name isn't
   in the dataset's columns and list the heuristic's candidates, but that
   warning is a safety net for catching an *unedited* scaffold, not a
   substitute for you actually having looked.
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

For a small dataset (≤25 training rows), `runner.py` already defaults the
reflection minibatch to the *whole* training set rather than GEPA's own
3-row sample — no action needed, but worth knowing if you're explaining why
results look stable rather than noisy run to run. For a larger dataset,
`gepa.config.yaml`'s `reflection_minibatch_size` field can be set explicitly
if the default (GEPA's own sampling) isn't giving a stable enough signal.

Two more config knobs, both off by default, worth raising if the user asks:
- `test_fraction` (default `0.0`): carves out a held-out split GEPA never
  searches against, for an honest seed-vs-best sanity check in the final
  report. Worth turning on once the dataset has enough rows that train/val/
  test can each still mean something (a few dozen total, at least) — don't
  suggest it on a tiny demo-sized dataset, it'll leave almost nothing in
  each split.
- `max_workers` (default `1`): runs a batch's rows concurrently above 1.
  Only raise this if the user confirms their provider's rate limits can
  take it — this session hit real `429` quota errors from *sequential*
  calls alone; don't make that worse by suggesting concurrency as a
  default speedup.

If a run's score looks suspiciously flat at 0.0 across every candidate,
check `run_log.txt` for a `RuntimeError` about every example in a batch
raising an exception before assuming the prompt itself is just bad —
`adapter.evaluate()` raises loudly in that case specifically so it isn't
mistaken for a legitimately low score.

## Phase 5 — Deliver results

Everything lives under `runs/latest/` (per `run_dir` in config):
- `report.md` — read this and present its contents conversationally (best
  prompt vs. seed, score trajectory, Pareto frontier size).
- `best_candidate.json` — the winning prompt(s) alone.
- `result.json` — every candidate tried, with lineage and scores (this is
  the "all the different approaches tried" the user asked for).
- `candidate_tree.html` — interactive lineage view.

Give the user:
1. The best prompt, clearly, ready to paste back into their real system —
   named by its role in the workflow (Phase 0's mental model), not just
   "the prompt" (e.g. "your reply-drafting node's new system prompt").
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
