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

**Narrate every phase, not just the two hard gates.** Before starting a
phase, say in one line what you're about to do and why. After finishing
one, say in one or two lines what happened before moving to the next —
don't silently do several minutes of work (reading a workflow file,
scaffolding a metric, running a search) and only speak again once
everything is completely done. The user should always be able to tell
which phase you're in and what you just found, not just see a final
report at the end. Phases 0 and 3 are *approval* gates (stop and wait for
an explicit answer); the others still get a visible status update, just
not a blocking one.

## Prerequisites

The user has (or will point you at) a project folder containing:
- a dataset of examples (JSONL/JSON/CSV),
- the prompt or workflow they want optimized (a plain prompt file, or an
  existing `workflow.yaml`, or something else — e.g. a Dify export — that
  you'll help translate; see "Non-native workflow sources" below).

If the folder has no `gepa.config.yaml` yet, this is a new project. If one
*does* exist, don't assume where things stand — run `gepa-opt status`
first (works at any point in the journey) to see what's wired up, what's
still a default placeholder, and whether a run has already happened, so
you pick up where things actually are instead of re-deriving it from
scratch or re-asking things already settled.

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

Then edit the generated `workflow.yaml` yourself — don't hand the user a
raw, unfilled scaffold to complete themselves — so it actually represents
their system: one node per LLM call, in order, with the target node(s)
marked `optimize: true` and everything else `optimize: false`. Reuse the
user's real prompt text as the seed — the whole point is evolving *their*
prompt, not a placeholder.

Point `dataset` at their real file. If it's CSV/JSON instead of JSONL, the
loader already handles that — just update `dataset:` in `gepa.config.yaml`.

Once wired up, show the user the resulting `workflow.yaml` (or a summary
of the node graph: what's the target, what's held fixed) and confirm it
actually represents their system before moving to Phase 3 — editing it
yourself means doing the work, not skipping the check-in. A wrong node
marked as the target, or the wrong text reused as the seed, wastes the
entire run that follows.

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
side-effects from other nodes) — see `dev/docs/adapters.md` if they want the
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
   "likely categorical" in `dev/examples/hallucination_judge_demo` just
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
5. Show the user the metric **and** translate it back into plain English
   before asking for approval — 2–3 sentences covering what counts as a
   pass, what counts as a hard fail regardless of anything else, and how
   the two kinds of mistake (if any) are weighted differently. Asking
   someone to "approve" a Python function is only a real approval if they
   actually understood what it checks; showing the code alone and asking
   "does this look right?" isn't that, even for a technical user skimming
   fast. For example: *"This gives full credit only if the output is
   exactly GROUNDED or HALLUCINATED. Missing a real hallucination scores
   zero; a false alarm on a grounded answer scores partial credit (0.4) —
   so it'll push harder to catch real hallucinations than to avoid false
   alarms. Does that match what you want?"* Ask for explicit approval or
   changes. If they ask for changes, edit and re-confirm — including the
   plain-English summary again, since the thing they're approving changed.

## Phase 4 — Run it

First, a cheap pre-flight check — don't skip this, several real bugs in
this project's own history only surfaced mid-run, after spending real
budget, because nothing checked the plumbing first:

```bash
gepa-opt validate
```

This runs the approved metric against a couple of real rows. If it fails,
fix the problem (missing dependency, wrong field name, bad `judge_lm`
wiring) and re-validate before going further — don't proceed to a real run
on a known-broken metric.

Next, get a rough estimate and tell the user what they're actually
committing to before they commit to it — don't just say "this might take a
while":

```bash
gepa-opt estimate --max-metric-calls <budget>
```

This prints a rough LM-call count and a time range (explicitly a rough
structural estimate, not a guarantee or a real dollar cost — it doesn't
call out to any pricing data). Relay it plainly: "~N calls, roughly M–M'
minutes" is enough for the user to make an informed call on the budget
before anything real starts.

Then run it:

```bash
gepa-opt optimize --max-metric-calls <budget>
```

Default budget (150) is reasonable for a first run on a small-to-medium
dataset; scale down for a quick smoke test (e.g. 20-30) or up if the user
wants a thorough search. Requires `ANTHROPIC_API_KEY` (or whatever provider
`task_lm`/`reflection_lm` in `gepa.config.yaml` need) to be set — check for it
and ask if missing, don't just fail silently into a wall of tracebacks.

While it's running, `run_dir/progress.log` gets one clean, regular line
per iteration (metric calls used so far, candidates found, best score) —
tail or periodically re-read that file and relay a short update ("iteration
3, 40/150 calls used, best score 0.82 so far") rather than going silent
until it's completely done; don't rely on eyeballing the verbose
`run_log.txt` for this, `progress.log` is the one meant for exactly this.
Confirm clearly when it finishes, before moving to Phase 5.

If `runs/latest/gepa_state.bin` already exists from a previous run,
`gepa-opt optimize` refuses and asks you to choose explicitly: `--resume`
to continue that run, or `--fresh` to archive it and start over. Don't
guess on the user's behalf here — ask which they want; silently resuming
served stale pre-bug-fix results earlier in this project's own history,
and silently archiving could throw away a run they wanted to keep.

Merge (crossover between two Pareto-frontier candidates, on top of
reflective mutation) is on by default — no action needed, but worth
knowing about if the user asks why a candidate has two parents instead of
one in the results.

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
- `report.md` — read this and present its contents conversationally. Its
  "Best prompt(s) vs. seed" section leads with a word-level diff (what
  actually changed, not two full blocks to eyeball), with the full seed
  and evolved text available below it. Its "All candidates tried" section
  lists *every* candidate GEPA explored, each with a diff from its
  immediate parent — this is the literal answer to "can the user see
  every prompt generated": yes, here, not just the seed and the winner.
- `best_candidate.json` — the winning prompt(s) alone.
- `result.json` — every candidate tried, with lineage and scores, as raw
  JSON (`report.md`'s "All candidates tried" is the readable version of
  this).
- `report.html` — the exact same report as `report.md`, as a self-contained
  styled HTML page (no external dependencies, no network calls). This is
  the portable version of "publish a nicer page than a wall of chat
  markdown" — it's generated by the plain `gepa_optimizer` package itself
  (`report.py`), so *any* harness produces it just by running `optimize`,
  not only one with access to a specific publishing tool. Point the user
  at this file to open in a browser as the default "nicer delivery"
  option.
- `candidate_tree.html` — interactive lineage view; hovering a node shows
  its full prompt text.

Give the user:
1. The best prompt, clearly, ready to paste back into their real system —
   named by its role in the workflow (Phase 0's mental model), not just
   "the prompt" (e.g. "your reply-drafting node's new system prompt").
2. A short narrative of what changed and why (pull this from the top
   candidates' score deltas and feedback, not just "it got better") — the
   word-diff in `report.md` is good material for this, quote it rather
   than re-describing the change in your own words.
3. Where the full cache lives, so they can dig into any candidate that
   didn't win — point specifically at the "All candidates tried" section,
   since that's the direct answer if they ask to see every approach tried.

`report.html` already covers "a nicer page than a wall of chat markdown"
without depending on any harness-specific feature. If this session
specifically has a publishing tool available (e.g. Claude Code's Artifact
tool), publishing it for a shareable link is a reasonable extra on top —
but it's an addition, not the mechanism this depends on; don't build a
one-off page by hand when `report.html` already exists.

## Notes for maintainers of this skill

This file is the Claude-Code-specific trigger wrapper. The actual
intelligence — reading a dataset, drafting a metric, deciding what "good"
feedback looks like — is just good judgment applied to the `gepa-opt` CLI's
inputs/outputs, not anything wired into Claude Code's tool-use loop
specifically. Porting this journey to a different agent harness means
rewriting this playbook in whatever instruction format that harness uses;
the underlying package (`src/gepa_optimizer/`) does not change.
