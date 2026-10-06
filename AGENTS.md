# AGENTS.md

North-star reference for any agent (Claude Code, a future internal harness,
or a human) working on this repo. If a change contradicts this doc, either
the change is wrong or this doc is stale — fix whichever is actually true,
don't let them silently drift apart.

## The original vision

This is the request that started the project (lightly cleaned up for typos,
meaning unchanged):

> 1. User has a folder with a dataset, as well as the Dify YAML file/code
>    file of the agent they want to optimize.
> 2. User gives the goal of their optimization, some criteria they want met,
>    which part of the workflow they want to optimize, and any constraints
>    they want to adhere to.
> 3. The skill suggests a scoring function/criteria; user approves.
> 4. GEPA runs on the selected prompt recursively.
> 5. User receives the best prompt, all the cached runs, all the different
>    approaches tried, and a summary presentation.

Everything in this repo exists to serve that five-step journey. Nothing
else.

## What kinds of prompts this optimizes

Not just verifiable QA/extraction tasks. Explicitly in scope, same
architecture for all three:

- **Verifiable tasks** (QA, extraction, classification) — a known-correct
  answer exists per row; metric is `exact_match`/`keyword_presence` against
  a `reference` field. `examples/qa_demo` demonstrates this.
- **LLM-as-judge prompts** — the prompt *being optimized* is itself a judge
  (evaluates/scores other content). Dataset rows are labeled calibration
  examples (content + known-correct verdict); metric compares the judge's
  verdict to that label with plain `exact_match`/`keyword_presence` — no
  recursion into another judge call needed.
  `examples/hallucination_judge_demo` demonstrates this (optimizing a
  hallucination-detection judge against labeled grounded/hallucinated rows);
  verified logically with fake stand-in judges -- not yet run against a real
  model (blocked on today's Gemini free-tier quota, see below).

  **If the real goal is recall/precision** (e.g. "catch most real
  hallucinations, don't cry wolf too often"), not raw accuracy: recall and
  precision are aggregate confusion-matrix numbers, not something any
  single `score(row, trace)` call can produce -- GEPA scores one row at a
  time. What actually works and is built in: the `classification` metric
  template (`gepa-opt suggest-metric --type classification`) scores each row
  with *asymmetric* penalties -- a missed positive (false negative) costs
  more than a false alarm (false positive) by default, tunable via
  `FN_PENALTY`/`FP_PENALTY` -- which is the per-example signal GEPA can
  actually act on to push toward higher recall without ignoring precision.
  Separately, `metrics.classification_report()` re-evaluates the seed and
  the winning candidate over the full valset after optimization and
  computes the *real* TP/FP/TN/FN/recall/precision/F1 (wired into
  `runner.py`, rendered in `report.md`'s "Real recall / precision" section)
  -- so the user sees the actual number they asked about, not just trust
  that the proxy score correlates with it. Verified end-to-end with fake
  judges: a seed that always says "grounded" scores recall=0.0; a stricter
  judge scores recall=1.0, precision=0.8, F1=0.89 on the same 8-row dataset.

  The per-example proxy's main weakness is noise: GEPA's own minibatch
  default (3 rows per reflection step) is a random sample, and a 3-row
  sample of a recall-sensitive metric frequently contains zero
  positive-labeled rows at all, making the sampled signal unstable. Good
  question from the user cut right to this: *if the dataset is small,
  why not just use the whole thing as the minibatch every time?* -- and
  that's exactly right, and wireable: `reflection_minibatch_size` is a real
  `gepa.optimize()` parameter we simply weren't passing. `runner.py` now
  defaults it to the full training set when `len(trainset) <=
  SMALL_DATASET_THRESHOLD` (25), overridable via
  `gepa.config.yaml`'s `reflection_minibatch_size` field. Verified against
  the real installed `EpochShuffledBatchSampler`
  (`reference/gepa-ai-gepa/src/gepa/strategies/batch_sampler.py`): when
  `minibatch_size == trainset_size`, every single call returns all rows,
  every iteration -- confirmed directly, not just read off the source.
  This doesn't change the mean-vs-ratio math above (the per-row proxy is
  still a proxy, not literal recall/precision), it just removes the sampling
  noise on top of it for small datasets.

  **Beyond the asymmetric-penalty proxy**: when a metric exposes `classify`
  + `POSITIVE_LABEL`, `runner.py` now also uses GEPA's *native*
  multi-objective Pareto tracking (`EvaluationBatch.objective_scores`,
  `frontier_type="hybrid"`) via `metrics.recall_precision_objective_scores()`
  -- a `recall_proxy` (0.0 only on a false negative) and `precision_proxy`
  (0.0 only on a false positive) tracked as two *separate* frontier
  dimensions, instead of collapsing the tradeoff into one hand-weighted
  scalar. `recall_proxy` is provably sound for a fixed valset: since the
  count of positive-labeled rows is fixed across candidates, mean(recall_proxy)
  is an exact affine transform of real recall (same engine aggregation GEPA
  always uses -- a plain mean, confirmed by reading
  `gepa.core.state.GEPAState._aggregate_objective_scores`). `precision_proxy`
  is a reasonable heuristic, not an exact proxy (TP varies per candidate too,
  unlike N_pos for recall) -- `classification_report()` remains the source
  of truth either way. Verified against the real engine, not just our own
  adapter code: a tiny fake-LM run with `frontier_type="hybrid"` produced
  `objective_pareto_front={'recall_proxy': 0.5, 'precision_proxy': 1.0}`
  matching the hand-derived math exactly, and the acceptance logic still
  worked correctly with objective_scores present.
- **Open-ended writing/creative tasks** — no single correct output exists;
  metric is `llm_judge`, scoring against stated criteria via a separate
  `judge_lm` call. Verified working end-to-end (fake task/judge LMs, no API
  calls) for a no-`reference` writing task: candidate evolves a "write X"
  system prompt, `judge_lm` rates the output, score + rationale flow
  correctly into the reflective dataset. No example committed for this path
  yet — worth adding one (e.g. `examples/writing_judge_demo/`) before
  relying on it against a real model.
- **Compound criteria** — a hard, checkable requirement (must mention an
  exact phrase) mixed with a vaguer qualitative one (must "capture industry
  detail") in the *same* output. No single template does both; the
  `composite` metric template composes them instead: a hard phrase-presence
  gate that fails immediately (score 0, judge never consulted) regardless
  of quality elsewhere, and only once that passes does a `judge_lm` call
  score the qualitative part. Verified end-to-end with fake task/judge LMs
  (`tests/test_metrics.py`): an output missing the required phrase scores
  0.0 even when the (fake) judge would have given it 0.95 -- the gate
  genuinely overrides the judge, not just averages with it; a present-phrase
  output then correctly gets judged low for generic filler (0.2) vs. high
  for real specific detail (0.9).

## Engine robustness improvements

- **Dataset structure is actually inspected, not assumed -- and the
  scaffolded templates are genuinely renameable, not just in theory.**
  Two real gaps, caught by the user asking "is this generalisable though?
  what if the column isn't called label": (1) nothing looked at the
  dataset's actual columns before scaffolding a metric, so a mismatched
  field name (e.g. a dataset with `verdict` instead of `label`) would only
  surface as a confusing `KeyError` (or a silently-always-empty `.get()`
  fallback) once a run was already underway; (2) the `classification`
  template's scaffold hardcoded `row["label"]` directly in `score()` even
  though `runner.py` already looked for an optional `LABEL_FIELD`
  override -- so even defining `LABEL_FIELD = "verdict"` wouldn't have
  done anything, the function body never read it. Fixed both: `dataset.
  inspect_dataset()` reports every column's distinct-value count and flags
  likely-categorical ones (a real, if imperfect, heuristic -- verified it
  both correctly flags a true 2-value label column and produces a known,
  documented false positive on a column that merely happens to repeat);
  a new `gepa-opt inspect-dataset` command surfaces this directly; `gepa-
  opt suggest-metric` now checks the chosen template's expected field(s)
  against the real columns *before* scaffolding, and when they don't
  match, suggests the actual likely-categorical column(s) by name instead
  of just flagging the absence. `classification`/`exact_match`/
  `keyword_presence` templates all now define their field name as a real,
  used top-level constant (`LABEL_FIELD`, `REFERENCE_FIELD`,
  `REQUIRED_KEYWORDS_FIELD`) that `score()` actually reads, not a hardcoded
  string literal -- renaming is a one-line edit, verified by scaffolding
  and compiling all three.

  **Deliberately not automatic.** The follow-up question ("i want the
  skill/coding harness do these kind of data discovery as it is more
  generalisable") pushed this further and correctly: a fixed distinct-
  value-count threshold has no semantic understanding of what a column
  *means* -- it can't tell "verdict" or "is_fraud" or "ground_truth" is a
  label column from its name, and it's already a known false positive
  generator (flags any small-enough repeated column, like a context
  passage that happens to repeat in a tiny demo dataset, as "categorical").
  An LLM agent reading the same sample rows and column names can use
  actual judgment an arithmetic threshold can't. So `inspect_dataset()`'s
  heuristic is deliberately kept as a weak, cheap fallback signal (useful
  when no agent is driving, or as a sanity-check even when one is), not
  the primary mechanism -- the skill (Phase 3, step 1 and step 3) now says
  explicitly that the agent's own reading of `inspect-dataset`'s output is
  what should decide `LABEL_FIELD`/`REFERENCE_FIELD`/etc., and that
  `suggest-metric`'s warning is a safety net for catching an unedited
  scaffold, not a substitute for actually having looked.

- **Loud failure detection.** A broken `task_lm` call (we hit five
  different ones in one session -- missing deps, a deprecated model, a
  missing adapter attribute) used to get silently folded into a `0.0`
  score by `adapter.evaluate()`'s per-example try/except, indistinguishable
  from "the prompt is just bad." If *every* row in a batch raises a real
  exception (not just scores low), `evaluate()` now raises `RuntimeError`
  naming the first error -- this is the sanctioned case to raise in per
  `GEPAAdapter`'s own contract ("reserved for unrecoverable, systemic
  failures"), and `gepa.optimize()`'s `raise_on_exception` (default `True`)
  governs what happens from there. A partial failure (some rows, not all)
  still doesn't raise -- that's a legitimately mixed result, not a
  systemic break. Covered by `tests/test_workflow.py`.

- **Held-out test split.** `trainset`/`valset` alone has the same
  overfitting risk as a model tuned only against a validation set with no
  separate test set -- GEPA searches against valset, so a prompt could
  overfit to it. `dataset.split_dataset()` now optionally carves out a
  third split (`test_fraction` in `gepa.config.yaml`, default `0.0` =
  off -- opt-in since a 3-way split of an already-small dataset leaves too
  little per split to mean anything) that GEPA never sees during search.
  `runner.py` evaluates seed vs. best on it after optimization and reports
  the real, unbiased mean score in `report.md`'s "Held-out test
  performance" section; when a classification metric is in play, the real
  recall/precision/F1 section also prefers this split over valset when
  available (more rigorous). Verified end-to-end via `run_optimization()`
  with fake LMs on a 20-row synthetic dataset.

- **Parallel per-batch evaluation.** `adapter.evaluate()` used to loop over
  a batch's rows sequentially, one task_lm call at a time, even though the
  actual bottleneck is network latency, not CPU. `max_workers` in
  `gepa.config.yaml` (default `1`, sequential) runs a batch's rows
  concurrently via `ThreadPoolExecutor` when raised. Default stays `1`
  deliberately: this session hit real Gemini free-tier `429` quota errors
  from sequential calls alone (see the bug log below) -- firing several
  requests at once by default would make that worse, not better. Raise it
  once you know your provider's rate limits can take it (a paid tier, a
  higher-limit provider, or a local model). Order of outputs/scores/
  trajectories is preserved regardless of worker count (`tests/
  test_workflow.py::test_adapter_parallel_execution_preserves_order_and_correctness`).

- **Pre-flight metric validation.** `gepa-opt validate` (also run
  automatically inside `run_optimization()` unless `skip_validate=True`)
  runs the approved metric against a couple of real dataset rows before
  committing to a full search. Reuses `adapter.evaluate()`'s own
  loud-failure check rather than duplicating it -- several of this
  project's real bugs this session (wrong field names, missing deps, a
  broken `task_lm`) only surfaced mid-run, after spending real quota;
  this catches the same class of problem for ~2 calls instead.

- **CSV + list-typed fields was a real latent bug, not just a theoretical
  one.** `keyword_presence`/`composite` expect `required_keywords`/
  `required_phrases` to be a Python list. `csv.DictReader` returns every
  cell as a plain string, so a CSV dataset would have silently iterated
  that string character-by-character instead of over keywords -- wrong,
  with no error. `load_dataset()` now decodes a CSV cell as JSON when it
  looks like a list/object (`[`/`{` prefix), falling back to the raw
  string otherwise. Verified: a `["dog", "bark"]`-shaped cell decodes to
  an actual list; a plain string cell that merely starts differently is
  left untouched (`tests/test_dataset.py`).

- **Deliberate resume/fresh control.** `gepa.optimize()` resumes
  automatically from `gepa_state.bin` if `run_dir` already has one --
  this is exactly what served stale, pre-bug-fix results earlier in this
  project's own history (see the bug log below) when a leftover state
  file from a failed run got silently reused. `gepa-opt optimize` now
  refuses outright when that file exists and neither `--resume` nor
  `--fresh` was passed, rather than silently picking a behavior on the
  user's behalf; `--fresh` archives the old run_dir (renamed with a
  timestamp, not deleted) before starting clean. Verified against a real
  run: `--fresh` archived to `runs/latest_archived_<timestamp>` and
  started over; `--resume` continued in place with no archiving and
  printed gepa's own "Loading gepa state from run dir".

- **Word-level diffs and a full candidate list in `report.md`.** Two
  gaps closed together: (1) "seed vs. best" showed two full text blocks
  to eyeball instead of what actually changed; (2) nothing gave a direct
  answer to "can the user see every prompt generated" -- `result.json`
  has every candidate as raw JSON, and `candidate_tree.html` shows full
  text on hover, but nothing presented a readable, linear list of every
  candidate tried. `report.word_diff_markdown()` (stdlib `difflib`, word
  granularity, `~~removed~~`/`**added**` markdown) now drives both the
  seed-vs-best section and a new "All candidates tried" section listing
  every candidate with a diff from its immediate parent (full text
  instead, for the seed or a two-parent merge result, where a single-
  parent diff would be misleading). `tests/test_report.py` covers the
  diff helper directly.

- **Merge enabled by default.** `use_merge=True` (GEPA's crossover
  between two Pareto-frontier candidates, on top of reflective mutation)
  is now a config field, defaulting on. Verified against the real engine
  with a scenario that actually produces more than one frontier
  candidate: the merge proposer engaged and correctly logged "No merge
  candidates found" when there was nothing complementary to combine,
  rather than erroring -- confirms the wiring is correct and safe even
  when a merge opportunity doesn't happen to exist in a given run.

## Decisions made since, and why

These came out of the vision as the architecture got built, each one a
direct answer to a constraint that showed up along the way:

- **Build a real, portable engine, not a Claude-Code-only trick.** The user
  will migrate this to a company environment that uses an internal harness,
  not Claude Code. So the actual optimizer is a standalone Python
  package/CLI (`src/gepa_optimizer/`, the `gepa-opt` command) with zero
  dependency on any particular coding agent's tool-use loop. A conversational
  layer on top (`.claude/skills/gepa-optimizer/SKILL.md`) makes the journey
  feel guided in Claude Code today; porting the *journey* to another harness
  later means rewriting that one instructions file in whatever format that
  harness uses, the engine underneath doesn't change.

- **Use the real upstream `gepa` library (gepa-ai/gepa), not a toy
  reimplementation.** It already has the real Pareto-frontier logic, budget
  management, and a clean adapter contract (`GEPAAdapter.evaluate` +
  `make_reflective_dataset`). `reference/gepa-ai-gepa` is a pinned submodule
  of this, pointing directly at upstream, for citation/reproducibility —
  read for reference, never built on top of or copied from.

- **Generic workflow backend now; Dify (or any other platform) is a
  pluggable seam, not baked in.** `workflow.yaml` + `WorkflowRunner`
  (`src/gepa_optimizer/workflow.py`) is the default, platform-agnostic
  backend — a small chain of LLM-call nodes, run with direct task-LM calls.
  Swapping in a real platform (Dify, LangGraph, an internal agent runtime)
  means subclassing `WorkflowRunner`, documented in `docs/adapters.md`.
  Nothing else changes. The Dify-specific version of this is intentionally
  deferred to a separate branch.

- **The LM itself is just as pluggable, for the same portability reason.**
  `task_lm`/`reflection_lm` resolve (`src/gepa_optimizer/llm.py`) from either
  a plain litellm model string or a `callable:module:attr` spec. Today that's
  a litellm string pointed at Google AI Studio's free tier (no card
  required, see below). The eventual company-harness version is expected to
  be a custom callable that calls a Dify workflow (system prompt + user
  prompt in, text out) instead — a one-function change, nothing upstream of
  it changes.

- **Default to Google AI Studio's free tier for a fresh project.** Testing
  shouldn't require a paid key. `gepa.config.yaml` defaults to
  `gemini/gemini-2.5-flash-lite` (frequent task calls) and
  `gemini/gemini-2.5-flash` (less-frequent reflection calls), both free with
  just a `GEMINI_API_KEY`. Any other litellm-supported provider works too —
  it's one line in config.

- **The skill narrates every phase, not just the two hard gates.** Early
  versions only stopped the user at Phase 0 (confirm the workflow
  understanding) and Phase 3 (approve the metric) — everything else
  happened silently in between. The user asked explicitly for status
  updates throughout, and separately pointed out that Phase 2's "edit
  workflow.yaml yourself" instruction never actually said to *show* the
  user the result before moving on. Both fixed: the skill's intro now
  states the narration expectation up front, Phase 2 gets an explicit
  show-and-confirm step, and Phase 4 explicitly says to announce the run
  starting and finishing rather than going quiet for its duration.

## Where each vision step actually lives

| Step | Where |
|---|---|
| 1. Dataset + workflow folder | `dataset.py` (JSONL/JSON/CSV loader) + `workflow.yaml` (generic now; a real Dify export is read in full and understood -- architecture, data flow, intent, not just the target node -- then translated by hand, per the skill's Phase 0 and "Non-native workflow sources" section; automatic Dify parsing is the deferred branch) |
| 2. Goal/criteria/target/constraints | Gathered conversationally by the skill (Phase 1), informed by Phase 0's understanding of the workflow (sharper questions, catches downstream-format constraints the user might not think to state); stored in `gepa.config.yaml`'s `goal`/`criteria` fields, expressed as `optimize: system\|user` on the target node(s) |
| 3. Scoring function, proposed then approved | `gepa-opt suggest-metric` scaffolds (`metrics.py`); the skill rewrites it to actually match the stated criteria/constraints and stops for explicit approval before running anything (Phase 3 — a hard gate, never skipped) |
| 4. GEPA runs recursively | `runner.py` → the real `gepa.optimize()`, via `WorkflowGEPAAdapter` (`adapter.py`) |
| 5. Best prompt + cached runs + approaches + summary | `run_dir` (`runs/latest/`): `best_candidate.json`, `result.json` (every candidate + lineage + score), `candidate_tree.html`, and `report.md` (the human-readable summary, built by `report.py`). `write_agent_state` (a per-iteration `iterations/` trace tree) exists upstream only on `gepa`'s unreleased `main` branch, not the installed `0.1.4` -- see the "real bug log" note below |

## Real bug log

Caught by actually running things, not by inspection — kept here so the next
session doesn't rediscover the same gap from scratch:

- **`workflow.py` field-targeting bug** (fixed): a node with both `system`
  and `user` text plus bare `optimize: true` silently overwrote *both*
  fields with the same candidate text, destroying the original system
  prompt. Fixed by requiring `optimize: system|user` when both fields are
  present; `optimize: true` now raises instead of guessing. Covered by
  `tests/test_workflow.py`.
- **`write_agent_state` doesn't exist in the real PyPI package**: I'd read
  `gepa`'s source off GitHub's `main` branch, which is ahead of what's
  actually published (`0.1.4` is the latest on PyPI as of this writing).
  `runner.py` passed `write_agent_state=True` to `gepa.optimize()`, which
  raised `TypeError` the first time it was actually run against the real
  installed library. Removed; `result.to_dict()` / `result.candidate_tree_html()`
  (already used) cover the same "every candidate on disk" need without it.
  **Lesson**: reading a dependency's GitHub source is not the same as
  checking what version is actually installable — verify against
  `inspect.signature()` on the real installed package, not just the repo.
- **`result.best_score` / `result.total_evals` don't exist in `0.1.4`
  either** (same ahead-of-release issue as above). Added `best_score()` /
  `total_metric_calls()` helpers in `report.py` (derived from
  `val_aggregate_scores[best_idx]` and `total_metric_calls`/
  `discovery_eval_counts`), used from `runner.py`, `cli.py`, `report.py`
  instead of the nonexistent properties.
- **`WorkflowGEPAAdapter` had no `propose_new_texts` attribute**:
  `GEPAAdapter` is a `Protocol`, not an ABC — its documented
  `propose_new_texts: ProposalFn | None = None` default is not inherited by
  an implementing class. `gepa`'s engine reads `self.adapter.propose_new_texts`
  unconditionally, so every reflection attempt raised `AttributeError` until
  `propose_new_texts = None` was added explicitly as a class attribute.
- **Missing `litellm`/`tenacity`/`tqdm` dependencies**: `gepa` itself
  declares *zero* dependencies (bring-your-own LM backend by design); our
  own `workflow.py` imports `litellm` directly, and `display_progress_bar=True`
  needs `tqdm`. `litellm`'s own retry logic lazily imports `tenacity`,
  which isn't pulled in by installing `litellm` alone. All three added to
  `pyproject.toml`.
- **`gemini-2.5-flash` deprecated mid-session; `gemini-3.8-flash`
  (its suggested replacement, officially current and free-tier-eligible)
  returned persistent `503 "high demand"` on every single reflection call
  in testing**: not a code bug, a real free-tier availability issue with a
  model released days before this was written. `reflection_lm` defaults to
  `gemini-2.5-flash-lite` instead -- the model actually proven to complete
  real calls end-to-end in this project. Revisit once 3.8-flash's free-tier
  availability settles, or on a paid tier.
- **The real free-tier daily quota is much tighter than public docs
  suggested**: search results earlier in this project said
  `gemini-2.5-flash-lite` gets ~1,000 requests/day free. The actual error
  hit in testing says the real limit is **20 requests/day per project per
  model** (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`, quota
  exhausted, ~9h retry delay given). Google tightened free-tier limits
  significantly at some point after that older documentation was written.
  **Practical consequence**: `max_metric_calls: 20` in `qa_demo`'s config
  is already close to or over one model's entire daily budget once seed
  eval + reflection calls are counted -- a real run needs either a lower
  budget, a paid tier, or patience for the daily reset. Quota is tracked
  per model name, so task_lm and reflection_lm sharing one model name share
  one budget; using two different model names doubles the effective daily
  budget.
- **Stale `run_dir` state masks fixes**: `gepa.optimize()` resumes from
  `gepa_state.bin` if `run_dir` already has one, including bad runs from
  before a bug fix -- it'll silently replay the old (broken) results instead
  of re-evaluating. `rm -rf <run_dir>` before any re-run meant to test a fix.
  **Fixed properly later** (see "Engine robustness improvements" above):
  `gepa-opt optimize` now refuses when `gepa_state.bin` exists unless
  `--resume` or `--fresh` is passed explicitly, instead of leaving this as
  a manual `rm -rf` discipline problem.
- **There already was a real log, just undocumented**: `gepa.optimize()`
  writes `run_log.txt`/`run_log_stderr.txt` into `run_dir` automatically
  (its default `Logger`, since we pass `run_dir` without overriding
  `logger`) -- every iteration, every reflection attempt, retries/errors
  verbatim. This is the thing to tail if a run looks stuck. Now listed in
  `report.md`'s "Cached run data" section and in `README.md`.

## Other docs

- `README.md` — how to actually install and run this.
- `docs/adapters.md` — how to swap in a real platform backend (Dify, etc.)
  instead of the generic one.
- `.claude/skills/gepa-optimizer/SKILL.md` — the guided conversational
  journey through all five steps above.
- `reference/README.md` — what's vendored for citation and why.
