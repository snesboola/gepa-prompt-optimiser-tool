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
  verified logically with fake stand-in judges (a hedgy one scoring 4/8, a
  strict one scoring 7/8) — not yet run against a real model (blocked on
  today's Gemini free-tier quota, see below).
- **Open-ended writing/creative tasks** — no single correct output exists;
  metric is `llm_judge`, scoring against stated criteria via a separate
  `judge_lm` call. Verified working end-to-end (fake task/judge LMs, no API
  calls) for a no-`reference` writing task: candidate evolves a "write X"
  system prompt, `judge_lm` rates the output, score + rationale flow
  correctly into the reflective dataset. No example committed for this path
  yet — worth adding one (e.g. `examples/writing_judge_demo/`) before
  relying on it against a real model.

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
  of this (via a fork) for citation/reproducibility — read for reference,
  never built on top of or copied from.

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

## Where each vision step actually lives

| Step | Where |
|---|---|
| 1. Dataset + workflow folder | `dataset.py` (JSONL/JSON/CSV loader) + `workflow.yaml` (generic now; a real Dify export is translated by hand for now, see the skill's "Non-native workflow sources" section — automatic Dify parsing is the deferred branch) |
| 2. Goal/criteria/target/constraints | Gathered conversationally by the skill (Phase 1), stored in `gepa.config.yaml`'s `goal`/`criteria` fields, expressed as `optimize: system\|user` on the target node(s) |
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
