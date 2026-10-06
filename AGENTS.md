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
| 5. Best prompt + cached runs + approaches + summary | `run_dir` (`runs/latest/`): `best_candidate.json`, `result.json` (every candidate + lineage + score), `iterations/` (full per-iteration traces, via `write_agent_state=True`), `candidate_tree.html`, and `report.md` (the human-readable summary, built by `report.py`) |

## Other docs

- `README.md` — how to actually install and run this.
- `docs/adapters.md` — how to swap in a real platform backend (Dify, etc.)
  instead of the generic one.
- `.claude/skills/gepa-optimizer/SKILL.md` — the guided conversational
  journey through all five steps above.
- `reference/README.md` — what's vendored for citation and why.
