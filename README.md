# gepa-optimizer

Agent-agnostic prompt optimization for workflow-based LLM systems, built on
[GEPA](https://github.com/gepa-ai/gepa) (reflective, Pareto-aware prompt
evolution — [arXiv:2507.19457](https://arxiv.org/abs/2507.19457)).

## Why this exists

GEPA improves a prompt by running it, reading the full execution trace (not
just a score), diagnosing what went wrong, and proposing a targeted rewrite —
repeated against a Pareto frontier of candidates so specialist strengths
don't get thrown away for being mediocre on average.

This repo wires that loop around **your workflow and your dataset**, end to
end:

1. You point it at a dataset and a workflow definition, and say which part of
   the workflow to optimize, plus your goal/criteria/constraints.
2. It drafts a scoring function for your review and approval.
3. It runs GEPA against your real dataset, caching every candidate, score,
   and trace it tried.
4. You get the best prompt back, the full run history, and a readable
   summary report.

It is deliberately **not** tied to any particular coding agent or company
harness. GEPA itself just needs two things handed to it as plain functions —
the thing that runs your workflow, and the LM used for reflection — so this
package can run identically from a terminal, from Claude Code, or from an
internal agent harness that can execute Python and reach some LLM.

## Quickstart

The bundled example and the `init` scaffold both default to Google AI
Studio's free tier (no card required) so you can try this without spending
anything: grab a free key at [aistudio.google.com](https://aistudio.google.com/apikey).

```bash
pip install -e .
export GEMINI_API_KEY=...

gepa-opt --path examples/qa_demo optimize
gepa-opt --path examples/qa_demo report
```

That runs the bundled example: a one-node workflow answering geography
questions, scored by whether the reference answer appears in the output.

To use a different provider instead, change `task_lm`/`reflection_lm` in
`gepa.config.yaml` to any litellm model string (e.g.
`anthropic/claude-sonnet-5` with `ANTHROPIC_API_KEY` set) or a
`callable:module:attr` spec (see [`docs/adapters.md`](docs/adapters.md)).

## Starting your own project

```bash
mkdir my-project && cd my-project
gepa-opt init \
  --goal "Classify support tickets into the right queue" \
  --criteria "Must pick exactly one of the allowed queue names, no explanation text"
```

This scaffolds:

| File | Purpose |
|---|---|
| `gepa.config.yaml` | Which files to use, which LMs, how big a budget |
| `workflow.yaml` | The node(s) to optimize — edit this to match your real prompt(s) |
| `dataset.jsonl` | Replace with your real rows (JSONL, JSON array, or CSV all work) |

Then draft and approve a scoring function:

```bash
gepa-opt suggest-metric --type exact_match   # or keyword_presence, llm_judge
```

This writes an editable `metric.py` with a `score(row, trace) -> (score, feedback)`
function, pre-filled from your stated goal/criteria. **Read it. Edit the
comparison logic to match what you actually care about.** The feedback string
matters as much as the score — it's what GEPA's reflection step reads to
figure out *why* a candidate failed and what to change. "score: 0.3" tells it
nothing; "expected 'Paris', got a three-sentence hedge" tells it exactly what
to fix.

Once you're happy with `metric.py`:

```bash
gepa-opt optimize --max-metric-calls 150
gepa-opt report
```

`optimize` writes everything under `runs/latest/`:

- `result.json` — every candidate tried, its lineage, and its score
- `best_candidate.json` — just the winner
- `candidate_tree.html` — an interactive view of how candidates evolved
- `report.md` — the human-readable summary (best prompt vs. seed, score
  trajectory, Pareto frontier size)
- `iterations/` — full per-iteration traces (since `write_agent_state=True`)

## Which part of the workflow gets optimized

`workflow.yaml` is an ordered chain of nodes. Mark the ones you want GEPA to
evolve with `optimize`; leave the rest fixed. When a node has only a
`system` or only a `user` text, `optimize: true` is unambiguous. When a node
has **both** (the common case — a fixed instruction plus a data-carrying
user template), you must say which field is the one being evolved:

```yaml
nodes:
  - id: classify_intent
    type: llm
    optimize: system        # GEPA evolves the system text; $input stays fixed wiring
    system: |
      You are a support ticket classifier.
    user: |
      $input

  - id: draft_reply
    type: llm
    optimize: false         # held fixed, but still executed and visible in traces
    system: |
      Write a reply for a ticket classified as: $classify_intent
    user: |
      $input

final_output: draft_reply
```

`optimize: true` on a node with both fields set raises an error rather than
guessing — picking the wrong one silently would mean GEPA "optimizing" your
`$input` wiring instead of your actual instructions.

Templates use `$var` (Python `string.Template`) substitution rather than
`{curly braces}`, since prompts routinely contain literal `{}` (JSON
examples, format instructions) that would otherwise collide. Dataset row
fields and every earlier node's output are both available as `$name`.

## Running this against your real platform instead of a bare LLM call

The default `WorkflowRunner` just calls the task LM directly for every node —
fine for optimizing a standalone prompt, but it isn't your production system
(no tools, no retrieval, no real orchestration). See
[`docs/adapters.md`](docs/adapters.md) for how to swap in a runner that
actually executes your platform (Dify, LangGraph, an internal agent runtime)
per candidate, and how to point the reflection/task LM at something other
than a direct provider API key (e.g. an internal gateway that only exposes a
system-prompt/user-prompt call). Both are single, isolated seams — nothing
else in this package needs to change.

## How the pieces map to GEPA

| This repo | GEPA concept |
|---|---|
| `workflow.yaml` node(s) marked `optimize: true` | the `candidate: dict[str, str]` GEPA evolves |
| `WorkflowGEPAAdapter.evaluate()` | runs a candidate against a batch, returns scores + traces |
| `WorkflowGEPAAdapter.make_reflective_dataset()` | turns traces into the feedback the reflection LM reads |
| `reflection_lm` in `gepa.config.yaml` | the LM that reads failures and proposes a better prompt |
| `run_dir` / `write_agent_state=True` | every candidate, score, and trace, kept on disk |

## Development

```bash
pip install -e ".[dev]"
pytest
```

`tests/test_workflow.py` exercises the node-chaining and adapter logic
directly (no API keys / network needed — it uses a fake task LM).

## License

MIT, see [LICENSE](LICENSE).
