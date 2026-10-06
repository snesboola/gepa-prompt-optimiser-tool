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
gepa-opt suggest-metric --type exact_match   # or keyword_presence, llm_judge, classification, composite
```

`classification` is the one to reach for when the goal is phrased in terms
of recall/precision (e.g. "catch most real hallucinations without too many
false alarms") rather than raw accuracy — see the worked example below for
why that needs a different approach than `exact_match`. `composite` is for
criteria that mix a hard requirement (must mention an exact phrase) with a
vaguer qualitative one (must "capture industry detail") in the same
output — it hard-gates on the exact requirement first, then only runs an
`llm_judge`-style call for the qualitative part once that gate passes, so
the hard requirement can never be traded away for a nicer-sounding answer.

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
- `run_log.txt` / `run_log_stderr.txt` — `gepa`'s own live log: every
  iteration, every reflection attempt, retries and errors verbatim (written
  automatically because `run_dir` is set; this is the thing to tail if a run
  looks stuck or you want to see exactly what happened)
- `gepa_state.bin` — `gepa`'s own pickled run state (lets a future version
  resume); not human-readable, `result.json` is the readable equivalent

## Worked example: optimizing a hallucination-detection judge

A full walkthrough, start to finish, using a different shape of problem than
the QA example above: here the prompt being optimized is itself a **judge**
— given a source context and a generated answer, it has to say whether the
answer is actually supported by the context, or hallucinated. This is the
LLM-as-judge case from [AGENTS.md](AGENTS.md)'s "what kinds of prompts this
optimizes" — scored against *known-correct labels*, not another judge call.
The finished files live in [`examples/hallucination_judge_demo`](examples/hallucination_judge_demo).

**1. State the goal and scaffold the project:**

```bash
mkdir hallucination-judge && cd hallucination-judge
gepa-opt init \
  --goal "Accurately flag whether a generated answer is fully supported by its source context" \
  --criteria "Output must be a single unambiguous verdict, no hedging or extra explanation"
```

**2. Point `workflow.yaml` at the real judge prompt, replace `dataset.jsonl`
with labeled examples** (context + answer + the true verdict):

```yaml
nodes:
  - id: judge
    type: llm
    optimize: system
    system: |
      You are a fact-checking judge. Decide whether the ANSWER is fully
      supported by the CONTEXT.
    user: |
      CONTEXT:
      $context

      ANSWER:
      $answer

      Is the answer grounded in the context?

final_output: judge
```

```jsonl
{"context": "The Eiffel Tower was completed in 1889 and stands 330 metres tall.", "answer": "The Eiffel Tower was finished in 1889.", "label": "GROUNDED"}
{"context": "The Eiffel Tower was completed in 1889 and stands 330 metres tall.", "answer": "The Eiffel Tower is 450 metres tall.", "label": "HALLUCINATED"}
```

**3. Draft and approve the scoring function:**

The goal here isn't just "match the label" — it's *catch most real
hallucinations without too many false alarms*, i.e. recall and precision
specifically. Those are aggregate, whole-dataset numbers; no single
`score(row, trace)` call can compute them, since GEPA scores one row at a
time. The `classification` template is built for exactly this: it scores
each row with asymmetric penalties (missing a real hallucination costs
more than a false alarm, by default) as the closest per-example signal
GEPA can act on, and separately reports the *real* recall/precision/F1 for
the winning prompt after the fact.

```bash
gepa-opt suggest-metric --type classification
```

The scaffold needs hand-editing to set the actual positive label and
negative-class check, and to tune the recall/precision tradeoff:

```python
POSITIVE_LABEL = "HALLUCINATED"
FN_PENALTY = 0.0  # missed a real hallucination -- the costlier mistake for this task
FP_PENALTY = 0.4  # false alarm on a grounded answer -- still bad, but softer


def classify(row: dict, trace: dict) -> str:
    output = str(trace["final_output"]).upper()
    mentions_not_grounded = "HALLUCINAT" in output or "NOT GROUNDED" in output
    mentions_grounded = "GROUNDED" in output and not mentions_not_grounded
    if mentions_not_grounded:
        return "HALLUCINATED"
    if mentions_grounded:
        return "GROUNDED"
    return "UNCLEAR"


def score(row: dict, trace: dict) -> tuple[float, str]:
    expected = row["label"].strip().upper()
    predicted = classify(row, trace)
    if predicted == expected:
        return 1.0, f"Correctly judged as {expected}."
    if expected == POSITIVE_LABEL:
        return FN_PENALTY, f"MISSED a real hallucination (got {predicted})."
    return FP_PENALTY, f"False alarm on a grounded answer (got {predicted})."
```

This is the approval gate: read the logic above, confirm `FN_PENALTY`/
`FP_PENALTY` actually reflect which mistake you care about more, before
running anything.

**4. Run it:**

```bash
gepa-opt optimize --max-metric-calls 20
gepa-opt report
```

**What actually happens**, verified against the logic above with stand-in
judges (no API calls — see `AGENTS.md`'s bug log for why a live run is
pending a Gemini free-tier quota reset as of this writing): the seed prompt
("Decide whether the ANSWER is fully supported by the CONTEXT") leaves the
output format open, so a judge that hedges ("the answer seems *mostly*
grounded, though it's hard to say for certain") parses as `GROUNDED` every
time — missing every single real hallucination. That's not just a low
score; the real confusion-matrix number for this seed is **recall = 0.0%**.
GEPA's reflection step reads the specific feedback ("MISSED a real
hallucination...") across several failing examples, diagnoses that *the
prompt doesn't forbid hedging*, and proposes a stricter instruction
demanding a single word. A judge following that stricter prompt catches
every real hallucination in this dataset with one false alarm: **recall =
100%, precision = 80%, F1 = 0.89**. Both the proxy score trajectory and
these real numbers (seed vs. best, side by side) end up in `report.md`'s
"Real recall / precision" section — so you see the actual metric you asked
about, not just a proxy score that's hopefully correlated with it.

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

## Using this with any coding agent or harness

Nothing here is tied to Claude Code specifically. The engine is a plain CLI
(`gepa-opt`) and an importable Python package; the guided journey is a
Markdown instructions file. Any environment that can run Python and read a
Markdown playbook can drive the whole thing — a different coding agent, a
custom internal harness, or just you in a terminal.

1. **Get the code there.** Clone this repo wherever you're working. If that
   environment can't reach GitHub, `reference/gepa-ai-gepa` (a submodule
   pointing at the upstream library, kept purely for citation) can be
   dropped — nothing in the package imports from it.

2. **Install it** (`>=3.10`, per `gepa`'s own requirement):
   ```bash
   pip install -e .
   ```
   No specific coding agent and no GUI required.

3. **Have your harness follow the skill.** `.claude/skills/gepa-optimizer/SKILL.md`
   is a plain Markdown playbook for the five-phase journey (clarify
   goal/target/constraints → draft+approve a metric → run → deliver
   results). Claude Code auto-discovers it from that path; any other agent
   that can be pointed at a Markdown file and told "follow these
   instructions" can use the exact same file — the phases, the approval
   gate, and the CLI commands it calls don't reference Claude Code
   anywhere. `AGENTS.md` is the harness-agnostic summary of the same intent,
   for a harness (or a person) that just wants the "why" without the
   step-by-step playbook.

4. **Point the LM at whatever you actually have access to.** A litellm
   model string works if you have a normal provider key. If the only
   sanctioned path to a model is through something else entirely — an
   internal gateway, a workflow platform that takes a system prompt + user
   prompt and returns text — write one function matching
   `Callable[[str | list[dict]], str]` (see [`docs/adapters.md`](docs/adapters.md)
   for a worked example) and point config at it:
   ```yaml
   task_lm: "callable:my_client:my_llm_call"
   reflection_lm: "callable:my_client:my_llm_call"
   ```
   Credentials go in `.env` (gitignored, loaded automatically — see
   `.env.example`); nothing in this package needs them hardcoded.

5. **If you want it to execute your real platform, not just a bare LLM
   call**, subclass `WorkflowRunner` and override `run()`/`call_llm()` to
   actually invoke that platform per candidate (`docs/adapters.md` has the
   pattern). Skip this if optimizing the prompt text in isolation is good
   enough — most of the time it is.

6. **Everything else is identical** regardless of where you're running
   this: `gepa-opt init` / `suggest-metric` / `optimize` / `report`, the
   same `gepa.config.yaml` shape, the same `run_dir` artifacts.

See [`AGENTS.md`](AGENTS.md) for the full reasoning behind this split
(why a real upstream library instead of a toy reimplementation, why the
LM and workflow backends are both pluggable seams) if you want the "why,"
not just the "how."

## How the pieces map to GEPA

| This repo | GEPA concept |
|---|---|
| `workflow.yaml` node(s) marked `optimize: true` | the `candidate: dict[str, str]` GEPA evolves |
| `WorkflowGEPAAdapter.evaluate()` | runs a candidate against a batch, returns scores + traces |
| `WorkflowGEPAAdapter.make_reflective_dataset()` | turns traces into the feedback the reflection LM reads |
| `reflection_lm` in `gepa.config.yaml` | the LM that reads failures and proposes a better prompt |
| `reflection_minibatch_size` in `gepa.config.yaml` | how many rows each reflection step sees — auto-set to the whole training set for ≤25 rows (no noisy 3-row sampling), overridable |
| `run_dir` + our own `result.json`/`report.md` dump | every candidate, score, and lineage, kept on disk |

## Development

```bash
pip install -e ".[dev]"
pytest
```

`tests/test_workflow.py` exercises the node-chaining and adapter logic
directly (no API keys / network needed — it uses a fake task LM).

## License

MIT, see [LICENSE](LICENSE).
