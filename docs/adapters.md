# Writing a platform adapter

This repo's default backend is `WorkflowRunner` (`src/gepa_optimizer/workflow.py`):
it reads a small YAML graph and calls the task LM directly for every `llm`
node. That's enough to optimize a standalone prompt, but it is not your real
production system -- it doesn't call your actual agent platform, tools, or
retrieval.

To optimize a prompt *inside* a real platform (Dify, LangGraph, an internal
agent runtime, ...), replace the execution backend without touching anything
else: dataset loading, the metric contract, the GEPA wiring in `runner.py`,
and the report all stay the same.

## What to change

Subclass `WorkflowRunner` and override `call_llm` (cheapest option: you keep
the generic node-chaining logic and only change how one node's output is
produced) or `run` entirely (full control: delegate the whole row to the
platform's own execution engine and just report back a trace).

```python
from gepa_optimizer.workflow import WorkflowRunner

class DifyWorkflowRunner(WorkflowRunner):
    def run(self, candidate: dict[str, str], row: dict) -> dict:
        # 1. Take the Dify DSL export for the real workflow.
        # 2. Substitute candidate[node_id] into the target node's prompt field
        #    for every node GEPA is allowed to evolve.
        # 3. Call Dify's /workflows/run API with row's fields as inputs.
        # 4. Return {"context": {...}, "nodes": [...], "final_output": ...}
        #    shaped the same way the generic runner does, so the metric and
        #    make_reflective_dataset code don't need to know the difference.
        ...
```

Then point `WorkflowGEPAAdapter` (`src/gepa_optimizer/adapter.py`) at your
runner subclass instead of the default one -- that's the only wiring change.

## The other portability seam: the reflection/task LM

GEPA calls `task_lm` and `reflection_lm` as plain
`Callable[[str | list[dict]], str]` (or a litellm model string). The `llm.py`
module resolves either a litellm string or a `callable:module:attr` spec from
config into that shape. If your environment's only sanctioned path to a model
is through something like a Dify workflow that takes a system + user prompt
and returns text, write:

```python
# dify_client.py
import requests

def dify_llm_call(prompt: str | list[dict]) -> str:
    if isinstance(prompt, list):
        system = next((m["content"] for m in prompt if m["role"] == "system"), "")
        user = "\n".join(m["content"] for m in prompt if m["role"] != "system")
    else:
        system, user = "", prompt

    resp = requests.post(
        "https://your-dify-host/v1/workflows/run",
        headers={"Authorization": "Bearer ..."},
        json={"inputs": {"system_prompt": system, "user_prompt": user}, "response_mode": "blocking"},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["data"]["outputs"]["text"]
```

and set, in `gepa.config.yaml`:

```yaml
task_lm: "callable:dify_client:dify_llm_call"
reflection_lm: "callable:dify_client:dify_llm_call"
```

No other file changes. This is the seam the project's internal-harness branch
is expected to use.
