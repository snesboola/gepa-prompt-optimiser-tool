"""Generic, platform-agnostic workflow spec.

This is the default "backend" GEPA optimizes against when you fork this repo
with no platform integration at all: an ordered chain of LLM calls described
in a small YAML file. A platform-specific adapter (Dify, LangGraph, your own
agent runtime, ...) replaces WorkflowRunner.run_node with a call into that
platform instead of calling the task LM directly -- everything upstream
(dataset loading, metric, GEPA wiring, reporting) stays the same.

Template variables use `string.Template` ($var / ${var}) syntax rather than
{curly} braces, since prompt text routinely contains literal {} (JSON
examples, format instructions) that would collide with str.format/Jinja.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any

import yaml


@dataclass
class WorkflowNode:
    id: str
    type: str  # "llm" (calls the task LM) or "template" (pure text substitution, no LM call)
    system: str | None = None
    user: str | None = None
    # False, True, "system", or "user". A node can carry both a system and a
    # user text; `optimize` must say which one GEPA is allowed to evolve. The
    # bare boolean `True` is only accepted when exactly one of system/user is
    # set -- if both are set it's ambiguous and must be spelled out.
    optimize: bool | str = False

    def render(self, context: dict[str, Any]) -> tuple[str | None, str | None]:
        system_text = Template(self.system).safe_substitute(context) if self.system else None
        user_text = Template(self.user).safe_substitute(context) if self.user else None
        return system_text, user_text

    def optimized_field(self) -> str | None:
        """Which field ('system' or 'user') `optimize` targets, or None if this
        node isn't optimized at all. Raises if the target is ambiguous."""
        if not self.optimize:
            return None
        if self.optimize in ("system", "user"):
            if getattr(self, self.optimize) is None:
                raise ValueError(f"Node {self.id!r} has optimize={self.optimize!r} but no {self.optimize} text")
            return self.optimize
        if self.optimize is True:
            if self.system is not None and self.user is not None:
                raise ValueError(
                    f"Node {self.id!r} has both `system` and `user` text and `optimize: true`. "
                    "Ambiguous -- specify which field to evolve with `optimize: system` or `optimize: user`."
                )
            if self.system is not None:
                return "system"
            if self.user is not None:
                return "user"
            raise ValueError(f"Node {self.id!r} has optimize: true but no system/user text to evolve")
        raise ValueError(f"Node {self.id!r} has invalid optimize value: {self.optimize!r}")


@dataclass
class WorkflowSpec:
    nodes: list[WorkflowNode] = field(default_factory=list)
    final_output: str | None = None  # node id whose output is the workflow's final answer; defaults to last node

    @property
    def optimizable_components(self) -> list[str]:
        return [n.id for n in self.nodes if n.optimize]

    def node(self, node_id: str) -> WorkflowNode:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(f"No node with id {node_id!r} in workflow")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "WorkflowSpec":
        data = yaml.safe_load(Path(path).read_text())
        nodes = [
            WorkflowNode(
                id=n["id"],
                type=n.get("type", "llm"),
                system=n.get("system"),
                user=n.get("user"),
                optimize=n.get("optimize", False),
            )
            for n in data["nodes"]
        ]
        for n in nodes:
            n.optimized_field()  # validates (raises) eagerly, even for non-optimized nodes with bad values
        if not any(n.optimize for n in nodes):
            raise ValueError(
                "workflow.yaml declares no node with `optimize: true` -- "
                "GEPA needs at least one component to evolve."
            )
        return cls(nodes=nodes, final_output=data.get("final_output"))

    def seed_candidate(self) -> dict[str, str]:
        """The starting candidate: component id -> current prompt text, for every
        optimizable node. GEPA evolves exactly the field `optimize` targets."""
        candidate = {}
        for n in self.nodes:
            field_name = n.optimized_field()
            if field_name is not None:
                candidate[n.id] = getattr(n, field_name) or ""
        return candidate


class WorkflowRunner:
    """Executes a WorkflowSpec for one dataset row, substituting candidate text
    for optimizable components. Default implementation calls the task LM
    directly for every "llm" node -- a platform adapter overrides `call_llm`
    (and optionally `run`) to delegate to that platform instead.
    """

    def __init__(self, spec: WorkflowSpec, task_lm):
        self.spec = spec
        self.task_lm = task_lm  # Callable[[str | list[dict]], str] or litellm string (resolved by caller)

    def call_llm(self, system_text: str | None, user_text: str | None) -> str:
        messages = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": user_text or ""})
        if callable(self.task_lm):
            return self.task_lm(messages)
        # litellm string path
        import litellm

        resp = litellm.completion(model=self.task_lm, messages=messages)
        return resp.choices[0].message.content

    def run(self, candidate: dict[str, str], row: dict[str, Any]) -> dict[str, Any]:
        """Run every node in order. Returns a trace dict:
        {
          "context": {...final variable bindings...},
          "nodes": [{"id", "system", "user", "output"}, ...],
          "final_output": str,
        }
        """
        context: dict[str, Any] = dict(row)
        node_traces = []
        last_output = None

        for n in self.spec.nodes:
            target_field = n.optimized_field()
            effective = WorkflowNode(
                id=n.id,
                type=n.type,
                system=candidate.get(n.id) if target_field == "system" else n.system,
                user=candidate.get(n.id) if target_field == "user" else n.user,
                optimize=n.optimize,
            )
            system_text, user_text = effective.render(context)

            if n.type == "llm":
                output = self.call_llm(system_text, user_text)
            else:  # "template" node: pure substitution, no LM call
                output = user_text or ""

            context[n.id] = output
            last_output = output
            node_traces.append(
                {"id": n.id, "system": system_text, "user": user_text, "output": output}
            )

        final_id = self.spec.final_output
        final_output = context[final_id] if final_id else last_output

        return {"context": context, "nodes": node_traces, "final_output": final_output}
