"""Pluggable LM resolution.

An LM spec is one of:
  - a litellm-style model string, e.g. "anthropic/claude-sonnet-5" (requires the
    matching provider API key in the environment, e.g. ANTHROPIC_API_KEY).
  - "callable:<module>:<attr>" pointing at a Python callable with signature
    (prompt: str | list[dict]) -> str, importable from the project directory
    or any module on PYTHONPATH.

This indirection is the whole portability story: today you point task_lm /
reflection_lm at a litellm string backed by a normal provider key. When you
move to an environment where the only sanctioned way to reach a model is
through some internal gateway (a Dify workflow, an internal proxy, ...), you
write one function matching the callable signature below and point the spec
at it instead. Nothing else in this package changes.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Callable, Union

LMSpec = Union[str, Callable[[Union[str, list[dict]]], str]]


def resolve_lm(spec: LMSpec, *, project_root: Path | None = None) -> LMSpec:
    """Resolve an LM spec from config into something gepa.optimize() accepts.

    - A plain model string is passed through untouched (gepa resolves it via
      litellm internally).
    - A "callable:module:attr" string is imported and returned as the callable
      itself.
    - An already-callable value is returned untouched.
    """
    if callable(spec):
        return spec

    if not isinstance(spec, str):
        raise TypeError(f"LM spec must be a string or callable, got {type(spec)!r}")

    if spec.startswith("callable:"):
        _, module_name, attr_name = spec.split(":", 2)
        if project_root is not None:
            root_str = str(project_root)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)
        module = importlib.import_module(module_name)
        fn = getattr(module, attr_name)
        if not callable(fn):
            raise TypeError(f"{spec!r} does not point at a callable")
        return fn

    # Plain litellm-style model string, e.g. "anthropic/claude-sonnet-5"
    return spec


def load_callable(dotted: str, *, project_root: Path | None = None) -> Callable[..., Any]:
    """Import 'module:attr' (no 'callable:' prefix) and return the attribute."""
    module_name, attr_name = dotted.split(":", 1)
    if project_root is not None:
        root_str = str(project_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)
