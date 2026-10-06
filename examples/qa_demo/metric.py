"""Loose match: the reference answer must appear somewhere in the output.
Demonstrates that feedback should explain *why*, not just report pass/fail --
that's what the reflection step actually reads.
"""


def score(row: dict, trace: dict) -> tuple[float, str]:
    expected = str(row.get("reference", "")).strip().lower()
    actual = str(trace["final_output"]).strip().lower()

    if expected in actual:
        return 1.0, f"Output contains the expected answer {expected!r}."
    return 0.0, (
        f"Expected answer {expected!r} not found in output {actual!r}. "
        "If the output is verbose or hedges, tighten the prompt to demand a direct, "
        "short answer; if it's simply wrong, the prompt may need an example."
    )
