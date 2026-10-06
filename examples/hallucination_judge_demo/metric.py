"""Compares the judge's verdict to the labeled ground truth, with
recall/precision in mind -- not just raw accuracy. A missed hallucination
(false negative: real HALLUCINATED answer judged GROUNDED) is treated as
worse than a false alarm (false positive: a GROUNDED answer wrongly flagged)
by giving it a harsher penalty. Tune FN_PENALTY / FP_PENALTY to match your
actual risk tolerance. `classify()` is exposed separately so the reporting
layer can compute the REAL recall/precision/F1 for the winning prompt after
optimization -- see metrics.classification_report() / runner.py, and the
"Real recall / precision" section this produces in report.md.
"""

LABEL_FIELD = "label"  # matches dataset.jsonl's column name; rename both together if it changes
POSITIVE_LABEL = "HALLUCINATED"
FN_PENALTY = 0.0  # missed a real hallucination -- the costlier mistake for this task
FP_PENALTY = 0.4  # false alarm on a grounded answer -- still bad, but softer


def classify(row: dict, trace: dict) -> str:
    output = str(trace["final_output"]).upper()
    mentions_not_grounded = "HALLUCINAT" in output or "NOT GROUNDED" in output or "NOT SUPPORTED" in output
    mentions_grounded = "GROUNDED" in output and not mentions_not_grounded

    if mentions_not_grounded and mentions_grounded:
        return "AMBIGUOUS"
    if mentions_not_grounded:
        return "HALLUCINATED"
    if mentions_grounded:
        return "GROUNDED"
    return "UNCLEAR"


def score(row: dict, trace: dict) -> tuple[float, str]:
    expected = row[LABEL_FIELD].strip().upper()
    predicted = classify(row, trace)

    if predicted == expected:
        return 1.0, f"Correctly judged as {expected}."

    if expected == POSITIVE_LABEL:
        return FN_PENALTY, (
            f"MISSED a real hallucination (expected {expected}, got {predicted} from "
            f"{trace['final_output']!r}). Missing real hallucinations is the worse failure "
            "mode here -- the prompt should be more willing to flag borderline cases."
        )

    return FP_PENALTY, (
        f"False alarm: a grounded answer was judged {predicted} (expected {expected}) from "
        f"{trace['final_output']!r}. Too many false alarms erode trust in the judge; tighten "
        "the criteria for what counts as unsupported."
    )
