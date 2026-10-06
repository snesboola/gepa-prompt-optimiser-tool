"""Compares the judge's verdict to the labeled ground truth. Deliberately
tolerant of phrasing ("not grounded" == HALLUCINATED) but strict about
requiring an unambiguous verdict -- a judge that hedges or buries the verdict
in a paragraph should score 0, since that's not usable downstream.
"""


def score(row: dict, trace: dict) -> tuple[float, str]:
    expected = row["label"].strip().upper()
    output = str(trace["final_output"]).upper()

    mentions_not_grounded = "HALLUCINAT" in output or "NOT GROUNDED" in output or "NOT SUPPORTED" in output
    mentions_grounded = "GROUNDED" in output and not mentions_not_grounded

    if mentions_not_grounded and mentions_grounded:
        predicted = "AMBIGUOUS"
    elif mentions_not_grounded:
        predicted = "HALLUCINATED"
    elif mentions_grounded:
        predicted = "GROUNDED"
    else:
        predicted = "UNCLEAR"

    if predicted == expected:
        return 1.0, f"Correctly judged as {expected}."

    return 0.0, (
        f"Expected {expected}, but the verdict parsed as {predicted} from raw output "
        f"{trace['final_output']!r}. If the output is verbose or hedges, the prompt "
        "should demand a single unambiguous word (e.g. GROUNDED or HALLUCINATED) as "
        "the entire response."
    )
