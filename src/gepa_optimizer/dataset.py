"""Dataset loading. Supports JSONL and CSV; each row becomes a plain dict that
the workflow's templates substitute by field name (e.g. a row with an `input`
column is available to node templates as $input).
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a JSON array of row objects")
        return data
    if path.suffix == ".csv":
        with path.open(newline="") as f:
            return [_decode_csv_row(row) for row in csv.DictReader(f)]
    raise ValueError(f"Unsupported dataset format: {path.suffix} (use .jsonl, .json, or .csv)")


def _decode_csv_row(row: dict[str, str]) -> dict[str, Any]:
    """csv.DictReader returns every cell as a plain string, with no concept
    of a list-typed column. A metric expecting `row["required_keywords"]`
    to be a list (keyword_presence, composite) would otherwise silently
    iterate the raw string character-by-character instead of raising --
    wrong, but no error, which is worse. Cells that look like a JSON array
    or object get decoded; anything else (including a cell that merely
    starts with '[' but isn't valid JSON) is left as the original string.
    """
    decoded: dict[str, Any] = {}
    for key, value in row.items():
        stripped = value.strip() if isinstance(value, str) else value
        if isinstance(stripped, str) and stripped[:1] in "[{":
            try:
                decoded[key] = json.loads(stripped)
                continue
            except json.JSONDecodeError:
                pass
        decoded[key] = value
    return decoded


def inspect_dataset(rows: list[dict[str, Any]], sample_size: int = 3) -> dict[str, Any]:
    """Summarize a dataset's actual structure -- column names, how many
    distinct values each one has, and example values -- so a metric's
    assumed field names (e.g. "label", "reference") can be checked against
    what the data is actually called, instead of assumed and silently
    wrong until it KeyErrors at runtime.

    A column with few distinct values relative to row count is flagged as
    "likely categorical" -- a reasonable signal for "this might be the
    label/verdict/category column," not a guarantee.
    """
    if not rows:
        return {"n_rows": 0, "columns": [], "column_stats": {}, "sample": []}

    columns = sorted(set().union(*(row.keys() for row in rows)))
    column_stats: dict[str, Any] = {}
    for col in columns:
        values = [row[col] for row in rows if col in row]
        distinct = sorted(set(str(v) for v in values))
        # Ratio-based, not an absolute floor: a floor like "<=5" would flag
        # a fully-unique 4-row column (no repeats at all) as categorical on
        # a tiny dataset just because 4 <= 5. Require actual repetition
        # (distinct count meaningfully below row count), capped so a
        # high-cardinality column on a large dataset doesn't still qualify.
        categorical_cap = min(20, max(1, int(len(rows) * 0.6)))
        likely_categorical = 1 < len(distinct) <= categorical_cap
        column_stats[col] = {
            "present_in_rows": len(values),
            "n_distinct": len(distinct),
            "examples": distinct[:5],
            "likely_categorical": likely_categorical,
        }

    return {"n_rows": len(rows), "columns": columns, "column_stats": column_stats, "sample": rows[:sample_size]}


def split_dataset(
    rows: list[dict[str, Any]], val_fraction: float = 0.3, test_fraction: float = 0.0, seed: int = 0
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Returns (train, val, test). `test` is held out from everything GEPA
    ever sees during search (it's not trainset or valset) -- used only for a
    final, independent sanity check against overfitting to the validation
    set, the same way a model's test split works. Defaults to empty (0.0):
    opt-in, since a 3-way split of an already-small dataset can leave too
    little in any one split to mean much -- turn it on once you have enough
    rows for all three to be meaningful (a few dozen at least).
    """
    if len(rows) < 4:
        # too small to split meaningfully; reuse the same rows for train and val, no test split
        return rows, rows, []
    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)
    n_test = int(len(shuffled) * test_fraction)
    test, remaining = shuffled[:n_test], shuffled[n_test:]
    n_val = max(1, int(len(remaining) * val_fraction))
    val, train = remaining[:n_val], remaining[n_val:]
    return train, val, test
