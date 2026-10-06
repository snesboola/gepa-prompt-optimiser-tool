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
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported dataset format: {path.suffix} (use .jsonl, .json, or .csv)")


def split_dataset(
    rows: list[dict[str, Any]], val_fraction: float = 0.3, seed: int = 0
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(rows) < 4:
        # too small to split meaningfully; reuse the same rows for train and val
        return rows, rows
    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    return shuffled[n_val:], shuffled[:n_val]
