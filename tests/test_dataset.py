from gepa_optimizer.dataset import inspect_dataset


def test_inspect_dataset_reports_columns_and_flags_categorical():
    rows = [
        {"input": "q1", "label": "GROUNDED"},
        {"input": "q2", "label": "HALLUCINATED"},
        {"input": "q3", "label": "GROUNDED"},
        {"input": "q4", "label": "HALLUCINATED"},
    ]

    info = inspect_dataset(rows)

    assert info["n_rows"] == 4
    assert info["columns"] == ["input", "label"]
    assert info["column_stats"]["label"]["n_distinct"] == 2
    assert info["column_stats"]["label"]["likely_categorical"] is True
    assert info["column_stats"]["input"]["n_distinct"] == 4
    assert info["column_stats"]["input"]["likely_categorical"] is False  # every value unique, not categorical
    assert len(info["sample"]) <= 3


def test_inspect_dataset_handles_empty_dataset():
    info = inspect_dataset([])
    assert info == {"n_rows": 0, "columns": [], "column_stats": {}, "sample": []}


def test_inspect_dataset_handles_missing_keys_across_rows():
    rows = [{"a": 1, "b": 2}, {"a": 3}]  # row 2 lacks "b" -- real CSVs/JSONL can be ragged

    info = inspect_dataset(rows)

    assert info["columns"] == ["a", "b"]
    assert info["column_stats"]["b"]["present_in_rows"] == 1
