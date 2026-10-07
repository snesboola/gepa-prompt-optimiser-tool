from pathlib import Path

from gepa_optimizer.metrics import classification_report, load_metric_module, scaffold_metric


def test_classification_report_computes_real_recall_precision_f1():
    rows = [
        {"label": "HALLUCINATED"},
        {"label": "HALLUCINATED"},
        {"label": "GROUNDED"},
        {"label": "GROUNDED"},
    ]
    # predictions: catches both hallucinations (recall=1.0), one false alarm on
    # a grounded row (precision=2/3)
    predictions = ["HALLUCINATED", "HALLUCINATED", "HALLUCINATED", "GROUNDED"]

    def classify_fn(row, trace):
        return predictions[trace["idx"]]

    traces = [{"idx": i} for i in range(len(rows))]

    stats = classification_report(rows, traces, classify_fn, positive_label="HALLUCINATED")

    assert stats["tp"] == 2
    assert stats["fp"] == 1
    assert stats["tn"] == 1
    assert stats["fn"] == 0
    assert stats["recall"] == 1.0
    assert stats["precision"] == 2 / 3
    assert abs(stats["f1"] - 0.8) < 1e-9


def test_classification_report_handles_no_positive_examples():
    rows = [{"label": "GROUNDED"}, {"label": "GROUNDED"}]
    traces = [{}, {}]

    stats = classification_report(rows, traces, lambda r, t: "GROUNDED", positive_label="HALLUCINATED")

    assert stats["tp"] == 0
    assert stats["fn"] == 0
    assert stats["recall"] is None  # 0/0 -- undefined, not 0.0
    assert stats["fp"] == 0
    assert stats["precision"] is None


def test_composite_template_hard_gate_overrides_judge(tmp_path: Path):
    metric_path = scaffold_metric(
        "composite", goal="Write a product blurb", criteria="Must capture real industry detail", out_path=tmp_path / "metric.py"
    )
    module = load_metric_module(metric_path)
    module.DEFAULT_REQUIRED_PHRASES = ["30-day guarantee"]

    trace_missing_phrase = {"final_output": "This blender is great and you will love it."}
    judge_that_would_score_high = lambda prompt: "SCORE: 0.95\nREASON: Sounds great."

    score, feedback = module.score({}, trace_missing_phrase, judge_lm=judge_that_would_score_high)

    assert score == 0.0  # the judge is never even consulted when the hard gate fails
    assert "30-day guarantee" in feedback


def test_composite_template_defers_to_judge_once_gate_passes():
    metric_path_dir = Path("/tmp")
    metric_path = scaffold_metric(
        "composite", goal="g", criteria="c", out_path=metric_path_dir / "composite_defer_test_metric.py"
    )
    module = load_metric_module(metric_path)
    module.DEFAULT_REQUIRED_PHRASES = ["30-day guarantee"]

    trace_has_phrase = {"final_output": "Ships with a 30-day guarantee and a 1200W motor."}
    judge = lambda prompt: "SCORE: 0.9\nREASON: Specific technical detail."

    score, feedback = module.score({}, trace_has_phrase, judge_lm=judge)

    assert score == 0.9
    assert "Specific technical detail" in feedback
    metric_path.unlink()
