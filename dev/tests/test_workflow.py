import pytest

from gepa_optimizer.adapter import WorkflowGEPAAdapter
from gepa_optimizer.workflow import WorkflowNode, WorkflowRunner, WorkflowSpec


def fake_task_lm(messages):
    user_msg = next(m["content"] for m in messages if m["role"] == "user")
    return f"echo: {user_msg}"


def test_workflow_runner_substitutes_candidate_and_chains_context():
    spec = WorkflowSpec(
        nodes=[
            # optimize="user" because the evolved text here is the user turn,
            # not the fixed "Be nice." system instruction -- see the ambiguity
            # test below for why bare `optimize=True` would reject this node.
            WorkflowNode(id="greeting", type="llm", system="Be nice.", user="Say hi to $name", optimize="user"),
            WorkflowNode(
                id="followup",
                type="llm",
                system="Be nice.",
                user="Given '$greeting', ask a question",
                optimize=False,
            ),
        ],
        final_output="followup",
    )
    runner = WorkflowRunner(spec, task_lm=fake_task_lm)
    candidate = {"greeting": "Say an enthusiastic hi to $name"}

    trace = runner.run(candidate, {"name": "Ada"})

    assert trace["nodes"][0]["system"] == "Be nice."  # untouched by the candidate mutation
    assert trace["nodes"][0]["user"] == "Say an enthusiastic hi to Ada"
    assert trace["nodes"][0]["output"] == "echo: Say an enthusiastic hi to Ada"
    assert "echo: Say an enthusiastic hi to Ada" in trace["nodes"][1]["user"]
    assert trace["final_output"] == trace["nodes"][1]["output"]


def test_optimize_true_is_ambiguous_when_both_fields_set():
    node = WorkflowNode(id="x", type="llm", system="S", user="U", optimize=True)
    with pytest.raises(ValueError, match="Ambiguous"):
        node.optimized_field()


def test_optimize_system_leaves_user_wiring_untouched():
    spec = WorkflowSpec(
        nodes=[
            WorkflowNode(
                id="system_prompt",
                type="llm",
                system="You are a QA bot. Answer in one sentence.",
                user="$input",
                optimize="system",
            )
        ],
        final_output="system_prompt",
    )
    seed = spec.seed_candidate()
    assert seed == {"system_prompt": "You are a QA bot. Answer in one sentence."}

    runner = WorkflowRunner(spec, task_lm=fake_task_lm)
    mutated = {"system_prompt": "Answer with only the city name."}
    trace = runner.run(mutated, {"input": "What is the capital of Japan?"})

    assert trace["nodes"][0]["system"] == "Answer with only the city name."
    assert trace["nodes"][0]["user"] == "What is the capital of Japan?"  # real input, not clobbered


def test_adapter_evaluate_scores_and_reflective_dataset():
    spec = WorkflowSpec(
        nodes=[WorkflowNode(id="system_prompt", type="llm", system="x", user="$input", optimize="system")],
        final_output="system_prompt",
    )

    def score_fn(row, trace):
        ok = row["reference"] in trace["final_output"]
        return (1.0, "matched") if ok else (0.0, f"expected {row['reference']!r}")

    adapter = WorkflowGEPAAdapter(spec=spec, task_lm=fake_task_lm, score_fn=score_fn)
    batch = [{"input": "Paris", "reference": "Paris"}, {"input": "Rome", "reference": "Nope"}]
    candidate = spec.seed_candidate()

    result = adapter.evaluate(batch, candidate, capture_traces=True)

    assert result.scores == [1.0, 0.0]
    assert len(result.trajectories) == 2

    reflective = adapter.make_reflective_dataset(candidate, result, components_to_update=["system_prompt"])
    assert len(reflective["system_prompt"]) == 2
    assert "expected" in reflective["system_prompt"][1]["Feedback"]


def test_adapter_raises_when_every_row_in_batch_raises():
    spec = WorkflowSpec(
        nodes=[WorkflowNode(id="system_prompt", type="llm", system="x", user="$input", optimize="system")],
        final_output="system_prompt",
    )

    def broken_task_lm(messages):
        raise ModuleNotFoundError("No module named 'litellm'")  # simulates today's real failure mode

    def score_fn(row, trace):
        return 1.0, "ok"  # never actually reached -- every row fails before scoring

    adapter = WorkflowGEPAAdapter(spec=spec, task_lm=broken_task_lm, score_fn=score_fn)
    batch = [{"input": "a"}, {"input": "b"}]

    with pytest.raises(RuntimeError, match="raised an exception"):
        adapter.evaluate(batch, spec.seed_candidate(), capture_traces=True)


def test_adapter_does_not_raise_when_only_some_rows_fail():
    spec = WorkflowSpec(
        nodes=[WorkflowNode(id="system_prompt", type="llm", system="x", user="$input", optimize="system")],
        final_output="system_prompt",
    )

    def flaky_task_lm(messages):
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        if user_msg == "fail_me":
            raise RuntimeError("simulated transient failure")
        return f"echo: {user_msg}"

    def score_fn(row, trace):
        return 1.0, "ok"

    adapter = WorkflowGEPAAdapter(spec=spec, task_lm=flaky_task_lm, score_fn=score_fn)
    batch = [{"input": "fail_me"}, {"input": "fine"}]

    result = adapter.evaluate(batch, spec.seed_candidate(), capture_traces=True)  # must not raise
    assert result.scores == [0.0, 1.0]
    assert "Execution error" in result.trajectories[0].feedback


def test_adapter_parallel_execution_preserves_order_and_correctness():
    spec = WorkflowSpec(
        nodes=[WorkflowNode(id="system_prompt", type="llm", system="x", user="$input", optimize="system")],
        final_output="system_prompt",
    )

    def score_fn(row, trace):
        return (1.0, "ok") if trace["final_output"] == f"echo: {row['input']}" else (0.0, "mismatch")

    adapter = WorkflowGEPAAdapter(spec=spec, task_lm=fake_task_lm, score_fn=score_fn, max_workers=4)
    batch = [{"input": str(i)} for i in range(10)]

    result = adapter.evaluate(batch, spec.seed_candidate(), capture_traces=True)

    assert result.scores == [1.0] * 10  # every row matched its own input, not some other thread's
    assert [t.row["input"] for t in result.trajectories] == [str(i) for i in range(10)]


def test_adapter_computes_recall_precision_objective_scores_when_classify_fn_given():
    spec = WorkflowSpec(
        nodes=[WorkflowNode(id="judge", type="llm", system="x", user="$input", optimize="system")],
        final_output="judge",
    )

    def score_fn(row, trace):
        return 1.0, "ok"

    def classify_fn(row, trace):
        return "HALLUCINATED" if "bad" in trace["final_output"] else "GROUNDED"

    adapter = WorkflowGEPAAdapter(
        spec=spec,
        task_lm=fake_task_lm,
        score_fn=score_fn,
        classify_fn=classify_fn,
        positive_label="HALLUCINATED",
    )
    # row 0: expected HALLUCINATED, task_lm echoes "bad" -> classified HALLUCINATED -> TP
    # row 1: expected GROUNDED, task_lm echoes "good" -> classified GROUNDED -> TN
    batch = [{"input": "bad", "label": "HALLUCINATED"}, {"input": "good", "label": "GROUNDED"}]

    result = adapter.evaluate(batch, spec.seed_candidate(), capture_traces=True)

    assert result.objective_scores == [
        {"recall_proxy": 1.0, "precision_proxy": 1.0},
        {"recall_proxy": 1.0, "precision_proxy": 1.0},
    ]
