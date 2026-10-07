from gepa_optimizer.report import word_diff_markdown


def test_word_diff_highlights_only_changed_words():
    diff = word_diff_markdown(
        "You are a question-answering assistant. Answer in one short sentence.",
        "You are a question-answering assistant. Answer with ONLY the direct answer, no extra words.",
    )
    assert diff.startswith("You are a question-answering assistant. Answer")
    assert "~~in one short sentence.~~" in diff
    assert "**with ONLY the direct answer, no extra words.**" in diff


def test_word_diff_identical_text_has_no_markup():
    text = "Respond with exactly one word."
    diff = word_diff_markdown(text, text)
    assert "~~" not in diff
    assert "**" not in diff
    assert diff == text


def test_word_diff_total_replacement():
    diff = word_diff_markdown("Decide whether it is true.", "Say YES or NO only.")
    assert "~~Decide whether it is true.~~" in diff
    assert "**Say YES or NO only.**" in diff
