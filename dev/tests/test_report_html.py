from gepa_optimizer.report import markdown_to_html_body


def test_converts_headers_and_bold():
    html_out = markdown_to_html_body("# Title\n## Section\n**bold text**")
    assert "<h1>Title</h1>" in html_out
    assert "<h2>Section</h2>" in html_out
    assert "<strong>bold text</strong>" in html_out


def test_escapes_special_characters_in_content():
    html_out = markdown_to_html_body('Goal: has "quotes" & <tags> and an apostrophe\'s')
    assert "<tags>" not in html_out  # must not be interpreted as a real tag
    assert "&quot;" in html_out
    assert "&amp;" in html_out
    assert "&lt;tags&gt;" in html_out


def test_converts_table():
    md = "| A | B |\n|---|---|\n| 1 | 2 |"
    html_out = markdown_to_html_body(md)
    assert "<table>" in html_out
    assert "<th>A</th><th>B</th>" in html_out
    assert "<td>1</td><td>2</td>" in html_out


def test_converts_fenced_code_block_and_escapes_its_content():
    md = "```\nrow['label'] <- test\n```"
    html_out = markdown_to_html_body(md)
    assert "<pre><code>" in html_out
    assert "&lt;-" in html_out  # the "<-" inside the code block must be escaped too


def test_passes_through_details_summary_while_inline_processing_summary_text():
    md = "<details><summary>Candidate 0 **(best)** -- score 1.0</summary>\nbody\n</details>"
    html_out = markdown_to_html_body(md)
    assert "<details><summary>Candidate 0 <strong>(best)</strong> -- score 1.0</summary>" in html_out
    assert "</details>" in html_out


def test_word_diff_markup_survives_as_strikethrough_and_strong():
    html_out = markdown_to_html_body("~~removed words~~ **added words**")
    assert "<del>removed words</del>" in html_out
    assert "<strong>added words</strong>" in html_out
