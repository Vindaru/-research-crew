"""Tests for the graders themselves. If a grader is wrong, every score it gives is wrong, so graders get tests too.

Each test feeds a grader a hand-made output with one known defect and checks the grader catches it.
"""
import json

from evals.build_golden_set import CASES
from evals.evaluators import citation_faithfulness, citations_valid, crew_output, must_mention, must_not_say, status
from evals.run_evals import GOLDEN, as_item, load_golden_set
from research_crew.graph import run_crew

PAGE = {"url": "https://a.example", "title": "A", "content": "RAG gives models documents at question time."}
GOOD = {"claim": "RAG gives models documents.", "source_url": PAGE["url"],
        "quote": "RAG gives models documents at question time"}
FAKE = {"claim": "RAG was invented in 1999.", "source_url": PAGE["url"], "quote": "RAG was invented in 1999"}
EXPECTED = {"expected_status": "answered", "min_citations": 1, "must_mention": ["documents"],
            "must_not_say": ["impossible"], "reference_answer": "..."}


def output(answer="RAG gives models documents. [1]", approved=(GOOD,), citations=(PAGE["url"],), status="answered"):
    return {"answer": answer, "status": status, "citations": list(citations), "attempts": 1,
            "approved": list(approved), "sources": [PAGE]}


def test_good_output_passes_every_code_check():
    for grade in (status, citation_faithfulness, citations_valid, must_mention, must_not_say):
        assert grade(output=output(), expected_output=EXPECTED).value == 1.0, grade.__name__


def test_faithfulness_drops_when_a_quote_is_invented():
    result = citation_faithfulness(output=output(approved=(GOOD, FAKE)), expected_output=EXPECTED)
    assert result.value == 0.5 and "1999" in result.comment


def test_citation_number_with_no_url_is_caught():
    result = citations_valid(output=output(answer="RAG gives models documents. [2]"), expected_output=EXPECTED)
    assert result.value == 0.0 and "out of range" in result.comment


def test_citing_a_source_the_critic_never_approved_is_caught():
    result = citations_valid(output=output(citations=(PAGE["url"], "https://sneaky.example"),
                                           answer="x [1] y [2]"), expected_output=EXPECTED)
    assert result.value == 0.0 and "unapproved" in result.comment


def test_too_few_citations_is_caught():
    result = citations_valid(output=output(answer="No citations here."), expected_output=EXPECTED)
    assert result.value == 0.0 and "needed 1" in result.comment


def test_wrong_status_and_forbidden_words_are_caught():
    assert status(output=output(status="insufficient_evidence"), expected_output=EXPECTED).value == 0.0
    assert must_not_say(output=output(answer="Hallucinations are impossible [1]"), expected_output=EXPECTED).value == 0.0
    assert must_mention(output=output(answer="Something else [1]"), expected_output=EXPECTED).value == 0.0


def test_golden_set_file_matches_the_builder_and_is_well_formed():
    cases = load_golden_set()
    assert [c["id"] for c in cases] == [c["id"] for c in CASES], "run: python -m evals.build_golden_set"
    assert len({c["id"] for c in cases}) == len(cases), "duplicate ids"
    for c in cases:
        assert c["expected_status"] in {"answered", "insufficient_evidence"}
        assert c["reviewed_by"] and c["version"]
        as_item(c)  # every case converts to a Langfuse dataset item


def test_crew_output_is_plain_json():
    json.dumps(crew_output(run_crew("What is retrieval-augmented generation?", fake=True)))
