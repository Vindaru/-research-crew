"""M2 tests: the critic catches bad homework, and the graph loops, retries and gives up correctly.

Every test here is offline. We build "bad researchers" on purpose (one that invents a quote, one that always lies)
because the interesting behaviour of a multi-agent system is what it does when one agent is wrong.
"""
import os

import pytest

from research_crew.critic import code_check, critique_fake
from research_crew.graph import build_graph, run_crew
from research_crew.models import Finding, ResearchNotes, Source
from research_crew.writer import cited_numbers, write_fake

PAGE = Source(url="https://example.org/a", title="A", content="LangGraph lets developers describe workflows as graphs.")
TRUE = Finding(claim="LangGraph describes workflows as graphs.", source_url=PAGE.url,
               quote="LangGraph lets developers describe workflows as graphs")
INVENTED_QUOTE = Finding(claim="LangGraph was released in 2019.", source_url=PAGE.url,
                         quote="LangGraph was released in 2019")
INVENTED_PAGE = Finding(claim="LangGraph is the fastest framework.", source_url="https://made-up.example",
                        quote="the fastest framework")


# ── the critic's code check ────────────────────────────────────────────────────────────
def test_critic_accepts_a_real_quote_from_a_real_page():
    assert code_check(TRUE, [PAGE]).ok


def test_critic_ignores_spacing_and_case_differences():
    sloppy = TRUE.model_copy(update={"quote": "  langgraph lets   developers describe workflows AS graphs "})
    assert code_check(sloppy, [PAGE]).ok


def test_critic_rejects_a_quote_that_is_not_on_the_page():
    verdict = code_check(INVENTED_QUOTE, [PAGE])
    assert not verdict.ok and "word for word" in verdict.reason


def test_critic_rejects_a_page_the_search_never_returned():
    verdict = code_check(INVENTED_PAGE, [PAGE])
    assert not verdict.ok and "never returned" in verdict.reason


# ── the graph: retry loop and honest give-up ──────────────────────────────────────────
def liar_then_honest():
    """Attempt 1 invents a quote. Attempt 2 (after the critic's feedback) fixes it."""
    calls = []

    def researcher(question, feedback=""):
        calls.append(feedback)
        findings = [TRUE, INVENTED_QUOTE] if len(calls) == 1 else [TRUE]
        return ResearchNotes(question=question, findings=findings), [PAGE]

    return researcher, calls


def test_rejected_finding_sends_work_back_to_the_researcher_with_feedback():
    researcher, calls = liar_then_honest()
    state = build_graph(researcher, critique_fake, write_fake).invoke({"question": "What is LangGraph?"})
    assert state["attempts"] == 2
    assert calls[0] == "" and "released in 2019" in calls[1], "retry must carry the critic's feedback"
    assert state["report"].status == "answered"
    assert "2019" not in state["report"].text, "a rejected claim must never reach the report"


def test_always_lying_researcher_stops_after_max_attempts_and_admits_it():
    def liar(question, feedback=""):
        return ResearchNotes(question=question, findings=[INVENTED_PAGE]), [PAGE]

    state = build_graph(liar, critique_fake, write_fake, max_attempts=3).invoke({"question": "Is LangGraph fast?"})
    assert state["attempts"] == 3
    assert state["report"].status == "insufficient_evidence"
    assert state["report"].citations == []


def test_partly_bad_research_still_reports_what_was_approved_when_attempts_run_out():
    def half_liar(question, feedback=""):
        return ResearchNotes(question=question, findings=[TRUE, INVENTED_QUOTE]), [PAGE]

    state = build_graph(half_liar, critique_fake, write_fake, max_attempts=2).invoke({"question": "q"})
    assert state["attempts"] == 2
    assert state["report"].status == "answered"
    assert "2019" not in state["report"].text


# ── the writer and the whole crew, offline ─────────────────────────────────────────────
def test_every_citation_number_in_the_report_points_to_a_url():
    report = write_fake("q", [TRUE])
    assert cited_numbers(report.text) and max(cited_numbers(report.text)) <= len(report.citations)


def test_fake_crew_answers_with_citations():
    state = run_crew("What is retrieval-augmented generation?", fake=True)
    report = state["report"]
    assert report.status == "answered" and len(report.citations) == 2
    assert cited_numbers(report.text) == {1, 2}


def test_fake_crew_says_not_enough_evidence_when_nothing_is_found():
    state = run_crew("Who won the 1987 chess olympiad?", fake=True)
    assert state["report"].status == "insufficient_evidence"


@pytest.mark.skipif(not (os.getenv("ANTHROPIC_API_KEY") and os.getenv("TAVILY_API_KEY")),
                    reason="needs ANTHROPIC_API_KEY and TAVILY_API_KEY")
def test_live_crew_report_cites_only_approved_sources():
    state = run_crew("What is retrieval-augmented generation and who introduced it?")
    report = state["report"]
    approved_urls = {f.source_url for f in state["approved"]}
    assert set(report.citations) <= approved_urls
    if report.status == "answered":
        assert cited_numbers(report.text), "an answered report must cite its sources"
