"""Tests for M1. The offline ones run in CI for free; the live one runs only when API keys are set.

What we test is the contract M2 depends on: every finding points at a real source and quotes it word for word.
"""
import os

import pytest

from research_crew import researcher
from research_crew.models import Source
from research_crew.search import fake_search
from research_crew.researcher import research, research_fake


def test_fake_research_returns_findings_with_sources():
    notes = research_fake("What is retrieval-augmented generation?")
    assert len(notes.findings) >= 2
    assert notes.gaps == []
    for f in notes.findings:
        assert f.source_url.startswith("https://")


def test_every_quote_appears_word_for_word_in_its_source():
    question = "How do agents use LangGraph?"
    pages = {s.url: s.content for s in fake_search(question)}
    notes = research_fake(question)
    for f in notes.findings:
        assert f.source_url in pages, f"finding cites a page the search never returned: {f.source_url}"
        assert f.quote in pages[f.source_url], f"quote not in source: {f.quote!r}"


def test_no_sources_means_a_gap_not_a_guess():
    notes = research_fake("Who won the 1987 chess olympiad?")
    assert notes.findings == []
    assert notes.gaps, "with no sources the researcher must report a gap"


def test_web_search_tool_formats_results_for_the_llm(monkeypatch):
    monkeypatch.setattr(researcher, "tavily_search",
                        lambda q: [Source(url="https://a.example", title="A", content="Alpha.")])
    text = researcher.web_search.invoke({"query": "anything"})
    assert "URL: https://a.example" in text and "CONTENT: Alpha." in text


def test_web_search_tool_says_so_when_empty(monkeypatch):
    monkeypatch.setattr(researcher, "tavily_search", lambda q: [])
    assert researcher.web_search.invoke({"query": "anything"}) == "No results."


@pytest.mark.skipif(not (os.getenv("ANTHROPIC_API_KEY") and os.getenv("TAVILY_API_KEY")),
                    reason="needs ANTHROPIC_API_KEY and TAVILY_API_KEY")
def test_live_researcher_cites_every_finding():
    notes = research("What is retrieval-augmented generation and who introduced it?")
    assert notes.findings
    for f in notes.findings:
        assert f.source_url.startswith("http") and f.quote.strip()
