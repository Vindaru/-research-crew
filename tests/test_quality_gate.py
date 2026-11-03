"""Tests for the CI quality gate, including a "sabotage" test: break the critic on purpose, and the gate must catch it.

This is mutation testing in miniature. A gate that never fails is decoration; these tests prove it bites.
"""
from evals.evaluators import DETERMINISTIC, crew_output
from evals.run_evals import as_item, load_golden_set, quality_gate
from research_crew.graph import build_graph
from research_crew.models import Critique, Finding, ResearchNotes, Source, Verdict
from research_crew.writer import write_fake


def row(passed, faithfulness=1.0):
    return {"id": "x", "category": "c", "passed": passed, "reasons": [],
            "scores": {"citation_faithfulness": faithfulness}}


def test_gate_passes_at_threshold():
    rows = [row(True)] * 9 + [row(False)]
    assert quality_gate(rows, total=10, min_pass_rate=0.9, min_faithfulness=1.0)["passed"]


def test_gate_fails_below_pass_rate():
    rows = [row(True)] * 8 + [row(False)] * 2
    gate = quality_gate(rows, total=10, min_pass_rate=0.9, min_faithfulness=1.0)
    assert not gate["passed"] and "80%" in gate["message"]


def test_gate_fails_on_any_unfaithful_citation_even_if_cases_pass():
    rows = [row(True)] * 9 + [row(True, faithfulness=0.5)]
    assert not quality_gate(rows, total=10, min_pass_rate=0.9, min_faithfulness=1.0)["passed"]


def test_crashed_cases_count_as_failures():
    rows = [row(True)] * 9  # the 10th case crashed and has no row
    assert quality_gate(rows, total=10, min_pass_rate=1.0, min_faithfulness=1.0)["pass_rate"] == 0.9


def test_sabotaged_critic_is_caught_by_the_evals():
    """Someone 'simplifies' the critic so it approves everything. The researcher invents a quote.
    The independent citation_faithfulness grader must catch it, and so must the gate."""
    page = Source(url="https://a.example", title="A", content="LangGraph lets developers describe workflows as graphs.")
    invented = Finding(claim="LangGraph is used by every Fortune 500 company.", source_url=page.url,
                       quote="used by every Fortune 500 company")

    def lying_researcher(question, feedback=""):
        return ResearchNotes(question=question, findings=[invented]), [page]

    def rubber_stamp_critic(notes, sources):
        return Critique(verdicts=[Verdict(finding=f, ok=True, reason="looks fine") for f in notes.findings])

    crew = build_graph(lying_researcher, rubber_stamp_critic, write_fake)
    case = as_item(load_golden_set()[1])  # any answered case
    output = crew_output(crew.invoke({"question": case["input"]}))
    scores = {g.__name__: g(output=output, expected_output=case["expected_output"]).value for g in DETERMINISTIC}
    assert scores["citation_faithfulness"] == 0.0
    gate = quality_gate([{"passed": False, "scores": scores}], total=1, min_pass_rate=0.9, min_faithfulness=1.0)
    assert not gate["passed"]
