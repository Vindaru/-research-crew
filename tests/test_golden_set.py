"""The golden set as a pytest suite against the REAL crew: one test per case. COSTS MONEY, opt-in only.

  RUN_LIVE_TESTS=1 python -m pytest -q tests/test_golden_set.py
Prefer `python -m evals.run_evals` for a priced run: same cases, same graders, plus a scorecard and total cost.
"""
import pytest

from evals.evaluators import DETERMINISTIC, crew_output, llm_judge
from evals.run_evals import JUDGE_PASS, as_item, load_golden_set
from research_crew.graph import run_crew
from tests.live import live

pytestmark = [pytest.mark.live, live]


@pytest.mark.parametrize("case", load_golden_set(), ids=lambda c: c["id"])
def test_crew_matches_golden_case(case):
    item = as_item(case)
    output = crew_output(run_crew(item["input"]))
    for grade in DETERMINISTIC:
        result = grade(input=item["input"], output=output, expected_output=item["expected_output"])
        assert result.value == 1.0, f"{result.name}: {result.comment}\nanswer: {output['answer']}"
    verdict = llm_judge(input=item["input"], output=output, expected_output=item["expected_output"])
    assert verdict.value >= JUDGE_PASS, f"judge {verdict.value}: {verdict.comment}"
