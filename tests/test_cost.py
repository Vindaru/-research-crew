"""The cost counter: tokens per model in, dollars out."""
from research_crew.cost import cost_summary, format_cost


def test_cost_uses_per_model_prices():
    usage = {"claude-haiku-5-5": {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
             "claude-sonnet-5-5": {"input_tokens": 100_000, "output_tokens": 10_000}}
    summary = cost_summary(usage, judge_usd=0.0)
    assert round(summary["models"]["claude-haiku-5-5"]["usd"], 4) == 0.60   # 0.10 + 0.50
    assert round(summary["models"]["claude-sonnet-5-5"]["usd"], 4) == 0.30  # 0.20 + 0.10
    assert round(summary["usd"], 4) == 0.90


def test_unknown_model_is_reported_not_guessed():
    summary = cost_summary({"some-new-model": {"input_tokens": 10, "output_tokens": 10}}, judge_usd=0.0)
    assert summary["models"]["some-new-model"]["usd"] is None
    assert "unknown price" in format_cost(summary)


def test_no_calls_costs_nothing():
    assert format_cost(cost_summary({}, judge_usd=0.0)) == "Cost: $0 (no LLM calls)"


def test_budget_guard_stops_the_next_llm_call(monkeypatch):
    """Once spending passes RESEARCH_CREW_MAX_USD, the next model call raises instead of running."""
    import pytest
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    from research_crew import cost
    from research_crew.tracing import langfuse_callbacks

    monkeypatch.setenv("RESEARCH_CREW_MAX_USD", "0.01")
    cost.reset()
    big = AIMessage(content="x", usage_metadata={"input_tokens": 200_000, "output_tokens": 0, "total_tokens": 200_000},
                    response_metadata={"model_name": "claude-haiku-5-5"})  # 200k * $0.10/M = $0.02
    model = GenericFakeChatModel(messages=iter([big, big]))
    model.invoke("first call runs", config={"callbacks": langfuse_callbacks()})
    with pytest.raises(cost.BudgetExceeded):
        model.invoke("second call is refused", config={"callbacks": langfuse_callbacks()})
    cost.reset()


def test_judge_cost_is_included():
    summary = cost_summary({}, judge_usd=0.25)
    assert summary["usd"] == 0.25 and "LLM judge" in format_cost(summary)


def test_plain_pytest_never_runs_live_tests():
    from tests import live
    import os
    if os.getenv("RUN_LIVE_TESTS") != "1":
        assert not live.LIVE
