"""The cost counter: tokens per model in, dollars out."""
from research_crew.cost import cost_summary, format_cost


def test_cost_uses_per_model_prices():
    usage = {"claude-haiku-5-5": {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
             "claude-sonnet-5-5": {"input_tokens": 100_000, "output_tokens": 10_000}}
    summary = cost_summary(usage)
    assert round(summary["models"]["claude-haiku-5-5"]["usd"], 4) == 0.60   # 0.10 + 0.50
    assert round(summary["models"]["claude-sonnet-5-5"]["usd"], 4) == 0.30  # 0.20 + 0.10
    assert round(summary["usd"], 4) == 0.90


def test_unknown_model_is_reported_not_guessed():
    summary = cost_summary({"some-new-model": {"input_tokens": 10, "output_tokens": 10}})
    assert summary["models"]["some-new-model"]["usd"] is None
    assert "unknown price" in format_cost(summary)


def test_no_calls_costs_nothing():
    assert format_cost(cost_summary({})) == "Cost: $0 (no LLM calls)"
