"""What did that run cost? Counts every token the agents used and turns it into dollars.

Every LLM call made through LangChain reports its token usage; a shared UsageMetadataCallbackHandler adds them up
per model. Prices are Anthropic's list prices in US$ per million tokens (checked 2026-11-03; update if they change).
Thinking tokens are billed as output tokens and are already included in output_tokens.

Not counted here: the LLM judge (DeepEval calls Claude directly, about 1-2 cents per graded case with Sonnet)
and Tavily searches (free plan credits; see your Tavily dashboard).
"""
from langchain_core.callbacks import UsageMetadataCallbackHandler

PRICE_PER_MTOK = {  # model: (input, output)
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}

usage_tracker = UsageMetadataCallbackHandler()


def _price(model: str):
    for name, price in PRICE_PER_MTOK.items():
        if model.startswith(name):
            return price
    return None


def cost_summary(usage: dict | None = None) -> dict:
    """{'models': {model: {input_tokens, output_tokens, usd}}, 'usd': total}. Unknown models get usd=None."""
    usage = usage_tracker.usage_metadata if usage is None else usage
    models, total = {}, 0.0
    for model, u in usage.items():
        price = _price(model)
        usd = None if price is None else (u["input_tokens"] * price[0] + u["output_tokens"] * price[1]) / 1_000_000
        total += usd or 0.0
        models[model] = {"input_tokens": u["input_tokens"], "output_tokens": u["output_tokens"], "usd": usd}
    return {"models": models, "usd": total}


def format_cost(summary: dict) -> str:
    if not summary["models"]:
        return "Cost: $0 (no LLM calls)"
    lines = [f"Cost: ${summary['usd']:.4f}"]
    for model, m in summary["models"].items():
        usd = "unknown price" if m["usd"] is None else f"${m['usd']:.4f}"
        lines.append(f"  {model}: {m['input_tokens']:,} input + {m['output_tokens']:,} output tokens = {usd}")
    return "\n".join(lines)


def reset():
    usage_tracker.usage_metadata.clear()
