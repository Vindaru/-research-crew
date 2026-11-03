"""What does a run cost, and a hard stop before it costs too much.

Every LLM call made through LangChain reports its token usage; `usage_tracker` adds it up per model.
The LLM judge (DeepEval) reports its own dollar cost, which is added with `add_judge_cost`.
Prices are Anthropic's list prices in US$ per million tokens (checked 2026-11-03; update if they change).
Thinking tokens are billed as output tokens and are already included in output_tokens.

BUDGET GUARD: once this process has spent more than RESEARCH_CREW_MAX_USD (default $1.00), the next LLM call
raises BudgetExceeded and the run stops. Added after a live test run cost far more than estimated.
Tavily searches are not counted here (free plan credits; see app.tavily.com -> Usage).
"""
import os
import threading

from langchain_core.callbacks import UsageMetadataCallbackHandler

PRICE_PER_MTOK = {  # model: (input, output)
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}
DEFAULT_MAX_USD = 1.00


class BudgetExceeded(RuntimeError):
    pass


def max_usd() -> float:
    return float(os.getenv("RESEARCH_CREW_MAX_USD", DEFAULT_MAX_USD))


class _BudgetedUsage(UsageMetadataCallbackHandler):
    """Counts tokens like its parent, and refuses to start a new LLM call once the budget is spent."""
    raise_error = True  # make LangChain propagate BudgetExceeded instead of just logging it

    def __init__(self):
        super().__init__()
        self.judge_usd = 0.0
        self._judge_lock = threading.Lock()

    def on_chat_model_start(self, *args, **kwargs):
        check_budget()

    def on_llm_start(self, *args, **kwargs):
        check_budget()


usage_tracker = _BudgetedUsage()


def _price(model: str):
    for name, price in PRICE_PER_MTOK.items():
        if model.startswith(name):
            return price
    return None


def add_judge_cost(usd: float | None):
    with usage_tracker._judge_lock:
        usage_tracker.judge_usd += usd or 0.0


def cost_summary(usage: dict | None = None, judge_usd: float | None = None) -> dict:
    """{'models': {model: {input_tokens, output_tokens, usd}}, 'judge_usd': x, 'usd': total}."""
    usage = usage_tracker.usage_metadata if usage is None else usage
    judge_usd = usage_tracker.judge_usd if judge_usd is None else judge_usd
    models, total = {}, judge_usd
    for model, u in usage.items():
        price = _price(model)
        usd = None if price is None else (u["input_tokens"] * price[0] + u["output_tokens"] * price[1]) / 1_000_000
        total += usd or 0.0
        models[model] = {"input_tokens": u["input_tokens"], "output_tokens": u["output_tokens"], "usd": usd}
    return {"models": models, "judge_usd": judge_usd, "usd": total}


def check_budget():
    spent = cost_summary()["usd"]
    if spent >= max_usd():
        raise BudgetExceeded(f"Budget reached: spent ${spent:.2f} of ${max_usd():.2f} "
                             f"(raise RESEARCH_CREW_MAX_USD to allow more)")


def format_cost(summary: dict) -> str:
    if not summary["models"] and not summary["judge_usd"]:
        return "Cost: $0 (no LLM calls)"
    lines = [f"Cost: ${summary['usd']:.4f}  (budget ${max_usd():.2f})"]
    for model, m in summary["models"].items():
        usd = "unknown price" if m["usd"] is None else f"${m['usd']:.4f}"
        lines.append(f"  {model}: {m['input_tokens']:,} input + {m['output_tokens']:,} output tokens = {usd}")
    if summary["judge_usd"]:
        lines.append(f"  LLM judge: ${summary['judge_usd']:.4f}")
    return "\n".join(lines)


def reset():
    usage_tracker.usage_metadata.clear()
    usage_tracker.judge_usd = 0.0
