"""Langfuse tracing: the security camera. If Langfuse keys are set, every LLM call and tool call is recorded.

No keys -> no camera, and the agent still works. Tracing must never be the reason the app breaks.
"""
import logging
import os

# Cost controls for every agent call (see research_crew/cost.py for the budget guard).
# Haiku 5.5 thinks by default at "medium" effort and thinking is billed as output; "low" keeps answers
# short and cheap, which is plenty for searching, checking a quote, and writing 200 words.
AGENT_EFFORT = os.getenv("AGENT_EFFORT", "low")
AGENT_MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", "4096"))  # the SDK default is 128,000 per call


def langfuse_callbacks() -> list:
    """Default callbacks for every agent call: the cost counter always, Langfuse when keys are set."""
    from research_crew.cost import usage_tracker

    if not os.getenv("LANGFUSE_PUBLIC_KEY"):
        logging.getLogger("langfuse").setLevel(logging.CRITICAL)
        return [usage_tracker]
    from langfuse.langchain import CallbackHandler

    return [usage_tracker, CallbackHandler()]
