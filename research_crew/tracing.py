"""Langfuse tracing: the security camera. If Langfuse keys are set, every LLM call and tool call is recorded.

No keys -> no camera, and the agent still works. Tracing must never be the reason the app breaks.
"""
import logging
import os


def langfuse_callbacks() -> list:
    """Default callbacks for every agent call: the cost counter always, Langfuse when keys are set."""
    from research_crew.cost import usage_tracker

    if not os.getenv("LANGFUSE_PUBLIC_KEY"):
        logging.getLogger("langfuse").setLevel(logging.CRITICAL)
        return [usage_tracker]
    from langfuse.langchain import CallbackHandler

    return [usage_tracker, CallbackHandler()]
