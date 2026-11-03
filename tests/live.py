"""Live tests call Claude and Tavily and COST MONEY, so they never run by accident.

They run only when you ask for them explicitly:
    RUN_LIVE_TESTS=1 python -m pytest -q -m live
Having API keys in your environment is NOT enough on its own (that was the old rule, and it made a plain
`pytest` spend money). The budget guard (RESEARCH_CREW_MAX_USD, default $1) still applies to live runs.
"""
import os

import pytest

LIVE = os.getenv("RUN_LIVE_TESTS") == "1"
HAS_KEYS = bool(os.getenv("ANTHROPIC_API_KEY") and os.getenv("TAVILY_API_KEY"))

live = pytest.mark.skipif(not (LIVE and HAS_KEYS),
                          reason="live test (costs money): set RUN_LIVE_TESTS=1 and API keys to run")
