"""Web search: the researcher's only way to learn new facts.

Two versions with the same signature:
- tavily_search: the real web, via Tavily (an API built for LLM agents, free tier 1,000 searches/month)
- fake_search:   canned pages from tests/fixtures, so tests run offline, free and identical every time
"""
import json
import os
import re
from pathlib import Path

from research_crew.models import Source

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "search_results.json"


def tavily_search(query: str, max_results: int = 5) -> list[Source]:
    from tavily import TavilyClient

    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    response = client.search(query, max_results=max_results)
    return [Source(url=r["url"], title=r["title"], content=r["content"]) for r in response["results"]]


def fake_search(query: str, max_results: int = 5) -> list[Source]:
    """Return the fixture pages whose keywords appear in the query. No match -> no results (that's a test case too)."""
    pages = json.loads(FIXTURES.read_text())
    words = set(re.findall(r"[a-z0-9-]+", query.lower()))
    hits = [p for p in pages if words & set(p["keywords"])]
    return [Source(url=p["url"], title=p["title"], content=p["content"]) for p in hits[:max_results]]
