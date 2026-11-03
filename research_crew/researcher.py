"""M1: the Researcher agent. Question in -> searches the web -> ResearchNotes out (facts, each with a source).

The agent loop (LangChain create_agent):
    think "what should I search?" -> call web_search -> read results -> maybe search again -> fill in ResearchNotes

Two rules make it testable later:
1. Every finding must quote a source it actually read (the critic in M2 will check this).
2. If the sources don't answer something, say so in `gaps` instead of guessing.
"""
import os

from langchain.agents import create_agent
from langchain_core.tools import tool

from research_crew.models import Finding, ResearchNotes, Source
from research_crew.search import fake_search, tavily_search
from research_crew.tracing import langfuse_callbacks

RESEARCHER_MODEL = os.getenv("RESEARCHER_MODEL", "claude-haiku-5-5")
MAX_SEARCHES = 3

SYSTEM_PROMPT = f"""You are the Researcher on a research team. You collect facts; you do not write the final report.
Rules:
1. Use the web_search tool to find sources. Search at most {MAX_SEARCHES} times.
2. Every finding must be supported by a source you actually received from web_search.
   Copy its URL exactly and quote the exact supporting words from its content.
3. Never use your own memory as a source. If the sources do not answer part of the question, list it in gaps.
4. Prefer 3 to 6 findings. Each claim is one short factual sentence."""


@tool
def web_search(query: str) -> str:
    """Search the web. Returns up to 5 pages, each with url, title and a content snippet."""
    results = tavily_search(query)
    return "\n\n".join(f"URL: {s.url}\nTITLE: {s.title}\nCONTENT: {s.content}" for s in results) or "No results."


def research(question: str, callbacks=None) -> ResearchNotes:
    """Run the real researcher (Claude + Tavily)."""
    from langchain_anthropic import ChatAnthropic

    agent = create_agent(ChatAnthropic(model=RESEARCHER_MODEL), tools=[web_search],
                         system_prompt=SYSTEM_PROMPT, response_format=ResearchNotes)
    result = agent.invoke({"messages": [{"role": "user", "content": question}]},
                          config={"callbacks": callbacks if callbacks is not None else langfuse_callbacks(),
                                  "run_name": "researcher"})
    return result["structured_response"]


def research_fake(question: str, callbacks=None) -> ResearchNotes:
    """Offline stand-in: no LLM, no internet. Takes the first sentence of each fixture page as a finding.

    Deliberately dumb, so the whole pipeline (and its tests) can run for free in CI.
    """
    sources: list[Source] = fake_search(question)
    findings = []
    for s in sources:
        first_sentence = s.content.split(". ")[0].rstrip(".") + "."
        findings.append(Finding(claim=first_sentence, source_url=s.url, quote=first_sentence.rstrip(".")))
    gaps = [] if findings else [f"No sources found for: {question}"]
    return ResearchNotes(question=question, findings=findings, gaps=gaps)
