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
from research_crew.tracing import AGENT_EFFORT, AGENT_MAX_TOKENS, langfuse_callbacks

RESEARCHER_MODEL = os.getenv("RESEARCHER_MODEL", "claude-haiku-5-5")
MAX_SEARCHES = 3

SYSTEM_PROMPT = f"""You are the Researcher on a research team. You collect facts; you do not write the final report.
Rules:
1. Use the web_search tool to find sources. Search at most {MAX_SEARCHES} times.
2. Every finding must be supported by a source you actually received from web_search.
   Copy its URL exactly and quote the exact supporting words from its content.
3. Never use your own memory as a source. If the sources do not answer part of the question, list it in gaps.
4. Prefer 3 to 6 findings. Each claim is one short factual sentence."""


def format_results(results: list[Source]) -> str:
    return "\n\n".join(f"URL: {s.url}\nTITLE: {s.title}\nCONTENT: {s.content}" for s in results) or "No results."


@tool
def web_search(query: str) -> str:
    """Search the web. Returns up to 5 pages, each with url, title and a content snippet."""
    return format_results(tavily_search(query))


def make_recording_search(seen: list[Source]):
    """Same tool, but every page it returns is also written into `seen`.

    That list is the evidence locker: the critic only trusts quotes that appear in a page from it.
    """
    @tool("web_search")
    def recording_search(query: str) -> str:
        """Search the web. Returns up to 5 pages, each with url, title and a content snippet."""
        results = tavily_search(query)
        seen.extend(results)
        return format_results(results)

    return recording_search


def research(question: str, callbacks=None) -> ResearchNotes:
    """Run the real researcher (Claude + Tavily). M1 entry point: notes only."""
    notes, _ = research_with_sources(question, callbacks=callbacks)
    return notes


def research_with_sources(question: str, feedback: str = "", callbacks=None) -> tuple[ResearchNotes, list[Source]]:
    """Run the real researcher and also return every page it was shown. Used by the M2 graph.

    `feedback` is the critic's list of rejected findings from the previous attempt, if any.
    """
    from langchain_anthropic import ChatAnthropic

    seen: list[Source] = []
    from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware

    # Hard caps, enforced by code (the prompt's "at most 3 searches" is only a request):
    #   - at most MAX_SEARCHES web searches per attempt, then the agent must answer with what it has
    #   - at most MAX_SEARCHES + 3 model calls per attempt, then the run ends
    agent = create_agent(ChatAnthropic(model=RESEARCHER_MODEL, max_tokens=AGENT_MAX_TOKENS, output_config={"effort": AGENT_EFFORT}),
                         tools=[make_recording_search(seen)], system_prompt=SYSTEM_PROMPT,
                         response_format=ResearchNotes,
                         middleware=[ToolCallLimitMiddleware(run_limit=MAX_SEARCHES, exit_behavior="continue"),
                                     ModelCallLimitMiddleware(run_limit=MAX_SEARCHES + 3, exit_behavior="end")])
    prompt = question if not feedback else (
        f"{question}\n\nYour previous findings were checked and these were rejected:\n{feedback}\n"
        "Search again and replace them with findings whose quotes appear word for word in a source.")
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]},
                          config={"callbacks": callbacks if callbacks is not None else langfuse_callbacks(),
                                  "run_name": "researcher"})
    return result["structured_response"], seen


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


def research_fake_with_sources(question: str, feedback: str = "", callbacks=None):
    """Offline version of research_with_sources, for the M2 graph."""
    return research_fake(question), fake_search(question)
