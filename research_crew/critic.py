"""M2: the Critic. It never searches and never writes. It only checks the researcher's homework.

Two layers, cheapest first (same idea as a test pyramid):
1. Code check (free, exact, always on): was the cited page really returned by the search,
   and does the quote appear in it word for word? This catches invented sources and invented quotes.
2. LLM check (real mode only): the quote is real, but does it actually SUPPORT the claim?
   e.g. quote "RAG was introduced in 2020" does not support the claim "RAG was invented by Google".
"""
import os
import re

from pydantic import BaseModel, Field

from research_crew.models import Critique, Finding, ResearchNotes, Source, Verdict
from research_crew.tracing import AGENT_EFFORT, AGENT_MAX_TOKENS, langfuse_callbacks

CRITIC_MODEL = os.getenv("CRITIC_MODEL", "claude-haiku-5-5")


def _normalize(text: str) -> str:
    """Ignore differences in spacing, case and curly quotes, nothing else."""
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip().lower()


def code_check(finding: Finding, sources: list[Source]) -> Verdict:
    # The same URL can come back from several searches with DIFFERENT snippets (Tavily returns the part of the
    # page that matches each query). So a quote is real if it appears in ANY snippet we saw for that URL.
    # (Bug found by the first live run: a dict {url: content} kept only the last snippet.)
    snippets = [s.content for s in sources if s.url == finding.source_url]
    if not snippets:
        return Verdict(finding=finding, ok=False, reason="it cites a page the search never returned")
    quote = _normalize(finding.quote)
    if not quote or not any(quote in _normalize(text) for text in snippets):
        return Verdict(finding=finding, ok=False, reason="its quote does not appear word for word in the cited page")
    return Verdict(finding=finding, ok=True, reason="quote found in the cited page")


def critique_fake(notes: ResearchNotes, sources: list[Source], callbacks=None) -> Critique:
    """Offline critic: the code check only."""
    return Critique(verdicts=[code_check(f, sources) for f in notes.findings])


class SupportCheck(BaseModel):
    """Deliberately flat: two plain fields. The first live run showed that asking for a nested list
    (checks: [{index, supported, reason}, ...]) in one call sometimes came back malformed and crashed the run."""
    supported: bool
    reason: str = Field(description="One sentence")


SUPPORT_PROMPT = """You are a strict fact-checking critic. You get one CLAIM and one QUOTE.
Decide only this: does the QUOTE by itself fully support the CLAIM? Do not use outside knowledge.
If the claim adds anything the quote does not say (a name, a number, a date, a stronger word), it is NOT supported."""


def support_check(finding: Finding, llm, config) -> Verdict:
    """One small Claude call per finding. Fails CLOSED: if the judge can't answer, the finding is rejected,
    never waved through, because an unchecked citation is exactly what this project exists to stop."""
    for _ in range(2):  # one retry for a malformed answer
        try:
            result = llm.invoke([("system", SUPPORT_PROMPT),
                                 ("user", f"CLAIM: {finding.claim}\nQUOTE: {finding.quote}")], config=config)
            break
        except Exception as error:  # noqa: BLE001  (bad structured output, rate limit, network)
            last_error = error
    else:
        return Verdict(finding=finding, ok=False, reason=f"the critic could not judge it ({type(last_error).__name__})")
    if not result.supported:
        return Verdict(finding=finding, ok=False, reason=f"the quote does not support the claim: {result.reason}")
    return Verdict(finding=finding, ok=True, reason="quote found and supports the claim")


def critique(notes: ResearchNotes, sources: list[Source], callbacks=None, llm=None) -> Critique:
    """Real critic: free code check first, then a Claude support check only for findings that passed it."""
    if llm is None:
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(model=CRITIC_MODEL, max_tokens=AGENT_MAX_TOKENS, output_config={"effort": AGENT_EFFORT}).with_structured_output(SupportCheck)
    config = {"callbacks": callbacks if callbacks is not None else langfuse_callbacks(), "run_name": "critic"}
    verdicts = []
    for finding in notes.findings:
        verdict = code_check(finding, sources)
        verdicts.append(support_check(finding, llm, config) if verdict.ok else verdict)
    return Critique(verdicts=verdicts)
