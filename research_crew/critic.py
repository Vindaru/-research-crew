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
from research_crew.tracing import langfuse_callbacks

CRITIC_MODEL = os.getenv("CRITIC_MODEL", "claude-haiku-5-5")


def _normalize(text: str) -> str:
    """Ignore differences in spacing, case and curly quotes, nothing else."""
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip().lower()


def code_check(finding: Finding, sources: list[Source]) -> Verdict:
    pages = {s.url: s.content for s in sources}
    if finding.source_url not in pages:
        return Verdict(finding=finding, ok=False, reason="it cites a page the search never returned")
    if not finding.quote.strip() or _normalize(finding.quote) not in _normalize(pages[finding.source_url]):
        return Verdict(finding=finding, ok=False, reason="its quote does not appear word for word in the cited page")
    return Verdict(finding=finding, ok=True, reason="quote found in the cited page")


def critique_fake(notes: ResearchNotes, sources: list[Source], callbacks=None) -> Critique:
    """Offline critic: the code check only."""
    return Critique(verdicts=[code_check(f, sources) for f in notes.findings])


class _SupportCheck(BaseModel):
    index: int
    supported: bool
    reason: str = Field(description="One sentence")


class _SupportChecks(BaseModel):
    checks: list[_SupportCheck]


SUPPORT_PROMPT = """You are a strict fact-checking critic. For each numbered item you get a CLAIM and a QUOTE.
Decide only this: does the QUOTE by itself fully support the CLAIM? Do not use outside knowledge.
If the claim adds anything the quote does not say (a name, a number, a date, a stronger word), it is NOT supported."""


def critique(notes: ResearchNotes, sources: list[Source], callbacks=None) -> Critique:
    """Real critic: code check first, then one Claude call to judge support for the findings that passed it."""
    from langchain_anthropic import ChatAnthropic

    verdicts = [code_check(f, sources) for f in notes.findings]
    to_judge = [i for i, v in enumerate(verdicts) if v.ok]
    if to_judge:
        items = "\n\n".join(f"[{i}] CLAIM: {verdicts[i].finding.claim}\n    QUOTE: {verdicts[i].finding.quote}"
                            for i in to_judge)
        llm = ChatAnthropic(model=CRITIC_MODEL).with_structured_output(_SupportChecks)
        result = llm.invoke([("system", SUPPORT_PROMPT), ("user", items)],
                            config={"callbacks": callbacks if callbacks is not None else langfuse_callbacks(),
                                    "run_name": "critic"})
        for check in result.checks:
            if check.index in to_judge and not check.supported:
                verdicts[check.index] = Verdict(finding=verdicts[check.index].finding, ok=False,
                                                reason=f"the quote does not support the claim: {check.reason}")
    return Critique(verdicts=verdicts)
