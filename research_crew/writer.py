"""M2: the Writer. It turns ONLY critic-approved findings into a short report with [n] citations.

The writer never sees the web and never sees rejected findings, so it has nothing to make things up from.
"""
import os
import re

from research_crew.models import Finding, Report
from research_crew.tracing import langfuse_callbacks

WRITER_MODEL = os.getenv("WRITER_MODEL", "claude-haiku-5-5")

WRITER_PROMPT = """You are the Writer on a research team. Write a short answer (under 200 words) to the question,
using ONLY the numbered findings given. After every sentence put the citation number(s) it relies on, like [1] or [1][3].
Do not add any fact that is not in the findings. If the findings only partly answer the question, say what is missing."""


def _citations(findings: list[Finding]) -> list[str]:
    """One number per unique URL, in order of first appearance."""
    urls: list[str] = []
    for f in findings:
        if f.source_url not in urls:
            urls.append(f.source_url)
    return urls


def write_fake(question: str, findings: list[Finding], gaps: list[str] | None = None, callbacks=None) -> Report:
    urls = _citations(findings)
    text = " ".join(f"{f.claim} [{urls.index(f.source_url) + 1}]" for f in findings)
    return Report(question=question, text=text, citations=urls, status="answered")


def write(question: str, findings: list[Finding], gaps: list[str] | None = None, callbacks=None) -> Report:
    from langchain_anthropic import ChatAnthropic

    urls = _citations(findings)
    numbered = "\n".join(f"[{urls.index(f.source_url) + 1}] {f.claim}" for f in findings)
    missing = "\n".join(f"- {g}" for g in gaps or [])
    user = f"QUESTION: {question}\n\nFINDINGS:\n{numbered}" + (f"\n\nKNOWN GAPS:\n{missing}" if missing else "")
    reply = ChatAnthropic(model=WRITER_MODEL).invoke(
        [("system", WRITER_PROMPT), ("user", user)],
        config={"callbacks": callbacks if callbacks is not None else langfuse_callbacks(), "run_name": "writer"})
    text = reply.content if isinstance(reply.content, str) else "".join(
        b.get("text", "") for b in reply.content if isinstance(b, dict))
    return Report(question=question, text=text.strip(), citations=urls, status="answered")


def insufficient_evidence(question: str, gaps: list[str], feedback: str) -> Report:
    """The honest ending: nothing survived the critic, so we say so instead of guessing."""
    text = "Not enough evidence to answer this reliably."
    if gaps:
        text += " Could not find: " + "; ".join(gaps) + "."
    if feedback:
        text += " Rejected findings:\n" + feedback
    return Report(question=question, text=text, citations=[], status="insufficient_evidence")


def cited_numbers(text: str) -> set[int]:
    """All the [n] numbers used in a report's text."""
    return {int(n) for n in re.findall(r"\[(\d+)\]", text)}
