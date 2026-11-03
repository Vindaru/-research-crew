"""The shapes of data the agents pass to each other. Think of these as the forms on a clipboard.

Pydantic checks every field, so if the LLM returns something malformed we find out right away,
instead of three steps later.
"""
from pydantic import BaseModel, Field


class Source(BaseModel):
    """One web page the researcher read."""
    url: str
    title: str
    content: str = Field(description="The text snippet the search tool returned for this page")


class Finding(BaseModel):
    """One fact, plus the page that proves it. A finding without a source is not allowed."""
    claim: str = Field(description="One short factual sentence")
    source_url: str = Field(description="URL of the source that supports this claim, copied exactly")
    quote: str = Field(description="The exact words from that source that support the claim")


class ResearchNotes(BaseModel):
    """What the researcher hands to the critic (in M2): the question, the findings, and what it couldn't find."""
    question: str
    findings: list[Finding]
    gaps: list[str] = Field(default_factory=list, description="Parts of the question the sources did not answer")


class Verdict(BaseModel):
    """The critic's verdict on one finding."""
    finding: Finding
    ok: bool
    reason: str = Field(description="Why it passed or failed, in one sentence")


class Critique(BaseModel):
    """What the critic hands back: which findings survived, and feedback for the researcher if it must retry."""
    verdicts: list[Verdict]

    @property
    def approved(self) -> list[Finding]:
        return [v.finding for v in self.verdicts if v.ok]

    @property
    def rejected(self) -> list[Verdict]:
        return [v for v in self.verdicts if not v.ok]

    def feedback(self) -> str:
        return "\n".join(f"- REJECTED: {v.finding.claim!r} because {v.reason}" for v in self.rejected)


class Report(BaseModel):
    """The writer's output. `citations[n-1]` is the URL behind every [n] in the text."""
    question: str
    text: str
    citations: list[str]
    status: str = Field(description="'answered', or 'insufficient_evidence' when the critic approved nothing")
