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
