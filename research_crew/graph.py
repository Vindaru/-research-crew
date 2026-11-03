"""M2: the crew, wired together as a LangGraph graph.

          ┌──────────────┐
 START ──►│  researcher  │◄──────────────┐
          └──────┬───────┘               │ some findings rejected
                 ▼                       │ AND attempts left
          ┌──────────────┐               │
          │    critic    │───────────────┘
          └──────┬───────┘
     approved    │      nothing approved, no attempts left
     ┌───────────┴────────────┐
     ▼                        ▼
 ┌────────┐            ┌──────────────┐
 │ writer │            │   give_up    │  "not enough evidence"
 └───┬────┘            └──────┬───────┘
     └──────────► END ◄───────┘

LangGraph in one sentence: nodes are functions that read and update a shared `state` dict,
and edges (some conditional) decide which node runs next. Loops are just edges that point backwards.
"""
from typing import Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from research_crew.critic import critique, critique_fake
from research_crew.models import Critique, Finding, Report, ResearchNotes, Source
from research_crew.researcher import research_fake_with_sources, research_with_sources
from research_crew.tracing import langfuse_callbacks
from research_crew.writer import insufficient_evidence, write, write_fake

MAX_ATTEMPTS = 3


class CrewState(TypedDict, total=False):
    question: str
    attempts: int                 # how many times the researcher has run
    notes: ResearchNotes          # latest research attempt
    sources: list[Source]         # every page the researcher has seen, across attempts
    critique: Critique            # latest verdicts
    approved: list[Finding]       # findings that passed the critic, kept across attempts
    report: Report


def build_graph(researcher: Callable = research_with_sources, critic: Callable = critique,
                writer: Callable = write, max_attempts: int = MAX_ATTEMPTS):
    """Pass in the three workers, so tests can swap in fakes (or deliberately bad ones)."""

    def research_node(state: CrewState) -> CrewState:
        feedback = state["critique"].feedback() if state.get("critique") else ""
        notes, seen = researcher(state["question"], feedback=feedback)
        return {"notes": notes, "sources": state.get("sources", []) + seen,
                "attempts": state.get("attempts", 0) + 1}

    def critic_node(state: CrewState) -> CrewState:
        result = critic(state["notes"], state["sources"])
        approved = list(state.get("approved", []))
        known = {f.claim for f in approved}
        approved += [f for f in result.approved if f.claim not in known]
        return {"critique": result, "approved": approved}

    def after_critic(state: CrewState) -> str:
        if state["critique"].rejected and state["attempts"] < max_attempts:
            return "researcher"          # send it back with the critic's feedback
        return "writer" if state["approved"] else "give_up"

    def write_node(state: CrewState) -> CrewState:
        return {"report": writer(state["question"], state["approved"], state["notes"].gaps)}

    def give_up_node(state: CrewState) -> CrewState:
        return {"report": insufficient_evidence(state["question"], state["notes"].gaps,
                                                state["critique"].feedback())}

    graph = StateGraph(CrewState)
    graph.add_node("researcher", research_node)
    graph.add_node("critic", critic_node)
    graph.add_node("writer", write_node)
    graph.add_node("give_up", give_up_node)
    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "critic")
    graph.add_conditional_edges("critic", after_critic, ["researcher", "writer", "give_up"])
    graph.add_edge("writer", END)
    graph.add_edge("give_up", END)
    return graph.compile()


def run_crew(question: str, fake: bool = False) -> CrewState:
    """Run the whole crew once. fake=True: offline, free, deterministic."""
    if fake:
        app = build_graph(research_fake_with_sources, critique_fake, write_fake)
        return app.invoke({"question": question})
    app = build_graph()
    return app.invoke({"question": question},
                      config={"callbacks": langfuse_callbacks(), "run_name": "research-crew"})
