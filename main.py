"""Ask the research crew a question from your terminal.

  python main.py "What is retrieval-augmented generation?"          # real: Claude + Tavily (+ Langfuse if keys set)
  python main.py --fake "What is retrieval-augmented generation?"   # offline: no keys, no cost
  python main.py --researcher-only --fake "..."                     # M1: just the researcher's notes
"""
import argparse

from research_crew.graph import run_crew
from research_crew.researcher import research, research_fake


def print_notes(notes):
    print(f"\nQuestion: {notes.question}\n")
    for i, f in enumerate(notes.findings, 1):
        print(f"{i}. {f.claim}\n   source: {f.source_url}\n   quote:  \"{f.quote}\"\n")
    for gap in notes.gaps:
        print(f"GAP: {gap}")


def print_crew(state):
    critique = state["critique"]
    print(f"\nQuestion: {state['question']}")
    print(f"Researcher attempts: {state['attempts']}   "
          f"Critic: {len(state['approved'])} approved, {len(critique.rejected)} rejected on the last pass\n")
    for v in critique.rejected:
        print(f"  ✗ {v.finding.claim}\n    {v.reason}")
    report = state["report"]
    print(f"\n── Report ({report.status}) ──\n{report.text}\n")
    for n, url in enumerate(report.citations, 1):
        print(f"[{n}] {url}")


def main():
    parser = argparse.ArgumentParser(description="Research crew: researcher -> critic -> writer")
    parser.add_argument("question")
    parser.add_argument("--fake", action="store_true", help="offline mode: canned search results, no LLM")
    parser.add_argument("--researcher-only", action="store_true", help="M1: show only the researcher's notes")
    args = parser.parse_args()

    if args.researcher_only:
        print_notes((research_fake if args.fake else research)(args.question))
    else:
        print_crew(run_crew(args.question, fake=args.fake))


if __name__ == "__main__":
    main()
