"""Ask the research crew a question from your terminal.

  python main.py "What is retrieval-augmented generation?"          # real: Claude + Tavily (+ Langfuse if keys set)
  python main.py --fake "What is retrieval-augmented generation?"   # offline: no keys, no cost
"""
import argparse

from research_crew.researcher import research, research_fake


def main():
    parser = argparse.ArgumentParser(description="Research crew (M1: researcher only)")
    parser.add_argument("question")
    parser.add_argument("--fake", action="store_true", help="offline mode: canned search results, no LLM")
    args = parser.parse_args()

    notes = (research_fake if args.fake else research)(args.question)
    print(f"\nQuestion: {notes.question}\n")
    for i, f in enumerate(notes.findings, 1):
        print(f"{i}. {f.claim}\n   source: {f.source_url}\n   quote:  \"{f.quote}\"\n")
    for gap in notes.gaps:
        print(f"GAP: {gap}")


if __name__ == "__main__":
    main()
