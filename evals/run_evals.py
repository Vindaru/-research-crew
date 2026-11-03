"""End-to-end eval: every golden case -> the crew -> all graders -> a scorecard (and Langfuse, if running).

  python -m evals.run_evals --fake        # offline: fake crew, no API key, no cost (judge skipped)
  python -m evals.run_evals               # real crew + Claude judge; scores and traces go to Langfuse

Exits with code 1 if any case fails, so it can gate a CI pipeline (that's M4).
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

from langfuse import Langfuse

from evals.evaluators import DETERMINISTIC, crew_output, llm_judge
from research_crew.graph import run_crew

DATASET = "research-crew-golden-v1"
JUDGE_PASS = 0.7
GOLDEN = Path(__file__).with_name("golden_set.jsonl")
EXPECTED_KEYS = ("expected_status", "min_citations", "must_mention", "must_not_say", "reference_answer")


def load_golden_set(path=GOLDEN):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def as_item(case):
    return {"input": case["question"], "expected_output": {k: case[k] for k in EXPECTED_KEYS},
            "metadata": {"id": case["id"], "category": case["category"]}}


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--fake", action="store_true", help="use the offline fake crew")
    args.add_argument("--no-judge", action="store_true", help="skip the LLM judge")
    opts = args.parse_args()

    use_judge = not opts.fake and not opts.no_judge and bool(os.getenv("ANTHROPIC_API_KEY"))
    if not os.getenv("LANGFUSE_PUBLIC_KEY"):
        logging.getLogger("langfuse").setLevel(logging.CRITICAL)  # offline: no Langfuse server, keep output clean
    langfuse = Langfuse()
    online = bool(os.getenv("LANGFUSE_PUBLIC_KEY")) and langfuse.auth_check()
    items = [as_item(c) for c in load_golden_set()]

    def task(*, item, **kwargs):
        question = item["input"] if isinstance(item, dict) else item.input
        return crew_output(run_crew(question, fake=opts.fake))

    data = items
    if online:  # upload the golden set as a Langfuse dataset, so runs can be compared side by side
        langfuse.create_dataset(name=DATASET, description="Research crew golden set (properties, not exact answers)")
        for it in items:
            langfuse.create_dataset_item(dataset_name=DATASET, id=it["metadata"]["id"], input=it["input"],
                                         expected_output=it["expected_output"], metadata=it["metadata"])
        data = langfuse.get_dataset(DATASET).items

    result = langfuse.run_experiment(
        name="research-crew", run_name="fake" if opts.fake else "live",
        description="Golden set, code checks + LLM judge", data=data, task=task,
        evaluators=DETERMINISTIC + ([llm_judge] if use_judge else []), max_concurrency=2)
    langfuse.flush()

    failures = print_scorecard(result, use_judge)
    crashed = len(data) - len(result.item_results)  # a case whose task raised has no result at all
    if crashed:
        print(f"{crashed} case(s) crashed before grading; see the errors above")
    if result.experiment_url:
        print(f"Langfuse: {result.experiment_url}")
    sys.exit(1 if failures + crashed else 0)


CODE_CHECKS = [g.__name__ for g in DETERMINISTIC]
SHORT = {"status": "status", "citation_faithfulness": "faithful", "citations_valid": "cites_ok",
         "must_mention": "mentions", "must_not_say": "no_bad", "llm_judge": "judge"}


def passed(scores: dict, use_judge: bool) -> bool:
    return all(scores.get(n, 0) == 1.0 for n in CODE_CHECKS) and (not use_judge or scores.get("llm_judge", 0) >= JUDGE_PASS)


def print_scorecard(result, use_judge):
    names = CODE_CHECKS + (["llm_judge"] if use_judge else [])
    print(f"\n{'case':<20}" + "".join(f"{SHORT[n]:>10}" for n in names) + f"{'result':>8}")
    failures = 0
    for r in sorted(result.item_results, key=lambda r: (r.item["metadata"] if isinstance(r.item, dict)
                                                         else r.item.metadata)["id"]):
        meta = r.item["metadata"] if isinstance(r.item, dict) else r.item.metadata
        scores = {e.name: e.value for e in r.evaluations}
        ok = passed(scores, use_judge)
        failures += 0 if ok else 1
        print(f"{meta['id']:<20}" + "".join(f"{scores.get(n, 0):>10.2f}" for n in names) + f"{'PASS' if ok else 'FAIL':>8}")
        if not ok:
            for e in r.evaluations:
                if (e.name != "llm_judge" and e.value < 1.0) or (e.name == "llm_judge" and e.value < JUDGE_PASS):
                    print(f"{'':<20}  ↳ {e.name}: {e.comment}")
    total = len(result.item_results)
    faith = [e.value for r in result.item_results for e in r.evaluations if e.name == "citation_faithfulness"]
    print(f"\n{total - failures}/{total} cases passed" + ("" if use_judge else "  (LLM judge skipped)"))
    if faith:
        print(f"citation faithfulness: {sum(faith) / len(faith):.0%} of cited findings quote their source word for word")
    return failures


if __name__ == "__main__":
    main()
