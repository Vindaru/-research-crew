"""End-to-end eval: every golden case -> the crew -> all graders -> a scorecard (and Langfuse, if running).

  python -m evals.run_evals --fake        # offline: fake crew, no API key, no cost (judge skipped)
  python -m evals.run_evals               # real crew + Claude judge; scores and traces go to Langfuse

Quality gate (M4): exits 1 when the run is below the thresholds, so CI can block a merge.
  --min-pass-rate 0.9        at least 90% of cases must pass every grader (default 1.0 = all of them)
  --min-faithfulness 1.0     every cited finding must quote its source word for word (default 1.0)
  --summary scorecard.md     also write the scorecard as Markdown (CI shows it on the run's page)
  --results results.json     also save every case's scores, for later comparison
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
    args.add_argument("--min-pass-rate", type=float, default=1.0)
    args.add_argument("--min-faithfulness", type=float, default=1.0)
    args.add_argument("--summary", help="write the scorecard as Markdown to this file")
    args.add_argument("--results", help="write per-case scores as JSON to this file")
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

    rows = score_rows(result, use_judge)
    print_scorecard(rows, use_judge)
    crashed = len(data) - len(rows)  # a case whose task raised has no result at all
    if crashed:
        print(f"{crashed} case(s) crashed before grading; see the errors above")
    gate = quality_gate(rows, total=len(data), min_pass_rate=opts.min_pass_rate,
                        min_faithfulness=opts.min_faithfulness)
    print(gate["message"])
    if result.experiment_url:
        print(f"Langfuse: {result.experiment_url}")
    if opts.summary:
        Path(opts.summary).write_text(scorecard_markdown(rows, use_judge, gate, "fake" if opts.fake else "live"))
    if opts.results:
        Path(opts.results).write_text(json.dumps({"gate": gate, "cases": rows}, indent=2))
    sys.exit(0 if gate["passed"] else 1)


CODE_CHECKS = [g.__name__ for g in DETERMINISTIC]
SHORT = {"status": "status", "citation_faithfulness": "faithful", "citations_valid": "cites_ok",
         "must_mention": "mentions", "must_not_say": "no_bad", "llm_judge": "judge"}


def passed(scores: dict, use_judge: bool) -> bool:
    return all(scores.get(n, 0) == 1.0 for n in CODE_CHECKS) and (not use_judge or scores.get("llm_judge", 0) >= JUDGE_PASS)


def score_rows(result, use_judge) -> list[dict]:
    """One plain dict per case: id, category, scores, pass/fail and why."""
    rows = []
    for r in result.item_results:
        meta = r.item["metadata"] if isinstance(r.item, dict) else r.item.metadata
        scores = {e.name: e.value for e in r.evaluations}
        reasons = [f"{e.name}: {e.comment}" for e in r.evaluations
                   if (e.name != "llm_judge" and e.value < 1.0) or (e.name == "llm_judge" and e.value < JUDGE_PASS)]
        rows.append({"id": meta["id"], "category": meta["category"], "scores": scores,
                     "passed": passed(scores, use_judge), "reasons": reasons})
    return sorted(rows, key=lambda row: row["id"])


def quality_gate(rows: list[dict], total: int, min_pass_rate: float, min_faithfulness: float) -> dict:
    """The CI gate. Crashed cases count as failures: a case we couldn't grade never counts as a pass."""
    pass_rate = sum(r["passed"] for r in rows) / total if total else 0.0
    faith = [r["scores"].get("citation_faithfulness", 0.0) for r in rows]
    faithfulness = sum(faith) / len(faith) if faith else 0.0
    ok = pass_rate >= min_pass_rate and faithfulness >= min_faithfulness
    message = (f"GATE {'PASSED' if ok else 'FAILED'}: pass rate {pass_rate:.0%} (min {min_pass_rate:.0%}), "
               f"citation faithfulness {faithfulness:.0%} (min {min_faithfulness:.0%})")
    return {"passed": ok, "pass_rate": pass_rate, "faithfulness": faithfulness,
            "min_pass_rate": min_pass_rate, "min_faithfulness": min_faithfulness, "message": message}


def print_scorecard(rows, use_judge):
    names = CODE_CHECKS + (["llm_judge"] if use_judge else [])
    print(f"\n{'case':<20}" + "".join(f"{SHORT[n]:>10}" for n in names) + f"{'result':>8}")
    for row in rows:
        print(f"{row['id']:<20}" + "".join(f"{row['scores'].get(n, 0):>10.2f}" for n in names)
              + f"{'PASS' if row['passed'] else 'FAIL':>8}")
        for reason in row["reasons"]:
            print(f"{'':<20}  ↳ {reason}")
    total = len(rows)
    print(f"\n{sum(r['passed'] for r in rows)}/{total} cases passed" + ("" if use_judge else "  (LLM judge skipped)"))


def scorecard_markdown(rows, use_judge, gate, mode) -> str:
    names = CODE_CHECKS + (["llm_judge"] if use_judge else [])
    icon = "✅" if gate["passed"] else "❌"
    lines = [f"## {icon} Research crew evals ({mode})", "", f"**{gate['message']}**", "",
             "| case | category | " + " | ".join(SHORT[n] for n in names) + " | result |",
             "|---|---|" + "---:|" * len(names) + "---|"]
    for row in rows:
        lines.append(f"| `{row['id']}` | {row['category']} | "
                     + " | ".join(f"{row['scores'].get(n, 0):.2f}" for n in names)
                     + f" | {'PASS' if row['passed'] else '**FAIL**'} |")
    failed = [r for r in rows if r["reasons"]]
    if failed:
        lines += ["", "### Why cases failed", ""]
        lines += [f"- `{r['id']}`: " + "; ".join(r["reasons"]) for r in failed]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
