"""Run the evaluation questions against the agent and score them.

Scored against `ground_truth.json`, which was derived from the data by SQL and by
reading the corpus -- never by asking the agent. Comparing runs to each other
only ever proves CONSISTENCY, and the agent was consistently wrong about credits
across five identical runs before this file existed.

Serial by design. Each reasoning turn costs ~9.7k tokens, so three concurrent
runs breach a 200k tokens/minute account ceiling and the 429s look exactly like
the agent giving up.

    python evaluation/run_eval.py            # 3 repeats of every question
    python evaluation/run_eval.py --repeats 5 --only english_requirement
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).parent
# Running this as a script puts `evaluation/` on the path, not the repo root.
sys.path.insert(0, str(HERE.parent))

from checks import mentions_code, scores  # noqa: E402,F401 -- re-exported for probes

GROUND_TRUTH = HERE / "ground_truth.json"
RESULTS = HERE / "results.json"
TRACES = HERE / "traces"

RATE_LIMIT_ATTEMPTS = 4
COOLDOWN_S = 45.0
SPACING_S = 8.0


def score(answer: str | None, question: dict) -> tuple[str, str]:
    """(verdict, why) for one answer, judged by the shared checks.

    The matching lives in `checks.py` because it was wrong five times when it
    lived in whichever script needed it -- always in the pessimistic direction,
    marking correct answers as failures.
    """
    if not answer:
        return "no-answer", "the run produced no answer at all"
    expected = question.get("expected_periods")
    return scores(
        answer,
        must=tuple(question.get("must_contain", [])),
        must_not=tuple(question.get("must_not_contain", [])),
        stance=question.get("stance"),
        periods=(expected["min"], expected["max"]) if expected else None,
    )


@dataclass
class Run:
    """One live run, kept whole.

    `steps` and `error` used to be discarded at the door -- `len(result.steps)`
    was stored and the rest dropped. That made every failure unexaminable after
    the fact: `semesters_to_graduate` failed three times for three unknown
    reasons, and diagnosing it meant paying for a fourth run to see what the
    first three had already shown. The trace is the evidence; a scorer that
    keeps only the verdict throws the evidence away.

    `error` matters as much as `steps`. Every non-answer reaches the student as
    the same sentence, so the student-facing text cannot tell a spent turn
    budget from a stall from a refusal -- but `AgentResult.error` names the
    outcome, and those three call for completely different fixes.
    """

    answer: str | None
    steps: list
    error: str | None
    elapsed_s: float


async def run_once(prompt: str, student_id: str) -> Run:
    from app.runner import run_agent

    for _attempt in range(RATE_LIMIT_ATTEMPTS):
        started = time.monotonic()
        result = await run_agent(prompt, student_id=student_id)
        error = str(result.error or "")
        if "RateLimit" in error or "429" in error:
            await asyncio.sleep(COOLDOWN_S)
            continue
        return Run(result.answer, list(result.steps), result.error, time.monotonic() - started)
    return Run(None, [], "rate limited on every attempt", 0.0)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--only", default=None, help="run a single question by id")
    args = parser.parse_args()

    truth = json.loads(GROUND_TRUTH.read_text())
    student = truth["_student"]["userId"]
    questions = [q for q in truth["questions"] if args.only in (None, q["id"])]
    if not questions:
        raise SystemExit(f"no question with id {args.only!r}")

    results = []
    for question in questions:
        print("=" * 74)
        print(f"{question['id']}: {question['prompt']}")
        print("=" * 74)
        for index in range(args.repeats):
            run = await run_once(question["prompt"], student)
            verdict, why = score(run.answer, question)
            mark = {"correct": "PASS", "incomplete": "THIN", "wrong": "FAIL",
                    "no-answer": "NONE"}[verdict]
            note = f" -- {run.error}" if run.error else ""
            print(f"  [{mark}] run {index + 1}: {len(run.steps)} steps, "
                  f"{run.elapsed_s:.0f}s -- {why}{note}")
            print(f"         {(run.answer or '').strip()[:190]}")
            # The whole trace, on disk, named so a failing run can be opened
            # without paying to reproduce it.
            TRACES.mkdir(exist_ok=True)
            trace = TRACES / f"{question['id']}-{index}.json"
            trace.write_text(json.dumps({
                "id": question["id"], "run": index, "prompt": question["prompt"],
                "verdict": verdict, "why": why, "error": run.error,
                "answer": run.answer, "elapsed_s": round(run.elapsed_s, 1),
                "steps": run.steps,
            }, ensure_ascii=False, indent=2))
            results.append({
                "id": question["id"], "run": index, "verdict": verdict, "why": why,
                "answer": run.answer, "steps": len(run.steps),
                "error": run.error, "elapsed_s": round(run.elapsed_s, 1),
                "trace": str(trace.relative_to(HERE.parent)),
            })
            await asyncio.sleep(SPACING_S)
        print()

    print("=" * 74)
    print("SUMMARY")
    print("=" * 74)
    total_correct = total_thin = total_wrong = 0
    for question in questions:
        rows = [r for r in results if r["id"] == question["id"]]
        correct = sum(1 for r in rows if r["verdict"] == "correct")
        thin = sum(1 for r in rows if r["verdict"] == "incomplete")
        total_correct += correct
        total_thin += thin
        total_wrong += len(rows) - correct - thin
        steps = [r["steps"] for r in rows]
        note = f" (+{thin} thin)" if thin else ""
        print(f"  {question['id']:<34} {correct}/{len(rows)} correct{note}   "
              f"steps {min(steps)}-{max(steps)} (mean {sum(steps) / len(steps):.1f})")
    # Reported apart on purpose. A THIN answer is right and unhelpful; a WRONG
    # one is a claim a student would act on. Summing them hid the moment
    # `eligibility_01040174` stopped saying "yes" to an ineligible student.
    print(f"\n  OVERALL: {total_correct}/{len(results)} correct, "
          f"{total_thin} right-but-thin, {total_wrong} wrong")

    RESULTS.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"  written to {RESULTS}")

    from app.db.postgres import close_pool

    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
