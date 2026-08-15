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
from pathlib import Path

HERE = Path(__file__).parent
# Running this as a script puts `evaluation/` on the path, not the repo root.
sys.path.insert(0, str(HERE.parent))

from checks import mentions_code, scores  # noqa: E402,F401 -- re-exported for probes

GROUND_TRUTH = HERE / "ground_truth.json"
RESULTS = HERE / "results.json"

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
    passed, why = scores(
        answer,
        must=tuple(question.get("must_contain", [])),
        must_not=tuple(question.get("must_not_contain", [])),
    )
    return ("correct" if passed else "wrong"), why


async def run_once(prompt: str, student_id: str) -> tuple[str | None, int, float]:
    from app.runner import run_agent

    for attempt in range(RATE_LIMIT_ATTEMPTS):
        started = time.monotonic()
        result = await run_agent(prompt, student_id=student_id)
        error = str(result.error or "")
        if "RateLimit" in error or "429" in error:
            await asyncio.sleep(COOLDOWN_S)
            continue
        return result.answer, len(result.steps), time.monotonic() - started
    return None, 0, 0.0


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
            answer, steps, elapsed = await run_once(question["prompt"], student)
            verdict, why = score(answer, question)
            mark = {"correct": "PASS", "wrong": "FAIL", "no-answer": "NONE"}[verdict]
            print(f"  [{mark}] run {index + 1}: {steps} steps, {elapsed:.0f}s -- {why}")
            print(f"         {(answer or '').strip()[:190]}")
            results.append({
                "id": question["id"], "run": index, "verdict": verdict, "why": why,
                "answer": answer, "steps": steps, "elapsed_s": round(elapsed, 1),
            })
            await asyncio.sleep(SPACING_S)
        print()

    print("=" * 74)
    print("SUMMARY")
    print("=" * 74)
    total_correct = 0
    for question in questions:
        rows = [r for r in results if r["id"] == question["id"]]
        correct = sum(1 for r in rows if r["verdict"] == "correct")
        total_correct += correct
        steps = [r["steps"] for r in rows]
        print(f"  {question['id']:<34} {correct}/{len(rows)} correct   "
              f"steps {min(steps)}-{max(steps)} (mean {sum(steps) / len(steps):.1f})")
    print(f"\n  OVERALL: {total_correct}/{len(results)} correct")

    RESULTS.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"  written to {RESULTS}")

    from app.db.postgres import close_pool

    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
