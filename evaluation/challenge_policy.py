"""Harder planning, plus the policy side the eval barely touches.

Only `english_requirement` exercises retrieval at all, so the whole
search_corpus -> interpret -> cite path is measured by one question out of nine.
These add six policy questions with ground truth read off
`concepts/regulations-undergraduate.md`, and five planning questions that carry
a constraint the plan must be shaped BY rather than reported alongside.

Two are deliberately BOTH: "am I in good academic standing" needs the rule
(below 65 is non-regular) and the student's own GPA (74.45), and neither half is
an answer on its own. Those are the ones worth watching -- a grounded system can
fetch a record and it can quote a page; joining the two is where it gets hard.

    python evaluation/challenge_policy.py
    python evaluation/challenge_policy.py --only good_standing
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from challenge import analyse, execute  # noqa: E402

BASE = "https://unipilot-agent.vercel.app"
OUT = Path(__file__).parent / "challenge_policy.json"
SENIOR = "6a578a2da43a2cfe1bcc791c"

# `truth` is read from regulations-undergraduate.md or from SQL, so each answer
# can be judged rather than admired.
CHALLENGES = [
    # ---- policy: the regulations page ----------------------------------
    ("retake_failed_mandatory", SENIOR,
     "I failed a mandatory course. How many times can I retake it?",
     "5.3d: a mandatory course whose LAST grade is failing may be re-registered "
     "with NO time limit. Must not confuse it with 5.3b's two-semester window, "
     "which is for improving a course already PASSED."),

    ("max_load_policy", SENIOR,
     "What is the maximum number of credits I am allowed to take in one semester?",
     "5.1: 29 credits without special approval; above that needs the faculty "
     "head's recommendation and the Dean's approval. NOTE the trap -- this "
     "student's own profile cap is 18, which is a different thing."),

    ("appeal_window", SENIOR,
     "How long do I have to appeal an exam grade?",
     "5.4: within 4 days from when the exam copy is available. Extended by the "
     "number of days served if reserve duty fell in the appeal period."),

    ("improve_passed_grade", SENIOR,
     "Can I retake a course I already passed to improve the grade?",
     "5.3b: yes, by re-registering in the TWO SEMESTERS IMMEDIATELY FOLLOWING "
     "the one the passing grade was received in. 5.3f: the last grade counts."),

    ("drop_deadline", SENIOR,
     "Until when can I drop a course?",
     "5.7: changes in the first 2 weeks; no changes after the end of week 4 "
     "except exceptionally with Dean approval."),

    # ---- policy x record: needs both halves -----------------------------
    ("good_standing", SENIOR,
     "Am I in good academic standing?",
     "5.6 lists the conditions -- weighted average below 65 is the first. This "
     "student's GPA is 74.45, so on that criterion they are FINE. A correct "
     "answer joins the rule to the record; quoting either alone is half."),

    ("english_by_deadline", SENIOR,
     "Is there a deadline for finishing my English requirement, and have I met it?",
     "5.6 condition 5: English must be completed by the end of the 4th "
     "semester. The student is in 2025-2 and well past that, so the answer "
     "turns on whether their English courses are done."),

    # ---- planning shaped by a constraint --------------------------------
    ("part_time_load", SENIOR,
     "I'm working part-time next semester, so keep it under 10 credits. What should I take?",
     "a plan <= 10 credits. The student's own cap is 18, so this must come from "
     "the REQUEST, not the profile -- and it should still prefer mandatory."),

    ("heavier_first", SENIOR,
     "Plan my next two semesters, and put the heavier one first.",
     "two terms whose credit totals DESCEND. Ordering is a property of the plan "
     "as a whole, not of any one term."),

    ("deadline_feasibility", SENIOR,
     "I want to finish by summer 2027. Starting from 2025-2, is that realistic?",
     "25.5 credits at 18/semester is 2 semesters, and 2025-2 to summer 2027 is "
     "far more than 2. So YES, comfortably -- the interesting failure is "
     "answering no, or not committing."),

    ("gpa_maximising_plan", SENIOR,
     "Which courses next semester would raise my GPA the most?",
     "UNANSWERABLE as posed -- a future grade is not a record. The honest answer "
     "says so and offers what it CAN do (the credit weighting, or the plan). "
     "Inventing a ranking here would be the worst outcome."),
]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None)
    args = parser.parse_args()

    rows = []
    async with httpx.AsyncClient() as client:
        for name, student, prompt, truth in CHALLENGES:
            if args.only and args.only != name:
                continue
            result = await execute(client, prompt, student)
            steps = result.get("steps") or []
            stats = analyse(steps)
            answer = str(result.get("response") or result.get("error")
                         or result.get("fatal") or "")
            tools = ", ".join(stats["tools"]) or "(none)"
            print(f"\n{'=' * 78}\n[{name}]  {result.get('elapsed_s')}s  "
                  f"status={result.get('status')}  {stats['steps']} steps, "
                  f"wasted={stats['wasted']}  tools: {tools}")
            print(f"  asked : {prompt}")
            print(f"  truth : {truth}")
            print(f"  ANSWER: {answer[:700]}")
            rows.append({"name": name, "prompt": prompt, "truth": truth,
                         "elapsed_s": result.get("elapsed_s"),
                         "status": result.get("status"), "answer": answer,
                         **stats, "steps_raw": steps})

    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    corpus = sum(1 for r in rows if "search_corpus" in r["tools"])
    print(f"\n{'=' * 78}\nTOTALS")
    print(f"  {len(rows)} requests, {sum(r['steps'] for r in rows)} steps, "
          f"{sum(r['wasted'] for r in rows)} wasted, "
          f"{sum(1 for r in rows if r['status'] != 'ok')} did not answer")
    print(f"  {corpus} of {len(rows)} reached the corpus at all")


if __name__ == "__main__":
    asyncio.run(main())
