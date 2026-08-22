"""Policy areas nothing has touched, plus the two that would not finish.

The first policy set reached seven sections of the undergraduate regulations.
These go at the parts still unmeasured -- physical education, social-activity
credits, elective minimums, exemptions, reserve-duty accommodations -- and retry
the two that ran out of turns before the corpus route was made cheaper.

Two are deliberately compound: a rule PLUS the student's record, where quoting
either half alone is not an answer. One is a rule that does not exist, where
inventing a plausible one is the failure.

    python evaluation/challenge_policy2.py
    python evaluation/challenge_policy2.py --only pe_requirement
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from challenge import analyse, execute  # noqa: E402

BASE = "https://unipilot-agent.vercel.app"
OUT = Path(__file__).parent / "challenge_policy2.json"
SENIOR = "6a578a2da43a2cfe1bcc791c"

CHALLENGES = [
    # ---- untouched sections of the regulations --------------------------
    ("pe_requirement", SENIOR,
     "Do I have to take physical education, and how many credits is it?",
     "Section 4: 2 credits mandatory for all students, max 1.5 per semester, "
     "medical exemption is full with no replacement credits."),

    ("social_activity_credits", SENIOR,
     "Can I get credit for volunteering or reserve duty?",
     "Section 4: 26 hours in a semester -> 1 credit; 30 hours a year OR >=14 "
     "days reserve duty -> 2 credits; MAXIMUM 2 credits over the whole degree."),

    ("elective_minimum", SENIOR,
     "How many all-Technion elective credits do I need?",
     "Section 4: minimum 12 credits (10 in a 3-year programme), of which at "
     "least 6 must be enrichment courses. The student is on a 4-year track."),

    ("exemption_route", SENIOR,
     "Can I get an exemption from a course I studied elsewhere?",
     "5.9: yes -- prior higher-education studies or an exemption exam, with or "
     "without credits per faculty recommendation. No exemption for courses "
     "studied during a disciplinary expulsion."),

    ("reserve_duty_exam", SENIOR,
     "I have reserve duty during the exam period. What am I entitled to?",
     "5.10: an alternative exam if reserve duty falls on the exam day, or >=3 "
     "days in the week before, or >=10 consecutive / >=14 cumulative in the "
     "semester, or >=10 cumulative during the exam period. The alternative sits "
     "no later than 6 weeks into the following semester."),

    # ---- the two that ran out of turns last time ------------------------
    ("improve_passed_grade", SENIOR,
     "Can I retake a course I already passed to improve the grade?",
     "5.3b: yes, by re-registering in the TWO SEMESTERS IMMEDIATELY FOLLOWING "
     "the one the passing grade came from. 5.3f: the last grade determines."),

    ("english_by_deadline", SENIOR,
     "Is there a deadline for finishing my English requirement, and have I met it?",
     "5.6 condition 5: English by the end of the 4th semester. The student is "
     "in 2025-2, well past it, so the answer turns on their own record."),

    # ---- rule x record, where half an answer is no answer ---------------
    ("standing_full_check", SENIOR,
     "Give me every reason I might be in non-regular academic standing, and tell me "
     "which ones apply to me.",
     "5.6 lists eight conditions. GPA 74.45 clears the 65 floor. A correct "
     "answer enumerates the rule AND checks the ones it can against the record, "
     "saying plainly which it cannot check."),

    ("load_vs_cap", SENIOR,
     "The regulations let me take more credits than my plan allows. Why the difference?",
     "29 is the institutional maximum (5.1); 18 is this student's own "
     "`maxCreditsPerSemester`. Both are true and they answer different "
     "questions -- the failure is claiming one is the other."),

    # ---- a rule that does not exist -------------------------------------
    ("invented_rule", SENIOR,
     "What is the minimum attendance percentage required to sit an exam?",
     "The regulations set NO attendance percentage. The honest answer says the "
     "corpus does not cover it. Inventing '80%' is the worst outcome, and it is "
     "the shape a plausible-sounding rule takes."),
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
            print(f"\n{'=' * 78}\n[{name}]  {result.get('elapsed_s')}s  "
                  f"status={result.get('status')}  {stats['steps']} steps, "
                  f"wasted={stats['wasted']}  tools: {', '.join(stats['tools']) or '-'}")
            print(f"  asked : {prompt}")
            print(f"  truth : {truth}")
            print(f"  ANSWER: {answer[:650]}")
            rows.append({"name": name, "prompt": prompt, "truth": truth,
                         "elapsed_s": result.get("elapsed_s"),
                         "status": result.get("status"), "answer": answer,
                         **stats, "steps_raw": steps})

    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"\n{'=' * 78}\nTOTALS")
    print(f"  {len(rows)} requests, {sum(r['steps'] for r in rows)} steps, "
          f"{sum(r['wasted'] for r in rows)} wasted, "
          f"{sum(1 for r in rows if r['status'] != 'ok')} did not answer")


if __name__ == "__main__":
    asyncio.run(main())
