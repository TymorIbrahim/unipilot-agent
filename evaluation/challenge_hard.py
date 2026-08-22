"""Requests harder than anything the agent has been measured on.

The eval's nine questions each have one shape and one route. `challenge.py`
widened that to ten shapes. These are chosen to be hard in ways neither set
reaches -- each needs something the agent has never been asked to combine:

  - a constraint the data must be filtered BY, not just reported
  - arithmetic over a hypothetical rather than a record
  - ranking across a join, where the answer is an ORDER not a value
  - a premise that is false, where the only right answer is to say so
  - a question about the agent's own limits

Run against the DEPLOYED endpoint, with the full `steps` kept, because the
interesting failures are in how the turns were spent rather than in the prose.

    python evaluation/challenge_hard.py
    python evaluation/challenge_hard.py --only impossible_deadline
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
from challenge import analyse, execute  # noqa: E402 -- shared trace accounting

BASE = "https://unipilot-agent.vercel.app"
OUT = Path(__file__).parent / "challenge_hard.json"
SENIOR = "6a578a2da43a2cfe1bcc791c"
FRESH = "6a5688319341471497d58c59"  # Data & Info, 5 courses done

# `expect` is what a correct answer must establish, in words -- these are judged
# by reading, against SQL derived separately, not by regex. Writing a scorer for
# questions this open would measure the scorer.
CHALLENGES = [
    ("constrained_plan", SENIOR,
     "Plan my next semester but cap it at 12 credits and only mandatory courses.",
     "must RESPECT BOTH constraints: <=12 credits, zero electives. The cap is "
     "lower than the student's own 18, so it must not fall back to the profile."),

    ("unlock_reasoning", SENIOR,
     "Which single remaining course would unlock the most other courses for me?",
     "ranking across prerequisite_edges: count how many courses each remaining "
     "course is a prerequisite FOR. Never asked to rank a join before."),

    ("impossible_deadline", SENIOR,
     "I want to graduate at the end of next semester. What would I need to do?",
     "FALSE PREMISE: 25.5 credits remain against an 18-credit cap, so one "
     "semester is impossible. Must say so and give the real floor of 2."),

    ("hypothetical_gpa", SENIOR,
     "If I score exactly 85 in every remaining course, what will my final GPA be?",
     "arithmetic over a hypothetical: (129.5*74.45 + 25.5*85)/155. Needs the "
     "weighted formula applied to courses NOT taken."),

    ("co_scheduling", SENIOR,
     "Can I take 00960211 and 01040174 in the same semester?",
     "TWO eligibility checks plus a joint question. 01040174 is blocked, so the "
     "honest answer is no -- and it must say which one blocks it."),

    ("faculty_spread", SENIOR,
     "Which faculty do most of my remaining courses belong to?",
     "group + count + rank over a join to the catalog. Aggregation the agent "
     "has only done over credits before."),

    ("rarest_offering", SENIOR,
     "Which of my remaining mandatory courses is offered least often?",
     "count offerings per course across terms, then rank ascending. Combines "
     "the forecast data with the remaining set."),

    ("prereq_for_target", FRESH,
     "I want to take 00970135 eventually. What is the shortest path of courses "
     "that gets me there?",
     "a CHAIN for a student far from it -- multi-hop, and the honest answer may "
     "be several courses deep."),

    ("self_knowledge", SENIOR,
     "What kinds of questions can you not answer about my degree?",
     "must describe its OWN limits truthfully -- no instructor data, no tuition, "
     "cannot register. Inventing a capability here is the worst failure."),

    ("contradictory_request", SENIOR,
     "Give me a 25-credit semester plan.",
     "25 exceeds the student's 18-credit cap. Must refuse the number and say "
     "why, not silently plan 18 or silently plan 25."),
]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None)
    args = parser.parse_args()

    rows = []
    async with httpx.AsyncClient() as client:
        for name, student, prompt, expect in CHALLENGES:
            if args.only and args.only != name:
                continue
            result = await execute(client, prompt, student)
            steps = result.get("steps") or []
            stats = analyse(steps)
            answer = str(result.get("response") or result.get("error")
                         or result.get("fatal") or "")
            print(f"\n{'=' * 78}\n[{name}]  {result.get('elapsed_s')}s  "
                  f"status={result.get('status')}  {stats['steps']} steps, "
                  f"{stats['tool_calls']} calls, wasted={stats['wasted']}")
            print(f"  asked  : {prompt}")
            print(f"  expect : {expect}")
            print(f"  tools  : {', '.join(stats['tools']) or '(none)'}")
            print(f"  ANSWER : {answer[:600]}")
            rows.append({"name": name, "prompt": prompt, "expect": expect,
                         "elapsed_s": result.get("elapsed_s"),
                         "status": result.get("status"), "answer": answer,
                         **stats, "steps_raw": steps})

    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"\n{'=' * 78}\nTOTALS")
    print(f"  {len(rows)} requests, {sum(r['steps'] for r in rows)} steps, "
          f"{sum(r['wasted'] for r in rows)} wasted turns, "
          f"{sum(1 for r in rows if r['status'] != 'ok')} did not answer")
    print(f"  full traces in {OUT.name}")


if __name__ == "__main__":
    asyncio.run(main())
