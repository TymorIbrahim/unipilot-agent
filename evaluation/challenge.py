"""Novel requests, and an analysis of the `steps` each one cost.

The eval measures nine questions whose shapes the agent has been tuned against.
These are deliberately OUTSIDE that set -- different algebra (sort, group, rank,
negate), different entities (people, faculties), different framings (comparative,
counterfactual, compound, under-specified) -- because a capability that has never
been asked for is one nobody knows is missing.

Scored on two axes, and the second is the point. CORRECTNESS is checked by hand
against SQL. EFFICIENCY is read out of the trace: a turn that produced a defect,
a call repeated with identical arguments, an answer attempt rejected, a turn that
gained no fact. Those are the four ways a run wastes a model call, they are all
visible in `steps`, and efficiency is graded.

    python evaluation/challenge.py            # everything, writes challenge.json
    python evaluation/challenge.py --only rank_worst_grades
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter
from pathlib import Path

import httpx

BASE = "https://unipilot-agent.vercel.app"
HERE = Path(__file__).parent
OUT = HERE / "challenge.json"
SENIOR = "6a578a2da43a2cfe1bcc791c"

# `probe` names the capability under test, so a failure says what is missing
# rather than only that something is.
CHALLENGES = [
    ("rank_worst_grades", SENIOR, "Which three courses did I do worst in?",
     "sort + limit over the transcript -- ranking has never been asked for"),
    ("negation_mandatory", SENIOR,
     "Which mandatory courses in my track have I not taken yet?",
     "set difference stated as an absence; the answer is a LIST, not a number"),
    ("temporal_filter", SENIOR, "Which courses did I take in winter 2024?",
     "filtering the transcript by semester code -- a field never filtered on"),
    ("compound_two_part", SENIOR,
     "How many credits am I short, and which single course would get me closest?",
     "two questions in one reply; the second depends on the first"),
    ("comparative", SENIOR,
     "Is 00960211 or 01040174 the easier one for me to take next semester?",
     "comparing two courses on eligibility -- needs both checks then a judgement"),
    ("counterfactual", SENIOR,
     "If I fail 00970800, how does that change my graduation timeline?",
     "hypothetical over a course not yet taken"),
    ("aggregate_group", SENIOR,
     "How many credits have I completed at each grade level -- 90+, 80s, 70s, below?",
     "grouping a continuous field into buckets; `group` has never been asked this way"),
    ("underspecified", SENIOR, "Can I take it next semester?",
     "no antecedent at all -- must ask, not guess a course"),
    ("out_of_records", SENIOR, "Who teaches 00960211?",
     "instructor data is not in the schema; must say so, not invent"),
    ("boundary_zero", "6a5688319341471497d58c59",
     "What was my grade in 00940412?",
     "a course this student has NOT taken -- absence, not a zero"),
]


async def execute(client: httpx.AsyncClient, prompt: str, student: str) -> dict:
    started = time.monotonic()
    try:
        r = await client.post(f"{BASE}/api/execute",
                              json={"prompt": prompt, "student_id": student}, timeout=310.0)
        return {"elapsed_s": round(time.monotonic() - started, 1), "http": r.status_code, **r.json()}
    except Exception as exc:
        return {"elapsed_s": round(time.monotonic() - started, 1),
                "fatal": f"{type(exc).__name__}: {exc}"}


def analyse(steps: list) -> dict:
    """The four ways a run wastes a model call, counted off the trace.

    All of them are visible in `steps` alone: a defect and a rejection are
    reported back to the model in the NEXT turn's User_prompt, and a repeated
    call is two turns emitting identical JSON arguments.
    """
    calls: list[tuple[str, str]] = []
    answer_attempts = 0
    modules = Counter()
    for step in steps:
        modules[step.get("module", "?")] += 1
        if step.get("module") != "ReasoningLoop":
            continue
        raw = step.get("response")
        try:
            reply = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            continue
        if not isinstance(reply, dict):
            continue
        if "answer" in reply:
            answer_attempts += 1
        for call in reply.get("calls") or []:
            calls.append((str(call.get("tool")), json.dumps(call.get("args"), sort_keys=True)))

    # A note beginning "A step failed" is the loop reporting a real defect;
    # "Your answer was refused" is a rejected answer. Both are read from the
    # prompt of the turn that FOLLOWS the mistake.
    defects = sum(
        1 for s in steps
        if "A step failed" in ((s.get("prompt") or {}).get("User_prompt") or "")
    )
    rejections = sum(
        1 for s in steps
        if "answer was refused" in ((s.get("prompt") or {}).get("User_prompt") or "")
    )
    repeats = sum(n - 1 for n in Counter(calls).values() if n > 1)
    return {
        "steps": len(steps),
        "reasoning_turns": modules.get("ReasoningLoop", 0),
        "tool_calls": len(calls),
        "tools": sorted({t for t, _ in calls}),
        "modules": dict(modules),
        "turns_reporting_a_defect": defects,
        "turns_after_a_rejection": rejections,
        "repeated_identical_calls": repeats,
        "answer_attempts": answer_attempts,
        "wasted": defects + rejections + repeats,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None)
    args = parser.parse_args()

    rows = []
    async with httpx.AsyncClient() as client:
        for name, student, prompt, probe in CHALLENGES:
            if args.only and args.only != name:
                continue
            result = await execute(client, prompt, student)
            steps = result.get("steps") or []
            stats = analyse(steps)
            answer = str(result.get("response") or result.get("error") or result.get("fatal") or "")
            print(f"\n{'=' * 78}\n[{name}]  {result.get('elapsed_s')}s  "
                  f"status={result.get('status')}  {stats['steps']} steps, "
                  f"{stats['tool_calls']} calls, wasted={stats['wasted']}")
            print(f"  probing : {probe}")
            print(f"  asked   : {prompt}")
            print(f"  tools   : {', '.join(stats['tools']) or '(none)'}")
            if stats["wasted"]:
                print(f"  WASTE   : defects={stats['turns_reporting_a_defect']} "
                      f"rejections={stats['turns_after_a_rejection']} "
                      f"repeats={stats['repeated_identical_calls']}")
            print(f"  answer  : {answer[:400]}")
            rows.append({"name": name, "prompt": prompt, "probe": probe,
                         "elapsed_s": result.get("elapsed_s"), "status": result.get("status"),
                         "answer": answer, **stats, "steps_raw": steps})

    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"\n{'=' * 78}\nTOTALS")
    print(f"  {len(rows)} requests, {sum(r['steps'] for r in rows)} steps, "
          f"{sum(r['wasted'] for r in rows)} wasted turns")
    for r in sorted(rows, key=lambda r: -r["wasted"])[:5]:
        if r["wasted"]:
            print(f"    {r['wasted']:2} wasted  {r['name']}")
    print(f"  full traces in {OUT.name}")


if __name__ == "__main__":
    asyncio.run(main())
