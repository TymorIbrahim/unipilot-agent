"""The same core questions, for every student in the dataset.

`ground_truth.json` holds nine questions and every one of them is about the
SENIOR -- one student of the four. So the agent has been measured on a single
transcript, a single track, and a single point in a degree, and every defect
found today came from a person poking the deployed GUI rather than from the
suite.

That is the gap this closes. The dataset holds four students across three
faculties and three different semesters, from 5 passed courses to 41, and the
paths they exercise genuinely differ: a student with almost nothing on their
transcript has empty `passed_courses` joins, and a student in summer ("2025-3")
resolves "next semester" differently from one in winter.

TRUTH IS DERIVED FROM SQL AT RUN TIME, not written down here. A hardcoded
expectation is a second copy of the data that goes stale the next time anyone
re-seeds, and this repo has already paid for measuring against a stale copy.
The derivation below was validated by reproducing the senior's figures --
129.5 / 155 / 25.5 / 74.45 / 41 -- which `ground_truth.json` states independently.

    python evaluation/cross_student.py
    python evaluation/cross_student.py --only gpa
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from challenge import analyse, execute  # noqa: E402

OUT = Path(__file__).parent / "cross_student.json"


async def truth_for_everyone() -> dict[str, dict]:
    """Every student's real figures, read straight out of Postgres."""
    from app.db.postgres import get_database

    database = await get_database()
    students = await database.fetch('select * from student_profiles order by "userId"')
    truth: dict[str, dict] = {}
    for profile in students:
        user = profile["userId"]
        totals = (
            await database.fetch(
                'select coalesce(sum("creditsCounted"), 0) counted, count(*) n '
                'from passed_courses where "userId" = $1',
                user,
            )
        )[0]
        program = await database.fetch(
            'select "totalCredits" t, name from degree_programs where "_id" = $1',
            profile["degreeId"],
        )
        gpa = (
            await database.fetch(
                'select round((sum(grade * "creditsCounted") / '
                'nullif(sum("creditsCounted"), 0))::numeric, 2) g '
                'from passed_courses where "userId" = $1',
                user,
            )
        )[0]
        required = float(program[0]["t"]) if program else None
        counted = float(totals["counted"])
        cap = float(profile["maxCreditsPerSemester"] or 0)
        remaining = round(required - counted, 2) if required else None
        truth[user] = {
            "program": profile["programSlug"],
            "semester": profile["currentSemesterCode"],
            "cap": cap,
            "passed_courses": totals["n"],
            "completed": counted,
            "required": required,
            "remaining": remaining,
            "gpa": float(gpa["g"]) if gpa["g"] is not None else None,
            "semesters_left": (
                math.ceil(remaining / cap) if remaining and cap else None
            ),
        }
    return truth


# Each probe: a question, and the figure the answer must contain. `field` names
# the key in the derived truth, so nothing here restates a number.
PROBES = [
    ("completed", "How many credits have I completed so far?", "completed"),
    ("required", "What is the total credit requirement for my degree?", "required"),
    ("remaining", "How many credits do I still need to graduate?", "remaining"),
    ("gpa", "What is my GPA?", "gpa"),
    ("semesters", "How many semesters until I graduate?", "semesters_left"),
]


def states(text: str, number: float) -> bool:
    """Whether the answer really states this number.

    Bounded exactly as the main scorer's `states_number` is, and for the same
    reason: 155 must not satisfy a check for 15.
    """
    import re

    needle = f"{number:g}"
    return re.search(rf"(?<![\d.]){re.escape(needle)}(?!\d)(?!\.\d)", text or "") is not None


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None, help="run one probe by name")
    args = parser.parse_args()

    truth = await truth_for_everyone()
    print(f"{len(truth)} students, truth derived from SQL\n")
    for user, t in truth.items():
        print(f"  {user}  {t['program']}  sem {t['semester']}  "
              f"{t['passed_courses']} passed  {t['completed']}/{t['required']} cr  "
              f"gpa {t['gpa']}  -> {t['semesters_left']} semesters left")

    rows = []
    async with httpx.AsyncClient() as client:
        for user, t in truth.items():
            print(f"\n{'=' * 78}\n{user}  ({t['program']})")
            for name, prompt, field in PROBES:
                if args.only and args.only != name:
                    continue
                expected = t.get(field)
                if expected is None:
                    continue
                result = await execute(client, prompt, user)
                answer = str(result.get("response") or result.get("error") or "")
                ok = states(answer, expected)
                stats = analyse(result.get("steps") or [])
                print(f"  [{'ok ' if ok else 'MISS'}] {name:11} want {expected:<8g} "
                      f"{result.get('elapsed_s')}s {stats['steps']}st  {answer[:110]}")
                rows.append({
                    "student": user, "program": t["program"], "probe": name,
                    "prompt": prompt, "expected": expected, "answer": answer,
                    "correct": ok, "elapsed_s": result.get("elapsed_s"),
                    "status": result.get("status"), **stats,
                })

    OUT.write_text(json.dumps({"truth": truth, "results": rows},
                              ensure_ascii=False, indent=2))
    right = sum(1 for r in rows if r["correct"])
    print(f"\n{'=' * 78}\nTOTALS  {right}/{len(rows)} correct across {len(truth)} students")
    for user in truth:
        mine = [r for r in rows if r["student"] == user]
        got = sum(1 for r in mine if r["correct"])
        flag = "" if got == len(mine) else "   <-- misses here"
        print(f"  {got}/{len(mine)}  {truth[user]['program']}{flag}")


if __name__ == "__main__":
    asyncio.run(main())
