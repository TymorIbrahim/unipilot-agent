"""Derive the correct answer to each evaluation question straight from the data.

Independent of the agent on purpose: this is the yardstick, so it is measured
with SQL rather than by asking the thing under test. Where the data admits more
than one defensible reading -- retakes, failed attempts -- both are printed, so
the choice of ground truth is made deliberately rather than inherited from
whichever one the agent happened to produce.
"""

from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, "/Users/tymoribrahim/Desktop/unipilot-agent")

from app.db.postgres import close_pool, get_database  # noqa: E402
from app.runner import DEFAULT_STUDENT_ID  # noqa: E402

STUDENT = DEFAULT_STUDENT_ID
TARGET_COURSE = "00960211"


async def main() -> None:
    db = await get_database()
    print(f"student: {STUDENT}\n")

    print("=" * 74)
    print("PROFILE")
    print("=" * 74)
    rows = await db.fetch(
        'select "userId", "programSlug", "degreeId", "catalogYear", "facultyId", '
        '"currentSemesterCode", "maxCreditsPerSemester" '
        'from student_profiles where "userId" = $1',
        STUDENT,
    )
    for row in rows:
        for key, value in row.items():
            print(f"  {key}: {value}")
    degree_id = rows[0]["degreeId"] if rows else None

    print("\n" + "=" * 74)
    print("Q2 GROUND TRUTH -- degree total")
    print("=" * 74)
    for row in await db.fetch(
        'select "_id", "name", "totalCredits" from degree_programs where "_id" = $1', degree_id
    ):
        print(f"  {row['name']}: totalCredits = {row['totalCredits']}")

    print("\n" + "=" * 74)
    print("Q1 GROUND TRUTH -- credits completed")
    print("=" * 74)
    total_rows = await db.fetchval(
        'select count(*) from completed_courses where "userId" = $1', STUDENT
    )
    distinct_courses = await db.fetchval(
        'select count(distinct "courseId") from completed_courses where "userId" = $1', STUDENT
    )
    print(f"  transcript rows          : {total_rows}")
    print(f"  distinct courseIds       : {distinct_courses}")

    naive = await db.fetchval(
        'select sum("creditsEarned") from completed_courses where "userId" = $1', STUDENT
    )
    print(f"  A) sum of every row      : {naive}   <- counts a retake twice")

    deduped = await db.fetchval(
        'select sum(c) from (select max("creditsEarned") as c from completed_courses '
        'where "userId" = $1 group by "courseId") t',
        STUDENT,
    )
    print(f"  B) one row per course    : {deduped}   <- retake counted once")

    passed = await db.fetchval(
        'select sum(c) from (select max("creditsEarned") as c from completed_courses '
        'where "userId" = $1 and "grade" >= 55 group by "courseId") t',
        STUDENT,
    )
    print(f"  C) per course, grade>=55 : {passed}   <- Technion pass mark")

    nonzero = await db.fetchval(
        'select sum(c) from (select max("creditsEarned") as c from completed_courses '
        'where "userId" = $1 and "creditsEarned" > 0 group by "courseId") t',
        STUDENT,
    )
    print(f"  D) per course, credits>0 : {nonzero}")

    print("\n  rows with a retake (attempt > 1 or duplicate courseId):")
    for row in await db.fetch(
        'select "courseId", count(*) n, array_agg("grade") grades, '
        'array_agg("creditsEarned") credits, array_agg("attempt") attempts '
        'from completed_courses where "userId" = $1 '
        'group by "courseId" having count(*) > 1',
        STUDENT,
    ):
        print(f"    {row['courseId']}: n={row['n']} grades={row['grades']} "
              f"credits={row['credits']} attempts={row['attempts']}")

    print("\n  failing/zero-credit rows:")
    for row in await db.fetch(
        'select cc."courseId", cc."grade", cc."creditsEarned", c."courseNumber", c."title" '
        'from completed_courses cc left join courses c on c."_id" = cc."courseId" '
        'where cc."userId" = $1 and (cc."grade" < 55 or cc."creditsEarned" = 0) '
        'order by cc."grade"',
        STUDENT,
    ):
        print(f"    {row['courseNumber']} grade={row['grade']} credits={row['creditsEarned']} "
              f"{(row['title'] or '')[:40]}")

    print("\n" + "=" * 74)
    print(f"Q4 GROUND TRUTH -- eligibility for {TARGET_COURSE}")
    print("=" * 74)
    for row in await db.fetch(
        'select "courseNumber", "title", "credits", "faculty" from courses where "courseNumber" = $1',
        TARGET_COURSE,
    ):
        print(f"  target: {row['courseNumber']} {row['title']} ({row['credits']} credits)")

    edges = await db.fetch(
        'select "course", "requires", "group" from prerequisite_edges where "course" = $1 '
        'order by "group", "requires"',
        TARGET_COURSE,
    )
    print(f"\n  prerequisite edges: {len(edges)}")
    groups: dict = {}
    for row in edges:
        groups.setdefault(row["group"], []).append(row["requires"])
    for group, requires in sorted(groups.items(), key=lambda kv: str(kv[0])):
        print(f"    group {group}: any one of {requires}")
    print(f"  distinct groups (each must be satisfied): {len(groups)}")

    completed_numbers = [
        row["courseNumber"]
        for row in await db.fetch(
            'select distinct c."courseNumber" from completed_courses cc '
            'join courses c on c."_id" = cc."courseId" where cc."userId" = $1',
            STUDENT,
        )
    ]
    print(f"\n  student has {len(completed_numbers)} distinct completed course numbers")
    for group, requires in sorted(groups.items(), key=lambda kv: str(kv[0])):
        met = [r for r in requires if r in completed_numbers]
        print(f"    group {group}: satisfied={bool(met)} via {met or '-'}")
    all_met = all(any(r in completed_numbers for r in req) for req in groups.values())
    print(f"\n  ELIGIBLE: {all_met}")

    already = TARGET_COURSE in completed_numbers
    print(f"  (already completed {TARGET_COURSE}? {already})")

    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
