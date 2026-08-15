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
BLOCKED_COURSE = "01040174"
"""Eligible only if a FAILED attempt counts. One group, two members, neither passed."""

FORECAST_COURSE = "00940412"
"""Ran every spring on record, and in winter and summer too -- so the rate is
1.00 per year and 0.43 per offering, and the two disagree about the answer."""


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

    passed_numbers = [
        row["courseNumber"]
        for row in await db.fetch(
            'select distinct c."courseNumber" from completed_courses cc '
            'join courses c on c."_id" = cc."courseId" where cc."userId" = $1 and cc."passed"',
            STUDENT,
        )
    ]
    failed_numbers = sorted(set(completed_numbers) - set(passed_numbers))

    print("\n" + "=" * 74)
    print(f"Q5 GROUND TRUTH -- eligibility for {BLOCKED_COURSE}, which turns on the PASSING rule")
    print("=" * 74)
    print(f"  failed courses on this transcript: {failed_numbers}")

    blocked_edges = await db.fetch(
        'select "requires", "group" from prerequisite_edges where "course" = $1 order by "group", "requires"',
        BLOCKED_COURSE,
    )
    blocked_groups: dict = {}
    for row in blocked_edges:
        blocked_groups.setdefault(row["group"], []).append(row["requires"])
    for group, requires in blocked_groups.items():
        via_passed = [r for r in requires if r in passed_numbers]
        via_failed = [r for r in requires if r in failed_numbers]
        print(f"    group {group}: any one of {requires}")
        print(f"      satisfied by a PASSED course: {via_passed or '-'}")
        print(f"      would be satisfied by a FAILED one: {via_failed or '-'}")
    strict = all(any(r in passed_numbers for r in req) for req in blocked_groups.values())
    loose = all(
        any(r in passed_numbers or r in failed_numbers for r in req)
        for req in blocked_groups.values()
    )
    print(f"\n  ELIGIBLE, counting only passes : {strict}   <- ground truth")
    print(f"  ELIGIBLE, counting every attempt: {loose}   <- the defect's answer")

    print("\n  every course whose eligibility FLIPS on the passing rule:")
    all_edges = await db.fetch('select "course", "requires", "group" from prerequisite_edges')
    by_course: dict = {}
    for row in all_edges:
        by_course.setdefault(row["course"], {}).setdefault(row["group"], []).append(row["requires"])
    flips = [
        course
        for course, groups in by_course.items()
        if all(any(r in passed_numbers or r in failed_numbers for r in req) for req in groups.values())
        and not all(any(r in passed_numbers for r in req) for req in groups.values())
    ]
    print(f"    {len(flips)}: {sorted(flips)}")

    print("\n" + "=" * 74)
    print(f"Q6 GROUND TRUTH -- will {FORECAST_COURSE} run next spring")
    print("=" * 74)
    offerings = await db.fetch(
        'select "academicYear", "semesterName", "status" from course_offerings '
        'where "courseNumber" = $1 order by "academicYear", "semesterName"',
        FORECAST_COURSE,
    )
    for row in offerings:
        print(f"    {row['academicYear']} {row['semesterName']:<8} {row['status']}")
    years = {row["academicYear"] for row in offerings}
    spring_years = {row["academicYear"] for row in offerings if row["semesterName"] == "spring"}
    spring_rows = [row for row in offerings if row["semesterName"] == "spring"]
    print(f"\n  observations: {len(offerings)} across {len(years)} academic years {sorted(years)}")
    print(f"  spring occurred in {len(spring_years)} of {len(years)} years {sorted(spring_years)}")
    print(f"  A) rate per CYCLE (correct)   : {len(spring_years)}/{len(years)} = "
          f"{len(spring_years) / len(years):.2f}   <- ground truth: it runs")
    print(f"  B) rate per OFFERING (the bug): {len(spring_rows)}/{len(offerings)} = "
          f"{len(spring_rows) / len(offerings):.2f}   <- forecasts 'will not run'")

    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
