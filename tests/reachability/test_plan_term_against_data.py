"""What `plan_term` returns, checked against the data rather than against itself.

`plan_term` is the one tool whose answer cannot be scored as prose. There is no
single correct winter plan -- many are valid -- so `evaluation/ground_truth.json`
cannot hold an expected string, and a `must_not_contain` on "courses that must
not be placed" would fail the BEST answers, the ones that name a course in order
to explain why it was left out.

So the yardstick here is structural: whatever plan comes back must satisfy the
invariants that were derived from Supabase by SQL. Measured 2026-08-15 for the
demo student, whose track is `track-information-systems-engineering`:

  49 track courses, 41 passed, 21 remaining
  11 of the 21 are offered in winter on record
   5 of the 21 are SPRING-ONLY          -- 00960211/212/324/617, 00970800
   5 of the 21 DO NOT EXIST in `courses` -- 00960221/226/311/335, 00970280
   credit cap: student_profiles.maxCreditsPerSemester = 18.0

The last two lines are the traps. The five phantom courses are in the track
table and in nothing else: a planner that trusts the track list alone will seat
a course with no catalog row, no credits and no timetable, and the student will
find out at registration. The spring-only five are the ordinary version of the
same mistake.

These numbers are asserted as a fixture of the data, not recomputed from it: a
test that derives its expectation from the same query the code uses agrees with
itself no matter what either one does.
"""

from __future__ import annotations

import pytest

from app.db.postgres import get_database
from app.runner import DEFAULT_STUDENT_ID
from app.services.term_plan_service import build_term_plan

pytestmark = pytest.mark.supabase

TRACK = "track-information-systems-engineering"
CREDIT_CAP = 18.0

PHANTOM_COURSES = frozenset({"00960221", "00960226", "00960311", "00960335", "00970280"})
"""In `track_courses`, in no other table. Never placeable, in any term."""

SPRING_ONLY = frozenset({"00960211", "00960212", "00960324", "00960617", "00970800"})
"""Offered in spring in all three years on record, and in no other term."""


async def _remaining_track_courses() -> list[str]:
    """The student's track, minus what they have already passed."""
    db = await get_database()
    passed = {
        row["courseNumber"]
        for row in await db.fetch(
            'select distinct c."courseNumber" from completed_courses cc '
            'join courses c on c."_id" = cc."courseId" '
            'where cc."userId" = $1 and cc."passed"',
            DEFAULT_STUDENT_ID,
        )
    }
    track = [
        row["course"]
        for row in await db.fetch(
            'select "course" from track_courses where "track" = $1 order by "course"', TRACK
        )
    ]
    return [number for number in track if number not in passed]


def _placed(plan: dict) -> list[dict]:
    return [course for term in plan.get("terms", []) for course in term.get("placedCourses", [])]


def _numbers(courses: list[dict]) -> list[str]:
    return [str(c.get("courseNumber") or c.get("number") or "") for c in courses]


@pytest.fixture
async def winter_plan() -> dict:
    """One plan, built once, from every remaining course the track names.

    Deliberately handed the WHOLE remaining list, phantoms and spring-only
    courses included. Filtering them out here would test a planner that was
    never given the chance to fail -- and the agent does not filter them either,
    because the model builds candidates from the track.
    """
    remaining = await _remaining_track_courses()
    return await build_term_plan(
        user_id=DEFAULT_STUDENT_ID,
        semester_codes=["winter"],
        candidates=[{"courseNumber": number, "category": "elective"} for number in remaining],
        max_credits=CREDIT_CAP,
    )


class TestTheDataStillLooksLikeThis:
    """If these drift, the invariants below are asserting against a fiction."""

    async def test_the_student_still_has_courses_left(self) -> None:
        remaining = await _remaining_track_courses()
        assert len(remaining) == 21, f"expected 21 remaining track courses, found {len(remaining)}"

    async def test_the_phantom_courses_are_still_phantoms(self) -> None:
        db = await get_database()
        found = await db.fetch(
            'select "courseNumber" from courses where "courseNumber" = any($1::text[])',
            sorted(PHANTOM_COURSES),
        )
        assert not found, (
            f"{[r['courseNumber'] for r in found]} now exist in `courses`; "
            "they were catalog-less when this invariant was written"
        )

    async def test_the_spring_only_courses_are_still_spring_only(self) -> None:
        db = await get_database()
        rows = await db.fetch(
            'select distinct "courseNumber", "semesterName" from course_offerings '
            'where "courseNumber" = any($1::text[])',
            sorted(SPRING_ONLY),
        )
        offered = {(r["courseNumber"], r["semesterName"]) for r in rows}
        assert offered, "no offerings at all -- the offerings table may not be seeded"
        assert all(term == "spring" for _, term in offered), (
            f"one of these is no longer spring-only: {sorted(offered)}"
        )


class TestAPlacedCourseIsRealAndAvailable:
    async def test_no_phantom_course_is_placed(self, winter_plan: dict) -> None:
        """The expensive failure: a course that exists only in the track table
        has no catalog row, so it has no credits and no timetable, and a student
        told to take it finds out at registration."""
        placed = set(_numbers(_placed(winter_plan)))
        assert not (placed & PHANTOM_COURSES), (
            f"placed a course with no catalog row: {sorted(placed & PHANTOM_COURSES)}"
        )

    async def test_no_spring_only_course_is_placed_in_winter(self, winter_plan: dict) -> None:
        placed = set(_numbers(_placed(winter_plan)))
        assert not (placed & SPRING_ONLY), (
            f"placed a spring-only course in a winter plan: {sorted(placed & SPRING_ONLY)}"
        )

    async def test_every_placed_course_has_a_winter_offering(self, winter_plan: dict) -> None:
        """The general form of both traps above, asked of the whole plan."""
        placed = set(_numbers(_placed(winter_plan)))
        if not placed:
            pytest.skip("nothing was placed; the availability invariant has no subject")
        db = await get_database()
        with_winter = {
            row["courseNumber"]
            for row in await db.fetch(
                'select distinct "courseNumber" from course_offerings '
                'where "courseNumber" = any($1::text[]) and "semesterName" = $2',
                sorted(placed),
                "winter",
            )
        }
        assert placed <= with_winter, (
            f"placed in winter with no winter offering on record: {sorted(placed - with_winter)}"
        )

    async def test_no_already_passed_course_is_placed(self, winter_plan: dict) -> None:
        db = await get_database()
        passed = {
            row["courseNumber"]
            for row in await db.fetch(
                'select distinct c."courseNumber" from completed_courses cc '
                'join courses c on c."_id" = cc."courseId" '
                'where cc."userId" = $1 and cc."passed"',
                DEFAULT_STUDENT_ID,
            )
        }
        placed = set(_numbers(_placed(winter_plan)))
        assert not (placed & passed), f"placed a course already passed: {sorted(placed & passed)}"


FLIP_ON_A_FAILED_ATTEMPT = {
    # course     -> the failed course that would satisfy it if grades were ignored
    "01040174": "01040166",  # one group, ['01040066', '01040166'], neither passed
    "02380125": "01040166",  # two groups; the second one holds only the failed course
    "01040018": "01030015",  # one group, ['01030015'], failed
    "00960200": "01040166",  # three groups; the third is met only via the failed course
}
"""Courses whose eligibility depends entirely on the passing rule.

Measured against the real transcript: the student failed 01030015, 01040166 and
03240053, and for these four courses a prerequisite group is satisfied by the
failed attempt and by nothing else. Count the attempt and all four are eligible;
apply the 55 rule and none of them are.

This is the credits defect in different clothing -- the same transcript rows,
the same rule, a different consumer. There it inflated a total by 5.5 credits;
here it would tell a student to register for a course they cannot take.
"""


class TestAFailedAttemptSatisfiesNothing:
    async def test_the_flip_set_is_still_a_flip_set(self) -> None:
        """These courses only discriminate while the student's record is what it
        was measured to be. If the transcript changes, the assertion below stops
        testing the passing rule and starts passing for free."""
        db = await get_database()
        failed = {
            row["courseNumber"]
            for row in await db.fetch(
                'select distinct c."courseNumber" from completed_courses cc '
                'join courses c on c."_id" = cc."courseId" '
                'where cc."userId" = $1 and not cc."passed"',
                DEFAULT_STUDENT_ID,
            )
        }
        assert set(FLIP_ON_A_FAILED_ATTEMPT.values()) <= failed, (
            f"these are no longer failed courses on this transcript: "
            f"{sorted(set(FLIP_ON_A_FAILED_ATTEMPT.values()) - failed)}"
        )

    @pytest.mark.parametrize("number", sorted(FLIP_ON_A_FAILED_ATTEMPT))
    async def test_a_failed_prerequisite_is_flagged_not_satisfied(self, number: str) -> None:
        plans = [
            await build_term_plan(
                user_id=DEFAULT_STUDENT_ID,
                semester_codes=[term],
                candidates=[{"courseNumber": number, "category": "mandatory"}],
                max_credits=30.0,
            )
            for term in ("winter", "spring")
        ]
        placements = [c for plan in plans for c in _placed(plan) if _numbers([c])[0] == number]
        if not placements:
            pytest.skip(f"{number} was not placed in either term, so it carries no prereq flag")
        for course in placements:
            assert course.get("prereqStatus") == "check_prerequisites", (
                f"{number} was reported {course.get('prereqStatus')!r}: its only satisfying "
                f"prerequisite is {FLIP_ON_A_FAILED_ATTEMPT[number]}, which was FAILED"
            )


class TestThePlanObeysItsOwnLimits:
    async def test_the_credit_cap_holds(self, winter_plan: dict) -> None:
        for term in winter_plan.get("terms", []):
            credits = float(term.get("credits") or 0)
            assert credits <= CREDIT_CAP, (
                f"term {term.get('semesterCode')} carries {credits} credits, over the {CREDIT_CAP} cap"
            )

    async def test_the_reported_credits_match_the_placed_courses(self, winter_plan: dict) -> None:
        """A cap enforced against a total that is not the sum of what was placed
        is not enforced at all."""
        for term in winter_plan.get("terms", []):
            placed = term.get("placedCourses", [])
            summed = round(sum(float(c.get("credits") or 0) for c in placed), 2)
            reported = round(float(term.get("credits") or 0), 2)
            assert summed == reported, (
                f"term {term.get('semesterCode')} reports {reported} credits "
                f"but its {len(placed)} placed courses sum to {summed}"
            )

    async def test_a_course_is_placed_at_most_once(self, winter_plan: dict) -> None:
        numbers = _numbers(_placed(winter_plan))
        assert len(numbers) == len(set(numbers)), f"a course was placed twice: {sorted(numbers)}"

    async def test_nothing_is_silently_dropped(self, winter_plan: dict) -> None:
        """Every candidate must come back either placed or explained. A course
        that vanishes from both lists is invisible to the student and to the
        model writing the answer."""
        remaining = set(await _remaining_track_courses())
        placed = set(_numbers(_placed(winter_plan)))
        unscheduled = {
            str(row.get("courseNumber") or row.get("number") or "")
            for row in winter_plan.get("unscheduled", [])
        }
        missing = remaining - placed - unscheduled
        assert not missing, f"candidates accounted for nowhere in the result: {sorted(missing)}"
