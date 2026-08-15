"""`plan_term` promised to flag unmet prerequisites and could not.

The tool catalog tells the model this tool "FLAGS an unmet prerequisite rather
than guessing", and the answer prompt tells it to react to `prereqStatus`. Both
were false: `_prereq_status` resolved a course's `prerequisites` /
`prerequisitesText` fields, neither of which survived the move to Postgres, so
the resolved list was always empty -- and an empty prerequisite list is
vacuously met. Every placed course was stamped "satisfied", including for a
student who had passed nothing at all.

Measured on the live demo student: five courses placed, five stamped
"satisfied", one of which (00940412) the student is not eligible for.
"""

from __future__ import annotations

from app.planning.term_plan import _prereq_status

CATALOG: dict[str, dict[str, object]] = {}


def _status(number: str, groups, passed: set[str]) -> str:
    return _prereq_status(
        {"courseNumber": number},
        set(),
        CATALOG,
        {number: groups} if groups else {},
        {str(p) for p in passed},
    )


class TestTheFlagCanActuallyFire:
    def test_an_unmet_prerequisite_is_flagged(self) -> None:
        """The exact live case: 00940412 requires one of four calculus courses
        and the student passed none of them."""
        status = _status(
            "00940412",
            [["01040031", "01040017", "01040018", "01040195"]],
            passed={"01040042", "01040044", "01040016"},
        )
        assert status == "check_prerequisites"

    def test_a_student_who_passed_nothing_is_not_satisfied(self) -> None:
        """The regression in its purest form."""
        assert _status("00960211", [["00940224"]], passed=set()) == "check_prerequisites"


class TestGroupSemantics:
    def test_any_one_member_of_a_group_satisfies_it(self) -> None:
        """Edges sharing a group are ALTERNATIVES. Reading them as separate
        obligations would flag a student who is genuinely eligible -- the same
        mistake the reasoning prompt had to be taught not to make."""
        status = _status("00960211", [["00940224", "00940226"]], passed={"00940224"})
        assert status == "satisfied"

    def test_every_group_must_be_satisfied(self) -> None:
        status = _status("X", [["A"], ["B"]], passed={"A"})
        assert status == "check_prerequisites"

    def test_all_groups_satisfied_passes(self) -> None:
        status = _status("X", [["A", "C"], ["B"]], passed={"C", "B"})
        assert status == "satisfied"


class TestCoursesWithNoEdges:
    def test_a_course_with_no_prerequisites_is_satisfied(self) -> None:
        """Absence of edges means nothing is required, not that nothing is known
        -- a first-year course must not be flagged forever."""
        assert _status("01040016", [], passed=set()) == "satisfied"
