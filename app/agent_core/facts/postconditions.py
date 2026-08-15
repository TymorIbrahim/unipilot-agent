"""Deterministic post-conditions over a finished plan answer -- the oracle the
grounding invariant is missing.

`resolve_answer` proves every number in an answer is a real derived fact: it
checks PROVENANCE. It cannot check that the number is SANE (a grade of -82), nor
that it answers the question actually asked rather than a subtly different one (a
threshold that holds the floor for one course in isolation, when the courses are
taken together). Those are the two ways a genuine, correctly-sourced fact still
makes a wrong answer, and nothing downstream catches either today -- a live
winter run shipped six per-course minimums, two of them negative, that would in
fact drop the student's GPA to 65 against an 80 floor if earned together.

These checks close that gap for the one shape we have seen fail: per-course
minimum grades that hold a GPA floor. They are PURE ARITHMETIC on numbers already
derived -- no model call, no I/O -- so they are cheap enough to run on every
answer and exact enough to trust as a gate. They are written to serve two callers
without change: the loop, as an independent verify step that hands a failure back
as a loud reason to retry; and the eval harness, as a pass/fail oracle over a
saved run. A caught violation names WHAT is wrong and WHY, in the same voice the
loop already feeds back, because a reason a model cannot act on wastes the retry.
"""

from __future__ import annotations

import re

from collections.abc import Sequence
from dataclasses import dataclass

MIN_GRADE = 0.0
MAX_GRADE = 100.0

MAX_TERM_CREDITS = 40.0
"""A sanity ceiling on ONE term's planned credits -- roughly twice a full load.
A term far above this is not a heavy schedule; it is the `optimize` output's
"(unscheduled)" overflow swept into the plan because the placed rows were never
selected out. A live winter plan put 27 courses / 83 credits in one term this
way. Like the 0-100 grade bound, this is a range check, not a policy: it catches
a number no real semester reaches."""

_FLOOR_EPSILON = 1e-6
"""Float slack when replaying a threshold. A minimum computed to land EXACTLY on
the floor must not be flagged for landing there -- only for landing below it."""


@dataclass(frozen=True)
class Standing:
    """The student's record BEFORE the planned term(s).

    `total_points` is the sum of grade*credits over completed courses; `gpa` is
    its ratio to `total_credits`. The floor is measured against the standing
    AFTER the plan, never against this -- this is only the starting point.
    """

    total_points: float
    total_credits: float

    @property
    def gpa(self) -> float:
        return self.total_points / self.total_credits


@dataclass(frozen=True)
class GradedCourse:
    """A planned course and the minimum grade the answer claims holds the floor."""

    code: str
    credits: float
    min_grade: float


@dataclass(frozen=True)
class Violation:
    """One way a plan answer is unsound, phrased loudly enough to hand back to the
    loop as the reason to try again. `kind` is a stable tag for the harness to
    count by; `message` is student-of-the-model prose, not a code."""

    kind: str
    message: str


_GROUP_ID = re.compile(r"\b\d{6,8}\.\d+\b")
_EDGE_ID = re.compile(r"\b\d{6,8}\s*->\s*\d{6,8}\b")


def check_no_group_identifiers(text: str) -> list[Violation]:
    """A prerequisite GROUP id must never be shown as if it were a course.

    Groups are labelled `<course>.<n>` -- `00970800.0`, `00970800.1` -- which is
    bookkeeping, not something a student can register for. A live answer read
    "the alternatives I derived are 00970800.0, 00970800.1", naming two things
    that do not exist instead of the four course codes behind them.

    Nothing else could catch it. The tokens were slotted from a real fact, so the
    answer boundary passed them, and they LOOK like course codes to a reader --
    which is precisely what makes them worse than a visible error.

    The prompt already says to name alternatives by their `requires` codes and
    mostly does; this is for when it does not.
    """
    found = _GROUP_ID.findall(text)
    if not found:
        return []
    return [
        Violation(
            kind="group_identifier_shown",
            message=(
                f"the answer shows {', '.join(sorted(set(found))[:3])}, which are prerequisite "
                "GROUP labels, not courses. A student cannot register for those. Project the "
                "`requires` field of the edges and name the actual course codes, grouping them "
                "as the choices they are."
            ),
        )
    ]


def check_no_edge_identifiers(text: str) -> list[Violation]:
    """An EDGE id must never stand in for the course it points at.

    `prerequisite_edges` rows are identified as `<course>-><requires>`. A live
    answer read "any one of the course codes in 00960211->00940224,
    00960211->00940226" -- the right two prerequisites, named as internal edge
    keys a student cannot look up, let alone register for.

    The prompt has warned against rendering edge rows into a sentence since
    before today, and it still happens. It is worth catching in code because it
    reads as PLAUSIBLE: the real course code is sitting inside the token, so the
    sentence looks specific and technical rather than broken -- and a substring
    check for the code even passes.
    """
    found = _EDGE_ID.findall(text)
    if not found:
        return []
    return [
        Violation(
            kind="edge_identifier_shown",
            message=(
                f"the answer shows {', '.join(sorted(set(found))[:3])}, which are prerequisite "
                "EDGE ids, not courses. Name the course on the RIGHT of the arrow: `project` the "
                "edges' `requires` field and slot that, grouped as the choices they are."
            ),
        )
    ]


_ALTERNATIVES = re.compile(
    r"(?:any one of|one of|either)\s+((?:\d{6,8}\s*(?:,|or|and)\s*)+\d{6,8})", re.IGNORECASE
)
_CODE = re.compile(r"\b\d{6,8}\b")


def check_alternatives_are_distinct(text: str, question: str = "") -> list[Violation]:
    """A choice between prerequisites must be a choice between DIFFERENT courses.

    A live answer read "you meet 1 of 1 prerequisite groups. The course requires
    1 requirement: any one of 00960211, 00960211" -- the course being asked
    about, listed twice as its own prerequisite. The edges were projected on
    `course` instead of `requires`, so every alternative collapsed onto the
    target.

    Two things make it checkable without knowing the curriculum: a choice whose
    options are all the same course is not a choice, and no course is its own
    prerequisite. Both hold for every course in every catalog, so this needs no
    data -- which is what makes it safe to run on every answer.
    """
    asked = set(_CODE.findall(question or ""))
    violations: list[Violation] = []
    for listing in _ALTERNATIVES.findall(text or ""):
        options = _CODE.findall(listing)
        if len(options) > 1 and len(set(options)) == 1:
            violations.append(
                Violation(
                    "degenerate_alternatives",
                    f"the answer offers a choice between {options[0]} and itself. Alternatives "
                    "come from the edges' `requires` field -- projecting `course` collapses every "
                    "option onto the course being asked about.",
                )
            )
        elif asked & set(options):
            violations.append(
                Violation(
                    "self_prerequisite",
                    f"the answer lists {sorted(asked & set(options))[0]} among its own "
                    "prerequisites. No course requires itself; that is the `course` side of the "
                    "edge, not the `requires` side.",
                )
            )
    return violations


def check_gpa_in_range(gpa: float) -> list[Violation]:
    """A GPA is a ratio in (0, 100]. Above 100 is the classic slotting slip --
    total_points landed where the ratio belonged."""
    if gpa <= 0.0 or gpa > MAX_GRADE:
        return [
            Violation(
                "gpa_range",
                f"GPA is {gpa:.3f}, outside the possible 0-100 range. A GPA above 100 usually "
                "means a total (total_points) was slotted where the ratio (total_points/total_credits) "
                "belonged.",
            )
        ]
    return []


def check_grades_in_range(courses: Sequence[GradedCourse]) -> list[Violation]:
    """Every minimum grade must be a grade a student could actually earn: 0-100.

    A NEGATIVE minimum is not wrong arithmetic -- it is a real signal, reported
    as the wrong thing: it means even a 0 holds the floor, so the honest report
    is 0 ("any passing grade"), never the negative number itself. An ABOVE-100
    minimum means the floor cannot be held by that course alone.
    """
    out: list[Violation] = []
    for course in courses:
        if course.min_grade < MIN_GRADE:
            out.append(
                Violation(
                    "min_grade_range",
                    f"{course.code}: min_grade {course.min_grade:g} is below 0 -- no grade is "
                    "negative. It means even a 0 in this course would hold the floor, so report 0 "
                    "(or 'any passing grade'), not a negative number.",
                )
            )
        elif course.min_grade > MAX_GRADE:
            out.append(
                Violation(
                    "min_grade_range",
                    f"{course.code}: min_grade {course.min_grade:g} exceeds 100 -- no grade is above "
                    "100, so the floor cannot be held by this course alone.",
                )
            )
    return out


def check_term_load(term_credits: float, term_label: str = "a term") -> list[Violation]:
    """One planned term must not exceed a plausible semester load. Over the ceiling
    means the raw `optimize` output (placed rows PLUS its "(unscheduled)" overflow)
    was scored instead of just the placed rows -- so the whole remaining catalog
    landed in one term."""
    if term_credits > MAX_TERM_CREDITS:
        return [
            Violation(
                "term_load",
                f"{term_label} totals {term_credits:g} credits -- no single semester is that large. "
                "The plan is built from the raw optimize output including its '(unscheduled)' overflow; "
                "select slot == the term FIRST and derive credits, min_grade and the listing from those "
                "placed rows only.",
            )
        ]
    return []


def check_joint_floor(
    standing: Standing,
    courses: Sequence[GradedCourse],
    floor: float,
    *,
    epsilon: float = _FLOOR_EPSILON,
) -> list[Violation]:
    """Earning EXACTLY each reported minimum in ALL the planned courses at once
    must still hold the floor.

    This is the check a per-course threshold cannot pass by luck: each minimum is
    computed so its course ALONE holds the floor, but the student takes them
    TOGETHER. When the current GPA already clears the floor, the isolated minimums
    run low -- even negative -- and earning all of them at once can drop the GPA
    well below the floor the answer promised to hold. Replaying the whole plan is
    the only way to see that.
    """
    added_credits = sum(course.credits for course in courses)
    total_credits = standing.total_credits + added_credits
    if total_credits <= 0.0:
        return []  # No credits to average over -- no GPA exists to check.

    added_points = sum(course.min_grade * course.credits for course in courses)
    joint_gpa = (standing.total_points + added_points) / total_credits
    if joint_gpa < floor - epsilon:
        return [
            Violation(
                "joint_floor",
                f"Earning each course's stated minimum at once gives GPA {joint_gpa:.2f}, below the "
                f"{floor:g} floor the plan promised to hold. The minimums were computed per course in "
                "isolation -- each holds the floor alone, but the courses are taken together, so the "
                "thresholds must be solved jointly.",
            )
        ]
    return []


def check_plan(
    standing: Standing,
    courses: Sequence[GradedCourse],
    floor: float,
) -> list[Violation]:
    """All post-conditions for a min-grade plan answer, in one pass.

    Order is by how directly a violation points at its cause: a bad GPA poisons
    every threshold below it, an out-of-range grade is a local slip, and the
    joint floor is the whole plan judged together.
    """
    return [
        *check_gpa_in_range(standing.gpa),
        *check_grades_in_range(courses),
        *check_joint_floor(standing, courses, floor),
    ]


__all__ = [
    "MAX_GRADE",
    "MAX_TERM_CREDITS",
    "MIN_GRADE",
    "GradedCourse",
    "Standing",
    "Violation",
    "check_gpa_in_range",
    "check_grades_in_range",
    "check_joint_floor",
    "check_plan",
    "check_term_load",
]
