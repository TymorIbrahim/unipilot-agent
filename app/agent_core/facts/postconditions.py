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
    """A planned course, and the minimum grade the answer claims holds the floor.

    `min_grade` is None on an ORDINARY plan row -- a term plan carries a course,
    its credits and its category, and no minimum at all. It used to be required,
    which meant every check here was reachable only from the min-grade planner
    and an ordinary plan was never verified: a live "how many semesters" answer
    put 23 credits in one term against an 18 cap, unexamined. The load checks
    need only `credits`; the grade checks skip a row with nothing to judge.
    """

    code: str
    credits: float
    min_grade: float | None = None
    term: str | None = None
    """Which term this row belongs to, when the collection spans several.

    A multi-semester answer slots a SUMMARY -- one row per term with that term's
    total -- and summing those against a per-semester cap compares the whole
    plan to one semester's limit."""


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
    r"(?:any one of|any of|one of|either|alternatives?|options?|choose from|choice of)"
    r"[:\s]+((?:\d{6,8}\s*(?:,|or|and)\s*)+\d{6,8})",
    re.IGNORECASE,
)
_CODE = re.compile(r"\b\d{6,8}\b")

_REPEATED_IN_A_LIST = re.compile(r"\b(\d{6,8})\b(?:\s*(?:,|or|and)\s*\1\b)+")
"""The same course code listed against itself, whatever words introduce it.

The lead-in vocabulary above was `any one of|one of|either`, and a live answer
said "with alternatives 00960211, 00960211" -- degenerate, and missed, because
`alternatives` was not in the list. That is the `prereqStatus` drift again: a
check keyed on prose and prose that got reworded.

This needs no lead-in. A list of one code repeated is never right in any
sentence: not as alternatives, not as prerequisites, not as courses to take."""


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

    # Phrasing-independent first, so a reworded lead-in cannot hide it.
    for repeated in _REPEATED_IN_A_LIST.findall(text or ""):
        violations.append(
            Violation(
                "degenerate_alternatives",
                f"the answer offers a choice between {repeated} and itself. Alternatives "
                "come from the edges' `requires` field -- projecting `course` collapses every "
                "option onto the course being asked about.",
            )
        )
    if violations:
        return violations

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


_FRACTIONAL_PERIOD = re.compile(
    r"(\d+\.\d+)\s+(semesters?|terms?|years?)\b", re.IGNORECASE
)


_MET_NONE = re.compile(
    r"\bmet?(?:s|ting)?\s+0\b|\b0\s+of\s+(\d+)\s+prerequisite\s+groups?\b|\bmeet 0 of\b",
    re.IGNORECASE,
)
_CLAIMS_ELIGIBLE = re.compile(
    r"\b(?:you\s+are|are)\s+eligible\b|\beligible:\s*yes\b|\byou\s+(?:can|may)\s+take\b",
    re.IGNORECASE,
)
_CLAIMS_INELIGIBLE = re.compile(
    r"\bnot\s+eligible\b|\bineligible\b|\beligible:\s*no\b", re.IGNORECASE
)


def check_eligibility_is_not_self_contradictory(
    text: str, question: str = ""
) -> list[Violation]:
    """An answer must not count zero met groups and then declare eligibility.

    Live answer, shipped:

        "No. You checked 1 prerequisite group and met 0, so you are eligible to
         take 01040174."

    Both halves are in one sentence and they are opposites. Worse than a plainly
    wrong answer: a reader who skims the front takes "No" and a reader who skims
    the end takes "eligible", and the run that produced it scored as CORRECT
    because the checker read only the leading word.

    Scoped by the QUESTION, not by the answer. The first version skipped any
    answer naming more than one course code, to spare "you are eligible for
    00960211 but you meet 0 of 1 groups for 01040174" -- two courses, two
    verdicts, no contradiction. That exemption disabled the check entirely: a
    good eligibility answer ALWAYS names the target course and the prerequisites
    that would satisfy it, so every real case has three or more codes in it. The
    guard went in, looked correct, and never fired once. It let this through:

        "You are eligible for 01040174, because you meet 0 of 1 prerequisite
         groups. To make it yes, pass any one of 01040066, 01040166."

    Counting what the QUESTION asks about instead gets both right. One course
    asked -> every verdict in the answer is about it, so "eligible" beside "met
    0" is a contradiction. Two or more asked -> which clause owns which verdict
    needs a parser this does not have, and blocking a correct answer is the
    worse error, so it stands aside.
    """
    body = text or ""
    if len(set(_CODE.findall(question or ""))) != 1:
        return []
    if not _MET_NONE.search(body):
        return []
    if _CLAIMS_INELIGIBLE.search(body):
        return []  # "you met 0 ... so you are NOT eligible" is the coherent form
    if not _CLAIMS_ELIGIBLE.search(body):
        return []
    return [
        Violation(
            "contradictory_eligibility",
            "the answer says 0 prerequisite groups are met AND that the student is eligible. "
            "Those are opposites. Meeting 0 of the required groups means NOT eligible -- say "
            "that, and name the prerequisite that is missing.",
        )
    ]


def check_periods_are_whole(text: str) -> list[Violation]:
    """You cannot take 0.42 of a semester.

    Asked how many semesters remained, the agent answered "you have 1.42
    semesters remaining at your current max load" -- 25.5 credits over an
    18-credit cap, reported as the raw quotient. The arithmetic is right and the
    sentence is not: a semester is indivisible, so the honest reading of 1.42 is
    "at least 2".

    Wrong in the optimistic direction, which is why it matters: a student reading
    1.42 hears "nearly done in one more term" when they need two. Both runs of
    that question answered this way, so it is the shape of the answer rather than
    a slip.

    Credits, grades and GPAs are untouched -- those are genuinely continuous.
    Only a count of PERIODS is flagged, because only periods cannot be part-taken.
    """
    found = _FRACTIONAL_PERIOD.findall(text or "")
    if not found:
        return []
    import math

    value, unit = found[0]
    whole = math.ceil(float(value))
    return [
        Violation(
            "fractional_period",
            f"the answer reports {value} {unit}, and a {unit.rstrip('s')} cannot be part-taken. "
            f"Round UP -- {whole} -- and say 'at least {whole}': the remainder is a term the "
            "student still has to attend, not a fraction they can skip.",
        )
    ]


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
        if course.min_grade is None:
            continue  # an ordinary plan row claims no minimum, so none can be wrong
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


def check_term_within_cap(
    term_credits: float, cap: float, term_label: str = "a term"
) -> list[Violation]:
    """A planned term must not exceed the student's OWN per-semester limit.

    Distinct from `check_term_load`, which is a sanity ceiling at 40 credits --
    "a number no real semester reaches", written to catch an 83-credit term made
    of `optimize` overflow. It is a range check by design, and 23 credits sails
    through it.

    This is the POLICY check, and it had no equivalent until now: nothing
    compared a plan to `student_profiles.maxCreditsPerSemester`. A live answer
    to "how many semesters will it take me to graduate" reported "Winter -- 23
    credits" against a cap of 18, and no guard looked, because the value was not
    a fact the answer layer could reach. It is one now -- the profile is seeded
    at the start of every run -- so the check is finally possible.

    The usual cause is not an over-full term but two terms collapsed into one:
    `plan_term` tags placed courses with the term NAME, so asking for
    ["winter", "spring", "winter"] returns two winters that a later
    `select term == "winter"` merges. The message says so, because a model told
    only "too many credits" drops a course instead of splitting the term.
    """
    if cap <= 0 or term_credits <= cap + _FLOOR_EPSILON:
        return []
    return [
        Violation(
            "term_over_cap",
            f"{term_label} totals {term_credits:g} credits, over this student's limit of "
            f"{cap:g} per semester. If you asked `plan_term` for the same term name twice, "
            "the two came back under one label and a `select` on it merged them -- give each "
            "term a distinct name, and pass `max_credits` so the planner enforces the cap "
            "rather than leaving it to be noticed here.",
        )
    ]


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
