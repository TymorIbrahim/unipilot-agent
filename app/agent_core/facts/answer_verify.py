"""The independent verify step: replay a plan answer's own numbers against the
post-conditions before it ships.

`resolve_answer` proves provenance -- every number came from a fact. This proves
SENSE: no impossible grade, no GPA out of range, and the plan's minimums hold the
floor when the courses are taken together, not just one at a time. It is the
check the grounding invariant structurally cannot be: a genuine, correctly-sourced
fact can still be an impossible grade or answer a subtly different question, and a
live winter run shipped exactly that -- six per-course minimums, two negative,
that jointly drop the GPA to 65 against an 80 floor.

Unlike the eval scorer (`tests/.../plan_eval_scoring.py`), which parses a saved
answer's prose because that is all it has, this reads the TYPED facts the answer
was built from: the plan comes from the Collection behind a `:detail` slot, whose
records carry `credits` and `min_grade` as real numbers, and the standing comes
from the scalar facts the min-grade formula itself consumes (`total_points`,
`total_credits`). Reading the facts, not the rendered text, keeps the check exact.

A verdict is a list of `Violation`s -- empty when the answer is sound OR when it
is simply not a min-grade plan (a normal answer has no such collection, so this
is a no-op for it). A non-empty verdict is handed back to the loop as a loud,
specific reason to try again, in the same voice a rejected answer already is.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from numbers import Number

from app.agent_core.facts.answer import Answer, HeldFact
from app.agent_core.facts.postconditions import (
    GradedCourse,
    Standing,
    Violation,
    check_alternatives_are_distinct,
    check_eligibility_is_not_self_contradictory,
    check_no_edge_identifiers,
    check_periods_are_whole,
    check_no_group_identifiers,
    check_gpa_in_range,
    check_grades_in_range,
    check_joint_floor,
    check_term_load,
)
from app.agent_core.facts.types import Collection, Scalar

_CREDITS_FIELD = "credits"
_GRADE_FIELD = "min_grade"
_CODE_FIELDS = ("number", "courseNumber", "code")
# The standing the joint-floor check replays against. The model names these
# facts freely -- the recipe suggests total_points/total_credits, but live runs
# have used points/credits and completed_* -- so several spellings are tried,
# most-specific first, and the result is CROSS-CHECKED against the gpa fact
# (_GPA_FACTS) below so a wrongly-named fact cannot pose as the standing. A live
# run named them `points`/`credits`, `_standing` found neither, and the whole
# joint check silently skipped -- shipping a plan that dropped the GPA to 65.
_POINTS_FACTS = ("total_points", "completed_points", "quality_points", "earned_points", "points")
_CREDITS_FACTS = ("total_credits", "completed_credits", "earned_credits", "credits")
_GPA_FACTS = ("gpa", "current_gpa")
_GPA_TOLERANCE = 0.5
_FLOOR = re.compile(r"above\s+(\d+(?:\.\d+)?)")


def verify_answer(
    answer: Answer, facts: Mapping[str, HeldFact], question: str
) -> list[Violation]:
    """Post-condition verdict for a resolved answer. Empty means sound (or not a
    plan). Runs each check for which the typed inputs are present; a missing input
    SKIPS its check rather than guessing -- an unverifiable answer is not blocked,
    only an actually-violated one is."""
    # Checked on EVERY answer, not just plans: a prerequisite question is not a
    # plan, and that is exactly where group labels were being shown as courses.
    violations = check_no_group_identifiers(answer.text)
    violations += check_no_edge_identifiers(answer.text)
    violations += check_alternatives_are_distinct(answer.text, question)
    violations += check_eligibility_is_not_self_contradictory(answer.text)
    violations += check_periods_are_whole(answer.text)

    collections = list(_plan_collections(answer, facts))
    courses = [course for _, term_courses in collections for course in term_courses]
    if not courses:
        return violations  # Not a min-grade plan -- nothing the rest judges.

    violations += list(check_grades_in_range(courses))

    # Each :detail collection is one rendered term; flag any whose load is really
    # the "(unscheduled)" overflow swept in.
    for name, term_courses in collections:
        violations += check_term_load(sum(course.credits for course in term_courses), name)

    standing = _standing(facts)
    if standing is not None:
        violations += check_gpa_in_range(standing.gpa)
        floor = _floor(question)
        if floor is not None:
            violations += check_joint_floor(standing, courses, floor)
    return violations


def _plan_collections(
    answer: Answer, facts: Mapping[str, HeldFact]
) -> "list[tuple[str, list[GradedCourse]]]":
    """(fact name, its planned courses) for each `:detail` collection the answer
    used -- one entry per rendered term.

    A course is a record carrying BOTH a numeric `credits` and `min_grade` -- the
    signature of a min-grade plan row, and what separates it from any other
    collection the answer might slot (offerings, a prereq list)."""
    collections: list[tuple[str, list[GradedCourse]]] = []
    for name in answer.used:
        held = facts.get(name)
        if held is None or not isinstance(held.value, Collection):
            continue
        courses: list[GradedCourse] = []
        for record in held.value.records:
            credits = _number(record.fields.get(_CREDITS_FIELD))
            grade = _number(record.fields.get(_GRADE_FIELD))
            if credits is None or grade is None:
                continue
            courses.append(GradedCourse(code=_code(record), credits=credits, min_grade=grade))
        if courses:
            collections.append((name, courses))
    return collections


def _standing(facts: Mapping[str, HeldFact]) -> Standing | None:
    points = _scalar_fact(facts, _POINTS_FACTS)
    credits = _scalar_fact(facts, _CREDITS_FACTS)
    gpa = _scalar_fact(facts, _GPA_FACTS)

    # Cross-fill from the identity gpa = points / credits, so holding the gpa and
    # EITHER total is enough -- a run that kept gpa and credits but not points is
    # still verifiable.
    if credits is None and points is not None and gpa:
        credits = points / gpa
    if points is None and credits is not None and gpa is not None:
        points = gpa * credits

    if points is None or credits is None or credits <= 0:
        return None

    # If a gpa fact is also held, it MUST agree with points/credits. When it does
    # not, a fact named `credits`/`points` is not the standing (it may be a
    # semester's credit total), and trusting it would replay the plan against a
    # phantom baseline -- worse than skipping, because it produces a confident
    # wrong verdict rather than an honest "not checked".
    if gpa is not None and abs(points / credits - gpa) > _GPA_TOLERANCE:
        return None
    return Standing(total_points=points, total_credits=credits)


def _floor(question: str) -> float | None:
    match = _FLOOR.search(question)
    return float(match.group(1)) if match else None


def _scalar_fact(facts: Mapping[str, HeldFact], names: tuple[str, ...]) -> float | None:
    for name in names:
        held = facts.get(name)
        if held is not None:
            value = _number(held.value)
            if value is not None:
                return value
    return None


def _number(value: object) -> float | None:
    """The float behind a numeric Scalar, or None for anything else -- a bool, a
    text scalar, a collection, or an absent field."""
    if isinstance(value, Scalar) and isinstance(value.value, Number) and not isinstance(value.value, bool):
        return float(value.value)
    return None


def _code(record: object) -> str:
    """A readable course code for the message, or "" if the row carries none."""
    fields = getattr(record, "fields", {})
    for name in _CODE_FIELDS:
        field = fields.get(name)
        if isinstance(field, Scalar) and field.value not in (None, ""):
            return str(field.value)
    return ""


__all__ = ["verify_answer"]
