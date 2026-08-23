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

A verdict is a list of `Violation`s -- empty when the answer is sound, or when it
holds nothing these checks judge. It is NOT limited to min-grade plans: that was
the original scope and it left an ordinary term plan entirely unexamined, which
is how "Winter -- 23 credits" shipped against an 18-credit cap. Any answer
slotting a collection whose records carry `credits` is now load-checked, with
the grade checks skipping rows that claim no minimum.

A non-empty verdict is handed back to the loop as a loud, specific reason to try
again, in the same voice a rejected answer already is.
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
    check_claimed_pass_is_on_the_transcript,
    check_eligibility_is_not_self_contradictory,
    check_no_edge_identifiers,
    check_periods_are_whole,
    check_count_states_its_basis,
    check_answer_does_not_narrate_the_catalog_lookup,
    check_prereq_verdict_matches_the_edges,
    check_term_within_requested_cap,
    check_no_group_identifiers,
    check_no_join_side_labels,
    check_no_object_identifiers,
    check_gpa_in_range,
    check_grades_in_range,
    check_joint_floor,
    check_plan_within_requirement,
    check_term_load,
    check_term_within_cap,
)
from app.agent_core.facts.types import Basis, Collection, Scalar

_CREDITS_FIELD = "credits"
_GRADE_FIELD = "min_grade"
_CODE_FIELDS = ("number", "courseNumber", "code")
_TERM_FIELD = "term"
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
_CAP_FACTS = ("max_credits_per_semester", "max_credits", "credit_cap", "maxCreditsPerSemester")
"""Where the student's per-semester limit is found.

`max_credits_per_semester` is seeded from the profile at the start of every run,
so the first name always resolves; the rest cover a run that recomputed it under
its own name, the same way `_POINTS_FACTS` does."""

_REQUIRED_FACTS = (
    "credits_required", "required_credits", "total_required_credits",
    "degree_credits", "program_credits", "totalCredits", "total_credits_required",
    "degree_total", "degree_total_credits",
)
_COMPLETED_FACTS = (
    "completed_credits", "credits_completed", "earned_credits", "total_credits",
    "total_completed_credits", "passed_credits",
)
_REMAINING_FACTS = (
    "remaining_credits", "credits_remaining", "credits_to_go", "remaining_required_credits",
    "credits_needed", "credits_gap", "needed_credits",
)
"""The three halves of "how much is left", kept apart on purpose.

`_REMAINING_FACTS` is consulted LAST and only when the subtraction is
unavailable -- see `_remaining_required`. A fact called `remaining_credits` has
held the credits still to EARN in one run and the credits still on OFFER in the
next, and they differ here by a factor of two.

The spellings are not guesses. Every name here was read off a live trace: three
runs of the same question named the gap `credits_needed` in all three and
`credits_gap` in one, and the one that ALSO named its inputs `required_credits`
and `earned_credits` is the only one where this resolved -- so the check fired
on that run and skipped on the other two, which is exactly the difference
between the correct answer and the two thin ones. A lookup that misses is
indistinguishable from a check that does not exist."""

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
    violations += check_no_object_identifiers(answer.text)
    violations += check_no_join_side_labels(answer.text)
    violations += check_claimed_pass_is_on_the_transcript(answer.text, _passed_codes(facts))
    violations += check_alternatives_are_distinct(answer.text, question)
    violations += check_eligibility_is_not_self_contradictory(answer.text, question)
    violations += check_periods_are_whole(answer.text)
    violations += check_answer_does_not_narrate_the_catalog_lookup(answer.text)
    # Replays an eligibility verdict the answer ASSERTED. The grounding
    # invariant refuses a typed digit, so "you need 25.5 credits" cannot be
    # invented -- but "its prerequisites are not satisfied" carries no number
    # and costs nothing to make up, and a live run told a student the chain
    # stopped at a course they were eligible for that day.
    violations += check_prereq_verdict_matches_the_edges(
        answer.text, _satisfied_courses(facts)
    )

    # Also on every answer, not only plan-shaped ones: "it will take you 2
    # semesters" needs its credits whether or not the reply slots a plan
    # collection beside it.
    requirement = _remaining_required(facts)
    if requirement is not None:
        violations += check_count_states_its_basis(answer.text, requirement, question)

    collections = list(_plan_collections(answer, facts))
    courses = [course for _, term_courses in collections for course in term_courses]
    if not courses:
        return violations  # Not a min-grade plan -- nothing the rest judges.

    violations += list(check_grades_in_range(courses))

    # Two checks, and they are not the same one twice: `check_term_load` is a
    # 40-credit sanity ceiling for the `optimize` overflow, and the cap is this
    # student's real limit, which a 23-credit term cleared without either
    # noticing.
    #
    # PER TERM, not per collection. A `:detail` collection is not always one
    # term: a multi-semester answer slots a SUMMARY, one row per term carrying
    # that term's total. Summing those and comparing to a per-semester cap
    # compares the whole degree plan to one semester's limit -- it refused
    # "term_summary totals 38.5 credits, over this student's limit of 18",
    # where 38.5 was 16 + 13.5 + 9 across three terms, each of them legal. The
    # model spent its remaining rejections on that and the run returned nothing.
    cap = _credit_cap(facts)
    # A limit the student stated in the request, which outranks their profile
    # cap and which nothing else here can see -- a 16-credit term against a
    # requested 10 cleared both checks below, since the profile allows 18.
    requested = _requested_cap(question)
    for name, term_courses in collections:
        for label, group in _by_term(name, term_courses):
            group_credits = sum(course.credits for course in group)
            violations += check_term_load(group_credits, label)
            if cap is not None:
                violations += check_term_within_cap(group_credits, cap, label)
            if requested is not None:
                violations += check_term_within_requested_cap(
                    group_credits, requested, label
                )

    # The plan WHOLE, against what the degree still needs. Every check above is
    # per term and all of them pass a plan that schedules every unfinished course
    # in the track -- each term legal, the total 13 credits past the requirement,
    # and the answer two semesters too long.
    #
    # Measured on ONE collection, the largest. A multi-term answer usually slots
    # both a per-term summary and the course-by-course listing, and they describe
    # the same credits twice: adding them doubles the plan and would flag a
    # correct one.
    # SIMULATED collections only. `_plan_collections` gathers anything whose
    # records carry `credits`, which a `course_offerings` listing does too --
    # and against a 25.5-credit requirement that would flag an answer holding
    # the catalog as an over-long plan. A plan is a proposal about the future
    # and is marked simulated at the tool that built it; a record of what is on
    # offer is not. The per-term checks above run a 40-credit range check and a
    # cap check, both of which a listing survives, so only this one needed it.
    planned = [
        sum(course.credits for course in term_courses)
        for name, term_courses in collections
        if getattr(facts[name], "basis", None) is Basis.SIMULATED
    ]
    if cap is not None and requirement is not None and planned:
        violations += check_plan_within_requirement(max(planned), requirement, cap)

    standing = _standing(facts)
    if standing is not None:
        violations += check_gpa_in_range(standing.gpa)
        floor = _floor(question)
        if floor is not None:
            violations += check_joint_floor(standing, courses, floor)
    return violations


def _by_term(
    name: str, courses: "list[GradedCourse]"
) -> "list[tuple[str, list[GradedCourse]]]":
    """The collection split into the terms it actually describes.

    Rows carrying DISTINCT term labels are a summary across terms, and each is
    its own load. Rows sharing one label -- or carrying none -- are one term's
    courses, which is the shape the cap was written for and the shape that
    produced the 23-credit winter.
    """
    groups: dict[str, list[GradedCourse]] = {}
    for course in courses:
        groups.setdefault(course.term or name, []).append(course)
    if len(groups) <= 1:
        return [(name, courses)]
    return [(f"{name} ({label})", group) for label, group in groups.items()]


def _plan_collections(
    answer: Answer, facts: Mapping[str, HeldFact]
) -> "list[tuple[str, list[GradedCourse]]]":
    """(fact name, its planned courses) for each `:detail` collection the answer
    used -- one entry per rendered term.

    A course is a record carrying a numeric `credits`. `min_grade` is OPTIONAL,
    and requiring it was a real gap: every check below was gated behind a field
    only the min-grade planner produces, so an ORDINARY term plan -- courseNumber,
    title, category, credits -- returned nothing here and shipped completely
    unverified. A live answer to "how many semesters will it take me to graduate"
    put 23 credits in one winter term, against a cap of 18, and nothing looked.

    Rows without `min_grade` get `min_grade=None`. The joint-floor and
    grade-range checks skip them, because they genuinely have nothing to judge;
    the LOAD checks do not, because credits are all they ever needed."""
    collections: list[tuple[str, list[GradedCourse]]] = []
    for name in answer.used:
        held = facts.get(name)
        if held is None or not isinstance(held.value, Collection):
            continue
        courses: list[GradedCourse] = []
        for record in held.value.records:
            credits = _number(record.fields.get(_CREDITS_FIELD))
            if credits is None:
                continue
            courses.append(
                GradedCourse(
                    code=_code(record),
                    credits=credits,
                    min_grade=_number(record.fields.get(_GRADE_FIELD)),
                    term=_text(record.fields.get(_TERM_FIELD)),
                )
            )
        if courses:
            collections.append((name, courses))
    return collections


def _credit_cap(facts: Mapping[str, HeldFact]) -> float | None:
    value = _scalar_fact(facts, _CAP_FACTS)
    return value if value and value > 0 else None


def _remaining_required(facts: Mapping[str, HeldFact]) -> float | None:
    """Credits the student still has to EARN -- not courses they could still take.

    Derived from `degree total - completed` FIRST, and only then read from a
    fact the model named, because "remaining" is the one word in this domain
    that reliably means two different things. For this student it is either 25.5
    (the degree needs 155, they hold 129.5) or 50.0 (the track still lists 21
    unfinished courses) depending which sense you take, and a run has shipped
    each. Both totals are usually held at once, so a name lookup alone is a coin
    flip; the subtraction is not.

    Returning None SKIPS the check, which is the safe direction: an overshoot
    that goes unflagged is the behaviour we already have, while a requirement
    read too small would refuse correct plans -- the failure this file's history
    is mostly made of.
    """
    required = _scalar_fact(facts, _REQUIRED_FACTS)
    completed = _scalar_fact(facts, _COMPLETED_FACTS)
    if required is not None and completed is not None and required > completed:
        return required - completed
    value = _scalar_fact(facts, _REMAINING_FACTS)
    return value if value and value > 0 else None


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


_REQUESTED_CAP = re.compile(
    # English: a limit word, a number, then a credit noun. All three required.
    r"(?:under|below|at most|no more than|not more than|less than|max(?:imum)?(?:\s+of)?"
    r"|cap(?:ped)?\s+at|within)\s*(\d+(?:\.\d+)?)\s*(?:credits?|points?|nekudot)"
    # Hebrew: the same three parts, same order.
    r"|(?:מתחת\s*ל|עד|לא\s*יותר\s*מ|פחות\s*מ|מקסימום|לכל\s*היותר|בתוך)"
    r"\s*-?\s*(\d+(?:\.\d+)?)\s*(?:נקודות|נקודה|נק|קרדיטים|קרדיט)",
    re.IGNORECASE,
)
"""A per-term credit limit stated in the REQUEST, in either language.

All three parts are required -- a limit word, a number, and a credit noun -- and
in that order, because the questions this must NOT fire on are full of bare
numbers that mean something else. "I want to finish by summer 2027" and
"starting from 2025-2" carry years; "what is the maximum number of credits I am
allowed" carries the limit word and the credit noun but no figure, and reading
one out of it would invent a constraint the student never set.

Deliberately conservative in the same direction as everything else here: a cap
this misses leaves today's behaviour, while a cap it invents refuses a correct
plan."""


def _requested_cap(question: str) -> float | None:
    """The tightest per-term limit the question names, if it names one.

    Tightest rather than first: a question that says both "under 10 credits" and
    "no more than 12" has set 10, and honouring the looser of the two is the
    same failure as honouring neither.
    """
    found = [
        float(group)
        for match in _REQUESTED_CAP.finditer(question or "")
        for group in match.groups()
        if group is not None
    ]
    return min(found) if found else None


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


def _text(value: object) -> str | None:
    """A Scalar's value as a string, when it has one."""
    if isinstance(value, Scalar) and value.value not in (None, ""):
        return str(value.value)
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


_PASSED_SOURCES = ("passed_courses",)
_COURSE_CODE_FIELDS = ("courseNumber", "course", "number", "code")


_EDGE_SOURCES = ("prerequisite_edges",)


def _satisfied_courses(facts: Mapping[str, HeldFact]) -> dict[str, str]:
    """Courses whose prerequisites the HELD edges say are met, and what meets them.

    Replays the group algebra the model is asked to do and sometimes skips: edges
    sharing a `group` are ALTERNATIVES, so one passed member satisfies that group,
    and a course is satisfied when every group it has is satisfied.

    Read off the DERIVATION rather than the fact's name, exactly as
    `_passed_codes` is and for the same reason -- the name is prose the model
    wrote, and a collection of edges has been named after the wrong course.

    Returns only courses that ARE satisfied. A course whose edges are incomplete
    or absent is simply not in the map, so the check it feeds stays silent rather
    than guessing -- the safe direction, since the failure being caught is an
    over-confident NEGATIVE and a wrong positive here would be worse.
    """
    passed = set(_passed_codes(facts))
    groups: dict[str, dict[str, set[str]]] = {}
    for held in facts.values():
        derivation = str(getattr(held, "derivation", "") or "")
        if not any(source in derivation for source in _EDGE_SOURCES):
            continue
        for record in getattr(held.value, "records", None) or ():
            course = record.fields.get("course")
            requires = record.fields.get("requires")
            if course is None or requires is None:
                continue
            group = record.fields.get("group")
            label = str(group.value) if group is not None else str(course.value)
            groups.setdefault(str(course.value), {}).setdefault(label, set()).add(
                str(requires.value)
            )

    satisfied: dict[str, str] = {}
    for course, by_group in groups.items():
        witnesses = []
        for members in by_group.values():
            met = sorted(members & passed)
            if not met:
                witnesses = []
                break
            witnesses.append(met[0])
        if witnesses:
            satisfied[course] = ", ".join(witnesses)
    return satisfied


def _passed_codes(facts: Mapping[str, HeldFact]) -> list[str]:
    """Every course number the student has actually passed, per the facts held.

    Read off the DERIVATION, not the fact's name. The name is prose the model
    wrote and is exactly what went wrong: a collection of all 41 passed courses
    was called `passed_00970800`, and the answer then described it as being
    about 00970800. The derivation says "read from passed_courses" whatever the
    model called it.
    """
    codes: list[str] = []
    for held in facts.values():
        derivation = str(getattr(held, "derivation", "") or "")
        if not any(source in derivation for source in _PASSED_SOURCES):
            continue
        records = getattr(held.value, "records", None) or ()
        for record in records:
            for field in _COURSE_CODE_FIELDS:
                scalar = record.fields.get(field)
                if scalar is not None and scalar.value:
                    codes.append(str(scalar.value))
                    break
    return codes
