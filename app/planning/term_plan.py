"""Pure term-plan orchestration -- the engine behind the agent's `plan_term` tool.

The agent supplies eligible CANDIDATE courses and one or more target terms; this
places them into a conflict-free weekly schedule, term by term, mirroring how a
student builds a plan by hand on the /plans page.

It REUSES the audited schedule/exam primitives (`pick_lessons_for_course`,
`slots_overlap`, `exams_from_offering`, `extract_lesson_options_from_offering`,
`build_weekly_schedule_payload`, `build_exam_summary`) but does NOT wrap
`select_conflict_aware_courses`, because that selector was proven (2026-07-21) to
seat a course even when its only lesson option hard-overlaps an occupied slot --
its "fewest overlaps" pick returns the least-bad combo, not a conflict-free one.
So the conflict-free guarantee is enforced HERE: a course whose chosen slots
overlap an already-placed course is DROPPED to `unscheduled`, never reserved.

Prerequisites and corequisites are FLAGGED, not hard-dropped (the app's
never-overclaim stance): a placed course carries `prereqStatus` / `coreqStatus`
so the caller can warn without the tool silently deciding eligibility. The one
hard exclusion is a no-additional-credit overlap with an already-credited course
(there is no benefit to taking it), which mirrors the planner's overlap rule.

Everything here is PURE -- catalog, offerings and completed courses are passed in
already loaded -- so it is unit-testable without Mongo. The DB wiring lives in the
service layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.planning.exam_summary import build_exam_summary
from app.planning.lesson_events import extract_lesson_options_from_offering
from app.planning.prerequisite_resolver import (
    canonical_course_number,
    extract_course_numbers_from_text,
    resolve_prerequisite_ids,
)
from app.planning.schedule_optimizer import (
    exams_conflict,
    exams_from_offering,
    pick_lessons_for_course,
    slots_overlap,
)
from app.planning.support import (
    course_number_keys,
    get_course_credits,
    normalize_course_id,
    prerequisites_met,
    round_credits,
)
from app.planning.weekly_schedule import build_weekly_schedule_payload, parse_time_range
from app.services.catalog_overlap_groups import overlap_group_for_course


@dataclass(frozen=True)
class Candidate:
    """A course the agent proposes for a term, with its priority category."""

    course: dict[str, Any]
    category: str = "elective"


@dataclass(frozen=True)
class TermInput:
    """One target term and the offerings available in it."""

    semester_code: str
    academic_year: int
    technion_semester_code: int
    offerings_by_number: dict[str, dict[str, Any]]


_CATEGORY_PRIORITY = {"mandatory": 0, "elective": 1}


def plan_terms(
    *,
    candidates: list[Candidate],
    terms: list[TermInput],
    completed_course_ids: set[str],
    completed_course_numbers: set[str],
    courses_by_number: dict[str, dict[str, Any]],
    overlap_groups: list[set[str]],
    max_credits_limit: float,
    prerequisite_groups: dict[str, list[list[str]]] | None = None,
) -> dict[str, Any]:
    """Place candidates into the given terms, conflict-free, mandatory first.

    `prerequisite_groups` maps a course number to ALTERNATIVE groups of course
    numbers: any one member of a group satisfies it, and every group must be
    satisfied. Optional so an existing caller keeps working, but without it
    `prereqStatus` degrades to the catalog fields -- which this deployment does
    not store, making every course vacuously satisfied.
    """
    cap = round_credits(max_credits_limit)
    satisfied_ids = set(completed_course_ids)  # completed OR already placed -> prereq + dedup
    satisfied_numbers = {_number_key(number) for number in completed_course_numbers}
    credited_raw = set(completed_course_numbers)  # raw numbers, for the overlap rule

    ordered = _ordered(candidates)
    placed_ids: set[str] = set()
    last_reason: dict[str, str] = {}
    term_results: list[dict[str, Any]] = []

    for term in terms:
        occupied_slots: list[dict[str, Any]] = []
        exam_entries: list[dict[str, Any]] = []
        term_credits = 0.0
        term_entries: list[dict[str, Any]] = []
        placed_courses: list[dict[str, Any]] = []

        for candidate in ordered:
            course = candidate.course
            course_id = normalize_course_id(course["_id"])
            number = _course_number(course)
            number_key = _number_key(number)

            if course_id in satisfied_ids or number_key in satisfied_numbers:
                continue  # already completed or placed in an earlier term

            clash = _overlapping_credited(number, overlap_groups, credited_raw)
            if clash:
                # Name the clashing course, and say which kind of clash it is.
                # `credited_raw` grows as courses are placed, so this fires both
                # for a course passed years ago and for one seated earlier in
                # this very plan -- and "overlaps an already-credited course"
                # reads as the first when it is the second, which is the
                # difference between "you cannot take this" and "not alongside
                # the one I just scheduled".
                passed_already = _number_key(clash) in {
                    _number_key(n) for n in completed_course_numbers
                }
                last_reason[number] = (
                    f"No additional credit alongside {clash}, which you "
                    + ("already passed" if passed_already else "are taking this term")
                )
                continue

            offering = _offering_for(term.offerings_by_number, number)
            options = extract_lesson_options_from_offering(offering, course_number=number)
            if not offering or not options:
                last_reason[number] = "Not offered with a published schedule in this term"
                continue

            credits = get_course_credits(course)
            if round_credits(term_credits + credits) > cap:
                last_reason[number] = "Exceeds the term credit cap"
                continue

            candidate_exams = exams_from_offering(
                offering, course_number=number, course_name=str(course.get("title") or "")
            )
            if exams_conflict(exam_entries, candidate_exams):
                last_reason[number] = "Exam date conflicts with an already-placed course"
                continue

            selected_lessons = pick_lessons_for_course(options, occupied_slots=occupied_slots)
            if selected_lessons is None:
                last_reason[number] = "No internally consistent lesson combination"
                continue

            chosen_options = _chosen_options(options, selected_lessons)
            chosen_slots = _slots_for(chosen_options, number)
            # The audit fix: `pick_lessons_for_course` minimizes overlaps but can
            # still return a combo that clashes an occupied slot. Enforce it here.
            if _clashes(chosen_slots, occupied_slots):
                last_reason[number] = "Schedule conflict with a higher-priority course"
                continue

            occupied_slots.extend(chosen_slots)
            exam_entries.extend(candidate_exams)
            term_credits = round_credits(term_credits + credits)
            satisfied_ids.add(course_id)
            satisfied_numbers.add(number_key)
            credited_raw.add(number)
            placed_ids.add(course_id)
            last_reason.pop(number, None)

            term_entries.append(
                {
                    "courseNumber": number,
                    "courseTitle": str(course.get("title") or ""),
                    "scheduleGroups": [option["rawGroup"] for option in chosen_options],
                }
            )
            placed_courses.append(
                {
                    "courseId": course_id,
                    "courseNumber": number,
                    "courseTitle": str(course.get("title") or ""),
                    "credits": credits,
                    "category": candidate.category,
                    "selectedLessonEvents": selected_lessons,
                    "prereqStatus": _prereq_status(
                        course,
                        satisfied_ids,
                        courses_by_number,
                        prerequisite_groups or {},
                        satisfied_numbers,
                    ),
                    "coreqStatus": _coreq_status(course, satisfied_numbers),
                }
            )

        term_results.append(
            {
                "semesterCode": term.semester_code,
                "placedCourses": placed_courses,
                "credits": round_credits(term_credits),
                "weeklySchedule": build_weekly_schedule_payload(term_entries),
                "examSummary": build_exam_summary(
                    [
                        {
                            "courseNumber": placed["courseNumber"],
                            "courseTitle": placed["courseTitle"],
                            "isActive": True,
                        }
                        for placed in placed_courses
                    ],
                    term.offerings_by_number,
                ),
            }
        )

    return {
        "terms": term_results,
        "unscheduled": _unscheduled(
            ordered, placed_ids, completed_course_ids, satisfied_numbers, last_reason
        ),
    }


def _ordered(candidates: list[Candidate]) -> list[Candidate]:
    """Mandatory before elective; original order preserved within a category."""
    return [
        candidate
        for _, candidate in sorted(
            enumerate(candidates),
            key=lambda pair: (_CATEGORY_PRIORITY.get(pair[1].category, 1), pair[0]),
        )
    ]


def _unscheduled(
    ordered: list[Candidate],
    placed_ids: set[str],
    completed_course_ids: set[str],
    satisfied_numbers: set[str],
    last_reason: dict[str, str],
) -> list[dict[str, Any]]:
    """One row per candidate that was never placed, INCLUDING the already-completed.

    Completed candidates used to be skipped entirely, so a model that handed
    over a whole curriculum got back a plan that simply did not mention them:
    six passed courses in one live call were neither placed nor reported, and
    nothing distinguished them from candidates that were lost. The catalog tells
    the model this tool returns the placed courses "plus the ones that did not
    fit and why", and a silent omission is the one answer a student cannot
    interrogate -- "why isn't X in my plan" had no answer at all.

    Reported by NUMBER as well as id, because that is how the placement loop
    skips them: a transcript row whose catalog id no longer resolves still
    matches by number, and 28% of this deployment's transcript rows reference a
    course that no longer exists.
    """
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in ordered:
        course_id = normalize_course_id(candidate.course["_id"])
        number = _course_number(candidate.course)
        if course_id in placed_ids:
            continue
        if number in seen:
            continue
        seen.add(number)
        if course_id in completed_course_ids or _number_key(number) in satisfied_numbers:
            reason = "Already completed -- it earns no further credit"
        else:
            reason = last_reason.get(number, "Did not fit the requested term(s)")
        rows.append({"courseNumber": number, "reason": reason})
    return rows


def _course_number(course: dict[str, Any]) -> str:
    return str(course.get("courseNumber") or course.get("number") or "")


def _number_key(number: str) -> str:
    return canonical_course_number(number) or number


def _offering_for(
    offerings: dict[str, dict[str, Any]], number: str
) -> dict[str, Any] | None:
    direct = offerings.get(number)
    if direct is not None:
        return direct
    canonical = canonical_course_number(number)
    return offerings.get(canonical) if canonical else None


def _overlapping_credited(
    number: str, overlap_groups: list[set[str]], credited_raw: set[str]
) -> Union[str, None]:
    """The credited course `number` grants no additional credit alongside, if any.

    Returns the course rather than a bool so the refusal can name it. A student
    told "no additional credit" without being told against WHAT cannot check the
    claim, and this rule is the one that removes a course from their plan
    outright.
    """
    group = overlap_group_for_course(number, overlap_groups)
    if not group:
        return None
    for credited in sorted(credited_raw):
        if set(course_number_keys(credited)) & group:
            return credited
    return None


def _chosen_options(
    options: list[dict[str, Any]], selected_lessons: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    selected_ids = {event["eventId"] for event in selected_lessons}
    return [option for option in options if option["eventId"] in selected_ids]


def _slots_for(options: list[dict[str, Any]], number: str) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    for option in options:
        parsed = parse_time_range(str(option.get("timeRange") or ""))
        day = str(option.get("day") or "")
        if not day or parsed is None:
            continue
        start, end = parsed
        slots.append(
            {"day": day, "startMinutes": start, "endMinutes": end, "courseNumber": number}
        )
    return slots


def _clashes(
    chosen_slots: list[dict[str, Any]], occupied_slots: list[dict[str, Any]]
) -> bool:
    return any(
        slots_overlap(chosen, occupied)
        for chosen in chosen_slots
        for occupied in occupied_slots
    )


def _prereq_status(
    course: dict[str, Any],
    satisfied_ids: set[str],
    courses_by_number: dict[str, dict[str, Any]],
    prerequisite_groups: dict[str, list[list[str]]],
    satisfied_numbers: set[str],
) -> str:
    """Conservative prereq flag. Only ever FLAGS; placement is never blocked.

    The EDGES are authoritative where they exist, because they are the only
    prerequisite data this deployment stores -- `courses` carries no
    `prerequisites` or `prerequisitesText` column, so the catalog path below
    resolves to an empty list and reports every course satisfied, which is how
    this flag came to be stamped "satisfied" on a plan for a student who had
    passed nothing.

    Group semantics: a group is satisfied by ANY one of its members, and every
    group must be satisfied. Reading the members as separate obligations would
    flag students who are genuinely eligible.
    """
    groups = prerequisite_groups.get(str(course.get("courseNumber") or ""))
    if groups:
        met = all(
            any(_number_key(number) in satisfied_numbers for number in group) for group in groups
        )
        return "satisfied" if met else "check_prerequisites"

    # No edges for this course: fall back to the catalog fields. Retained for
    # any deployment that does store them; here it means "nothing known".
    resolved = resolve_prerequisite_ids(course, courses_by_number=courses_by_number)
    probe = {**course, "prerequisites": resolved}
    return "satisfied" if prerequisites_met(probe, satisfied_ids) else "check_prerequisites"


def _coreq_status(course: dict[str, Any], satisfied_numbers: set[str]) -> str:
    """VESTIGIAL in this deployment: it returns "none" for every course, always.

    `courses` has no `corequisitesText` column, so the read below is always
    None, `needed` is always empty, and the answer is always "none" -- which the
    model is handed as though it meant "this course has no corequisites" rather
    than "no corequisite data exists here". The same silent degradation as the
    `prereqStatus` and no-additional-credit defects: a rule reading a column
    that is not there, failing open with a reassuring value.

    Left unwired DELIBERATELY, unlike those two. Measured against UniPilot's
    Mongo on 2026-08-15: 3 of 2,613 courses carry the field at all, and one of
    those three (00940333) names two corequisites that exist in no catalog. Two
    usable rows do not earn a column, a backfill and a migration.

    So this is a note, not a fix. If `corequisitesText` is ever seeded, this
    starts working with no change -- and until then the honest reading of a
    "none" here is "not checked".
    """
    needed = extract_course_numbers_from_text(course.get("corequisitesText"))
    if not needed:
        return "none"
    return "satisfied" if {_number_key(n) for n in needed} <= satisfied_numbers else "check_corequisites"


__all__ = ["Candidate", "TermInput", "plan_terms"]
