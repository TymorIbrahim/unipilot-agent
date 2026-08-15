"""Build a conflict-free term plan -- the data half of `plan_term`.

The pure engine (`app/planning/term_plan.py`) was ported already and takes
everything as arguments. This is what feeds it, and it is the last piece of the
agent that still reached into UniPilot: without it, `plan_term` was advertised to
the model and failed on every call, because `internal_api_client` imported a
module that did not exist here.

**A much smaller context than UniPilot loaded.** Its `load_planning_context`
also fetched degree requirements, course pools, semester-matrix rules and a full
graduation-progress calculation -- three collections this deployment does not
have. None of it reached the planner: `build_term_plan` reads exactly three
things from that context (the profile, the completed records, the catalog
courses), so this loads those three and nothing else. Porting the whole loader
would have meant porting three tables to serve fields nobody reads.

**Targeted queries, not table scans.** The catalog is 2,613 courses and there are
6,580 offerings; a plan concerns a handful. Every query here is bounded by the
candidate numbers or the student id, because this runs inside a request with a
240s budget and a cold connection.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from app.planning.prerequisite_resolver import build_courses_by_number, canonical_course_number
from app.planning.semester_codes import pick_best_offering, resolve_plan_term
from app.planning.support import round_credits
from app.planning.term_plan import Candidate, TermInput, plan_terms
from app.services.catalog_overlap_groups import build_catalog_overlap_groups
from app.services.completed_course_attempts import latest_attempt_rank
from app.services.grade_evaluation import is_passing_grade, resolve_record_numeric_grade

DEFAULT_MAX_CREDITS = 18.0
"""The cap when the profile does not name one. The seed writes it explicitly on
every demo profile precisely so this fallback is not what an answer stands on."""

MAX_CREDITS_CEILING = 40.0
VALID_CATEGORIES = frozenset({"mandatory", "elective"})


async def build_term_plan(
    *,
    user_id: str,
    semester_codes: list[str],
    candidates: list[dict[str, Any]],
    max_credits: float | None = None,
    current_year: int | None = None,
    database: Any | None = None,
) -> dict[str, Any]:
    """A conflict-free plan for one or more terms, from agent-supplied candidates.

    A term may be NAMED ("winter") or CODED ("2025-1"); a bare name resolves
    against `current_year`. The label is echoed back on each placed course, so
    the caller splits the plan on exactly the string it passed.

    Every failure is a STATUS, never an exception: `dispatch._plan_term` turns
    the payload into a fact or a defect the model can read and act on, and an
    exception there would abort a request that could still answer without a plan.
    """
    if not semester_codes:
        return {"status": "validation_error", "errors": ["At least one semesterCode is required"]}
    if not candidates:
        return {"status": "validation_error", "errors": ["At least one candidate course is required"]}

    if database is None:
        from app.db.postgres import get_database

        database = await get_database()

    preferred_year = current_year or date.today().year
    term_keys: list[tuple[str, int, int]] = []
    for code in semester_codes:
        resolved = resolve_plan_term(code, preferred_year=preferred_year)
        if resolved is None:
            return {"status": "validation_error", "errors": [f"Invalid semesterCode: {code}"]}
        term_keys.append(resolved)

    profile = await _load_profile(database, user_id)
    if profile is None:
        return {"status": "profile_not_found"}

    completed_records = await _load_completed(database, user_id)
    wanted_numbers = sorted(
        {str(item.get("courseNumber") or "").strip() for item in candidates if item.get("courseNumber")}
    )
    catalog_courses = await _load_catalog(database, wanted_numbers, completed_records)

    courses_by_number = build_courses_by_number(catalog_courses)
    courses_by_id = {
        str(course["_id"]): course for course in catalog_courses if course.get("_id") is not None
    }

    max_credits_limit = _resolve_cap(profile, max_credits)
    completed_course_ids = set(_effective_completions(completed_records))
    completed_course_numbers = _completed_numbers(completed_records, courses_by_id)
    overlap_groups = build_catalog_overlap_groups(catalog_courses)

    resolved_candidates, unknown = _resolve_candidates(candidates, courses_by_number)
    candidate_numbers = sorted(
        {
            str(candidate.course.get("courseNumber") or candidate.course.get("number") or "")
            for candidate in resolved_candidates
        }
    )

    terms: list[TermInput] = []
    for label, academic_year, technion_code in term_keys:
        terms.append(
            TermInput(
                semester_code=label,
                academic_year=academic_year,
                technion_semester_code=technion_code,
                offerings_by_number=await _load_term_offerings(
                    database, candidate_numbers, academic_year, technion_code
                ),
            )
        )

    result = plan_terms(
        candidates=resolved_candidates,
        terms=terms,
        completed_course_ids=completed_course_ids,
        completed_course_numbers=completed_course_numbers,
        courses_by_number=courses_by_number,
        overlap_groups=overlap_groups,
        max_credits_limit=max_credits_limit,
    )

    result["unscheduled"] = [
        *result["unscheduled"],
        *[{"courseNumber": number, "reason": "Not found in the catalog"} for number in unknown],
    ]
    result["status"] = "ok"
    result["maxCredits"] = max_credits_limit
    return result


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


async def _load_profile(database: Any, user_id: str) -> dict[str, Any] | None:
    rows = await database.fetch(
        'select "userId", "programSlug", "catalogYear", "currentSemesterCode", '
        '"maxCreditsPerSemester" from student_profiles where "userId" = $1',
        user_id,
    )
    return dict(rows[0]) if rows else None


async def _load_completed(database: Any, user_id: str) -> list[dict[str, Any]]:
    """The student's transcript, with each row's course NUMBER joined in.

    A LEFT join, deliberately. 28% of transcript rows reference a catalog course
    that no longer exists, and an inner join would silently drop them -- shrinking
    the completed set, so the planner would offer courses the student has already
    passed. The number comes back null for those and the overlap rules simply do
    not fire, which is a weaker plan rather than a wrong one.
    """
    rows = await database.fetch(
        'select cc."courseId", cc."semesterCode", cc."grade", cc."gradePoints", '
        'cc."creditsEarned", cc."attempt", c."courseNumber" '
        "from completed_courses cc "
        'left join courses c on c."_id" = cc."courseId" '
        'where cc."userId" = $1',
        user_id,
    )
    return [dict(row) for row in rows]


async def _load_catalog(
    database: Any, candidate_numbers: list[str], completed_records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """The catalog rows this plan needs: the candidates, plus what was completed.

    Both halves are required and for different reasons. The candidates resolve
    into `Candidate` objects and carry the credits the cap is counted against;
    the completed courses are what the overlap and duplicate rules compare
    against, and a plan that cannot see them will happily re-plan a passed course.
    """
    completed_ids = sorted({str(r["courseId"]) for r in completed_records if r.get("courseId")})
    rows = await database.fetch(
        'select "_id", "courseNumber", "title", "titleHebrew", "credits", "faculty", '
        '"studyFramework", "catalogYear", "status" '
        'from courses where "courseNumber" = any($1) or "_id" = any($2)',
        candidate_numbers,
        completed_ids,
    )
    return [dict(row) for row in rows]


async def _load_term_offerings(
    database: Any, numbers: list[str], academic_year: int, technion_semester_code: int
) -> dict[str, dict[str, Any]]:
    """The best offering per course for one term.

    Mirrors UniPilot's two-pass lookup: prefer an offering in the REQUESTED year,
    and fall back to the same term in any other year. The fallback is what makes
    planning possible at all here -- the catalog holds 2023-2025 offerings, so a
    plan for a future year would otherwise find nothing and place no course.
    A fallback offering's lesson times are a reasonable estimate of when the
    course runs; its year is not claimed to be the requested one.
    """
    if not numbers:
        return {}

    rows = await database.fetch(
        'select "_id", "courseNumber", "semesterName", "semesterCode", "academicYear", '
        '"catalogVersion", "status", "scheduleGroups", "examDates" '
        'from course_offerings where "courseNumber" = any($1) and "semesterCode" = $2',
        numbers,
        technion_semester_code,
    )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        offering = dict(row)
        grouped.setdefault(str(offering["courseNumber"]), []).append(offering)

    best: dict[str, dict[str, Any]] = {}
    for number in numbers:
        options = grouped.get(number, [])
        if not options:
            continue
        exact = [o for o in options if o.get("academicYear") == academic_year]
        chosen = pick_best_offering(
            exact or options,
            preferred_academic_year=academic_year,
            semester_code=technion_semester_code,
        )
        if chosen:
            best[number] = chosen
    return best


# ---------------------------------------------------------------------------
# Shaping
# ---------------------------------------------------------------------------


def _resolve_cap(profile: dict[str, Any], max_credits: float | None) -> float:
    """The credit ceiling for one term.

    An explicit `max_credits` from the tool call wins -- it is the student's own
    constraint, arriving through the question. The profile's value is next, and
    the ceiling is applied last so no input can ask for an unschedulable term.
    """
    if max_credits is not None:
        cap = float(max_credits)
    else:
        cap = float(profile.get("maxCreditsPerSemester") or DEFAULT_MAX_CREDITS)
    return round_credits(min(cap, MAX_CREDITS_CEILING))


def _resolve_candidates(
    candidates: list[dict[str, Any]], courses_by_number: dict[str, dict[str, Any]]
) -> tuple[list[Candidate], list[str]]:
    resolved: list[Candidate] = []
    unknown: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        number = str(item.get("courseNumber") or "").strip()
        if not number or number in seen:
            continue
        seen.add(number)
        category = str(item.get("category") or "elective")
        if category not in VALID_CATEGORIES:
            category = "elective"
        course = courses_by_number.get(number) or courses_by_number.get(
            canonical_course_number(number) or ""
        )
        if not course:
            # Reported back rather than dropped: "not found in the catalog" is an
            # answer the model can act on, where silence looks like a course that
            # was considered and rejected on merit.
            unknown.append(number)
            continue
        resolved.append(Candidate(course=course, category=category))
    return resolved, unknown


def _effective_completions(completed_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """One row per course: the LATEST attempt, and only if it passed.

    Two rules that only matter together. Latest-attempt-only stops a failed first
    sitting from masking a later pass; passing-only stops a failed retake from
    counting as completion. Applying either alone gets a retaken course wrong in
    one direction or the other.

    Ranked without `recordedAt`, which this deployment does not store: the
    (semester, attempt) pair orders the rows on every real transcript, and both
    fields are present on 100% of them.
    """
    latest: dict[str, dict[str, Any]] = {}
    ranks: dict[str, tuple[int, int, int, float]] = {}

    for record in completed_records:
        course_id = str(record.get("courseId") or "")
        if not course_id:
            continue
        rank = latest_attempt_rank(
            attempt=int(record.get("attempt") or 1),
            recorded_at_timestamp=0.0,
            semester_code=str(record.get("semesterCode") or ""),
        )
        if course_id in ranks and rank <= ranks[course_id]:
            continue
        ranks[course_id] = rank
        latest[course_id] = record

    effective: dict[str, dict[str, Any]] = {}
    for course_id, record in latest.items():
        if not is_passing_grade(record):
            continue
        numeric = resolve_record_numeric_grade(record)
        effective[course_id] = {
            "courseId": course_id,
            "creditsEarned": round_credits(float(record.get("creditsEarned") or 0.0)),
            "grade": numeric if numeric is not None else record.get("grade"),
            "semesterCode": record.get("semesterCode"),
            "attempt": int(record.get("attempt") or 1),
        }
    return effective


def _completed_numbers(
    completed_records: list[dict[str, Any]], courses_by_id: dict[str, dict[str, Any]]
) -> set[str]:
    """Course NUMBERS of completed courses, for the overlap and coreq rules.

    Best effort by design: the transcript keys on `courseId`, and an id that
    resolves to no catalog row yields no number. That weakens overlap detection
    for that one course; it never weakens dedup, which is id-based.
    """
    numbers: set[str] = set()
    for record in completed_records:
        number = record.get("courseNumber")
        if number:
            numbers.add(str(number))
            continue
        course = courses_by_id.get(str(record.get("courseId") or ""))
        if course and course.get("courseNumber"):
            numbers.add(str(course["courseNumber"]))
    return numbers


__all__ = ["DEFAULT_MAX_CREDITS", "MAX_CREDITS_CEILING", "build_term_plan"]
