"""Plan semester codes (YYYY-1/2/3) ↔ Technion offering keys (academicYear + 200/201/202)."""

from __future__ import annotations

import re
from typing import Any

PLAN_SEMESTER_PATTERN = re.compile(r"^(\d{4})-([123])$")

TERM_NAME_TO_INDEX = {"winter": 1, "spring": 2, "summer": 3}
_YEAR_NAME_PATTERN = re.compile(r"^(\d{4})-(winter|spring|summer)$")


def plan_semester_to_offering_keys(semester_code: str) -> tuple[int, int] | None:
    """Map plan code like 2025-2 to offering academicYear=2025, semesterCode=201."""
    match = PLAN_SEMESTER_PATTERN.match(str(semester_code).strip())
    if not match:
        return None
    academic_year = int(match.group(1))
    term_index = int(match.group(2))
    return academic_year, 200 + term_index - 1


def resolve_plan_term(raw_term: str, *, preferred_year: int) -> tuple[str, int, int] | None:
    """Resolve a term the agent may NAME or CODE to (label, academicYear, semesterCode).

    The agent reasons about terms by name ("winter"); the offering keys are
    year-specific. Both forms are accepted, and the returned LABEL is the caller's
    own string echoed back unchanged, so the plan is split on exactly what was
    passed:
      - "2025-1" / "2025-2" / "2025-3"   year-coded (winter / spring / summer)
      - "winter" / "spring" / "summer"   bare name -> preferred_year
      - "2025-winter" / "2025-spring"    year + name
    academicYear is a PREFERENCE -- offering lookup falls back to the nearest year
    -- so an approximate preferred_year for a bare name still finds offerings.
    """
    label = str(raw_term).strip()

    coded = plan_semester_to_offering_keys(label)
    if coded is not None:
        return label, coded[0], coded[1]

    lowered = label.lower()
    year_name = _YEAR_NAME_PATTERN.match(lowered)
    if year_name:
        index = TERM_NAME_TO_INDEX[year_name.group(2)]
        return label, int(year_name.group(1)), 200 + index - 1

    index = TERM_NAME_TO_INDEX.get(lowered)
    if index is not None:
        return label, preferred_year, 200 + index - 1

    return None


def offering_keys_to_plan_semester_code(academic_year: int, semester_code: int) -> str | None:
    """Map offering academicYear + semesterCode (200/201/202) to plan code YYYY-1/2/3."""
    if semester_code not in {200, 201, 202}:
        return None
    term_index = semester_code - 199
    return f"{academic_year}-{term_index}"


def pick_best_offering(
    offerings: list[dict[str, Any]],
    *,
    preferred_academic_year: int,
    semester_code: int,
) -> dict[str, Any] | None:
    """Choose an offering for the term, preferring exact year then nearest catalog year."""
    same_term = [
        offering
        for offering in offerings
        if int(offering.get("semesterCode") or 0) == semester_code
    ]
    if not same_term:
        return None

    exact = next(
        (
            offering
            for offering in same_term
            if int(offering.get("academicYear") or 0) == preferred_academic_year
        ),
        None,
    )
    if exact:
        return exact

    return min(
        same_term,
        key=lambda offering: abs(int(offering.get("academicYear") or 0) - preferred_academic_year),
    )
