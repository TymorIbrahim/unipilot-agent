"""Five small helpers the term planner needs, lifted out of their origins.

In UniPilot these live in three modules totalling ~2,100 lines --
`semester_planner`, `course_reference_keys`, and `graduation_progress_calculator`
-- which the planner imported in full for these five functions alone. Copying
those modules would have brought their own dependency trees (and three MongoDB
repositories) into a service that has no MongoDB.

They are reproduced VERBATIM, deliberately. Each encodes a convention the rest of
the ported planner already assumes, and the conventions are narrower than they
look: `normalize_course_id` is a bare `str()` with no stripping, `round_credits`
carries an epsilon so a credit total lands on the right side of .005, and
`course_number_keys` returns a SET. A tidier-looking reimplementation of any of
them changes results the tests here would not catch, because the tests came from
the same codebase and share its assumptions.
"""

from __future__ import annotations

from typing import Any

from app.planning.prerequisite_resolver import canonical_course_number

# --- from app.services.graduation_progress_calculator -------------------------


def round_credits(value: float) -> float:
    # The epsilon is load-bearing: binary floats leave sums like 19.499999997,
    # which would round down and under-report a credit total by 0.01.
    return round(float(value) + 1e-9, 2)


# --- from app.planning.semester_planner ---------------------------------------


def normalize_course_id(course_id: Any) -> str:
    return str(course_id)


def get_course_credits(course: dict[str, Any]) -> float:
    return round_credits(course.get("credits") or 0)


def prerequisites_met(course: dict[str, Any], satisfied_course_ids: set[str]) -> bool:
    prerequisite_ids = [
        normalize_course_id(course_id) for course_id in (course.get("prerequisites") or [])
    ]
    return all(course_id in satisfied_course_ids for course_id in prerequisite_ids)


# --- from app.services.course_reference_keys ----------------------------------


def course_number_keys(raw: str | None) -> set[str]:
    if raw is None:
        return set()
    value = str(raw)
    keys = {value}
    canonical = canonical_course_number(value)
    if canonical:
        keys.add(canonical)
    return keys


def merge_overlapping_equivalence_groups(groups: list[set[str]]) -> list[set[str]]:
    """Union matrix rows that refer to the same course slot (duplicate catalog rows)."""
    merged: list[set[str]] = []
    for raw_group in groups:
        group = set(raw_group)
        if not group:
            continue

        overlap_indexes: list[int] = []
        for index, existing in enumerate(merged):
            if group & existing:
                overlap_indexes.append(index)

        for index in reversed(overlap_indexes):
            group |= merged.pop(index)

        merged.append(group)

    return merged


__all__ = [
    "course_number_keys",
    "get_course_credits",
    "merge_overlapping_equivalence_groups",
    "normalize_course_id",
    "prerequisites_met",
    "round_credits",
]
