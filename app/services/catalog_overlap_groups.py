"""Catalog overlap rules from Technion semester JSON (מקצועות ללא זיכוי נוסף)."""

from __future__ import annotations

from typing import Any

from app.planning.prerequisite_resolver import extract_course_numbers_from_text
from app.services.completed_course_attempts import latest_attempt_rank
from app.planning.support import course_number_keys, merge_overlapping_equivalence_groups


def build_catalog_overlap_groups(catalog_courses: list[dict[str, Any]]) -> list[set[str]]:
    """Build mutual-exclusion / substitution groups from noAdditionalCreditText."""
    groups: list[set[str]] = []
    for course in catalog_courses:
        number = course.get("courseNumber") or course.get("number")
        if number is None:
            continue
        overlap_text = course.get("noAdditionalCreditText")
        if not overlap_text:
            continue

        members = set(course_number_keys(str(number)))
        for overlap_number in extract_course_numbers_from_text(str(overlap_text)):
            members |= course_number_keys(overlap_number)

        if len(members) > 1:
            groups.append(members)

    return merge_overlapping_equivalence_groups(groups)


def collect_overlap_partner_numbers(catalog_courses: list[dict[str, Any]]) -> set[str]:
    """Course numbers referenced in noAdditionalCreditText for loaded catalog rows."""
    partners: set[str] = set()
    for course in catalog_courses:
        overlap_text = course.get("noAdditionalCreditText")
        if not overlap_text:
            continue
        for number in extract_course_numbers_from_text(str(overlap_text)):
            partners |= course_number_keys(number)
    return partners


def expand_keys_with_equivalence(
    keys: set[str],
    equivalence_groups: list[set[str]],
) -> set[str]:
    """Union all equivalence groups that intersect the given course-number keys."""
    if not keys or not equivalence_groups:
        return set(keys)

    expanded = set(keys)
    for group in equivalence_groups:
        if expanded & group:
            expanded |= group
    return expanded


def overlap_group_for_course(
    course_number: str | None,
    overlap_groups: list[set[str]],
) -> frozenset[str] | None:
    if not course_number or not overlap_groups:
        return None
    keys = course_number_keys(course_number)
    for group in overlap_groups:
        if keys & group:
            return frozenset(group)
    return None


def _completion_precedence_key(
    completion: dict[str, Any],
    *,
    recorded_at_timestamp,
) -> tuple[int, float, str]:
    """Latest completion wins within a catalog overlap group (not max credits)."""
    return latest_attempt_rank(
        attempt=int(completion.get("attempt") or 1),
        recorded_at_timestamp=recorded_at_timestamp(completion.get("recordedAt")),
        semester_code=str(completion.get("semesterCode") or ""),
    )


def exclude_overlap_duplicate_credits(
    effective_completions: dict[str, dict[str, Any]],
    catalog_courses_by_id: dict[str, dict[str, Any]],
    overlap_groups: list[set[str]],
    *,
    recorded_at_timestamp,
) -> set[str]:
    """Return course ids excluded from credit totals when overlap rules forbid double counting."""
    if not overlap_groups:
        return set()

    id_to_keys: dict[str, set[str]] = {}
    for course_id in effective_completions:
        catalog_course = catalog_courses_by_id.get(course_id)
        if not catalog_course:
            continue
        number = catalog_course.get("courseNumber") or catalog_course.get("number")
        if number is None:
            continue
        id_to_keys[course_id] = course_number_keys(str(number))

    excluded_ids: set[str] = set()
    for group in overlap_groups:
        members = [
            (course_id, completion)
            for course_id, completion in effective_completions.items()
            if id_to_keys.get(course_id, set()) & group
        ]
        if len(members) <= 1:
            continue

        winner_id, _ = max(
            members,
            key=lambda item: _completion_precedence_key(
                item[1],
                recorded_at_timestamp=recorded_at_timestamp,
            ),
        )
        for course_id, _ in members:
            if course_id != winner_id:
                excluded_ids.add(course_id)

    return excluded_ids
