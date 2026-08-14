"""Resolve catalog prerequisites for semester planning."""

from __future__ import annotations

import re
from typing import Any

COURSE_NUMBER_PATTERN = re.compile(r"(?<!\d)(0\d{6,8}|\d{7,8})(?!\d)")


def canonical_course_number(raw: str | None) -> str | None:
    """Normalize Technion course numbers to 8-digit 0-prefixed strings."""
    digits = re.sub(r"\D", "", str(raw or ""))
    if not digits or len(digits) < 6 or len(digits) > 9:
        return None
    padded = digits.zfill(8)[-8:]
    if not re.fullmatch(r"0\d{7}", padded):
        return None
    return padded


def extract_course_numbers_from_text(text: str | None) -> list[str]:
    if not text:
        return []
    numbers: list[str] = []
    seen: set[str] = set()
    for match in COURSE_NUMBER_PATTERN.finditer(str(text)):
        number = canonical_course_number(match.group(1))
        if number and number not in seen:
            seen.add(number)
            numbers.append(number)
    return numbers


def resolve_prerequisite_ids(
    course: dict[str, Any],
    *,
    courses_by_number: dict[str, dict[str, Any]],
) -> list[Any]:
    explicit = course.get("prerequisites")
    if explicit:
        return list(explicit)

    numbers = extract_course_numbers_from_text(course.get("prerequisitesText"))
    resolved: list[Any] = []
    for number in numbers:
        match = courses_by_number.get(number)
        if match is None:
            # tolerate legacy 7-digit keys in catalog indexes
            alt = canonical_course_number(number)
            if alt:
                match = courses_by_number.get(alt)
        if match and match.get("_id") is not None:
            resolved.append(match["_id"])
    return resolved


def build_courses_by_number(catalog_courses: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for course in catalog_courses:
        number = course.get("courseNumber") or course.get("number")
        if number is None:
            continue
        canonical = canonical_course_number(str(number)) or str(number)
        indexed[canonical] = course
        indexed[str(number)] = course
    return indexed
