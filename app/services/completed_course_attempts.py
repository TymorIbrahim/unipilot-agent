"""Attempt numbering for retaken courses on a student transcript."""

from __future__ import annotations

import re

MAX_COURSE_ATTEMPTS = 10

SEMESTER_CODE_PATTERN = re.compile(r"^(\d{4})-([123])$")


def semester_code_rank(semester_code: str = "") -> tuple[int, int]:
    """Sortable (academicYear, termIndex) for Technion YYYY-1/2/3 codes."""
    match = SEMESTER_CODE_PATTERN.match(str(semester_code or "").strip())
    if not match:
        return (0, 0)
    return (int(match.group(1)), int(match.group(2)))


def resolve_available_attempt(used_attempts: set[int], requested: int = 1) -> int:
    """Return requested attempt when free, otherwise the next open attempt slot."""
    normalized_requested = max(1, min(int(requested), MAX_COURSE_ATTEMPTS))
    if normalized_requested not in used_attempts:
        return normalized_requested

    for attempt in range(1, MAX_COURSE_ATTEMPTS + 1):
        if attempt not in used_attempts:
            return attempt

    raise ValueError(f"Maximum of {MAX_COURSE_ATTEMPTS} attempts reached for this course")


def latest_attempt_rank(
    *,
    attempt: int,
    recorded_at_timestamp: float,
    semester_code: str = "",
) -> tuple[int, int, int, float]:
    """Rank transcript rows: later semester wins, then attempt, then recordedAt."""
    year, term = semester_code_rank(semester_code)
    return (
        year,
        term,
        max(1, int(attempt)),
        recorded_at_timestamp,
    )
