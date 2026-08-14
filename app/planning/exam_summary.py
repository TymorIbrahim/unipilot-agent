"""Exam date parsing and summary for semester planner."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

MOED_A_KEYS = frozenset({"moedA", "moed_a", "examA", "exam_a", "Moed A", "מועד א", "מועד א'", "מועד א׳"})
MOED_B_KEYS = frozenset({"moedB", "moed_b", "examB", "exam_b", "Moed B", "מועד ב", "מועד ב'", "מועד ב׳"})

DATE_PATTERNS = (
    re.compile(r"^(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{1,2}):(\d{2}))?"),
    re.compile(r"^(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})(?:[ T](\d{1,2}):(\d{2}))?"),
)


def _parse_exam_datetime(raw: str | None) -> tuple[date | None, str | None, str | None]:
    if not raw or not str(raw).strip():
        return None, None, None
    text = str(raw).strip()
    for pattern in DATE_PATTERNS:
        match = pattern.match(text)
        if not match:
            continue
        groups = match.groups()
        if len(groups) >= 3 and len(groups[0]) == 4:
            year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
        else:
            day, month, year = int(groups[0]), int(groups[1]), int(groups[2])
        start_time = None
        if groups[3] is not None and groups[4] is not None:
            start_time = f"{int(groups[3]):02d}:{groups[4]}"
        try:
            return date(year, month, day), start_time, text
        except ValueError:
            continue
    return None, None, text


def _moed_from_key(key: str) -> str | None:
    normalized = key.strip()
    if normalized in MOED_A_KEYS:
        return "A"
    if normalized in MOED_B_KEYS:
        return "B"
    lower = normalized.lower()
    if "exama" in lower or "moeda" in lower or "moed_a" in lower:
        return "A"
    if "examb" in lower or "moedb" in lower or "moed_b" in lower:
        return "B"
    if "מועד א" in normalized:
        return "A"
    if "מועד ב" in normalized:
        return "B"
    return None


def exams_from_offering(
    offering: dict[str, Any] | None,
    *,
    course_number: str,
    course_name: str,
) -> list[dict[str, Any]]:
    if not offering:
        return []

    exam_dates = offering.get("examDates") or {}
    if not exam_dates:
        return []

    items: list[dict[str, Any]] = []
    for key, value in exam_dates.items():
        moed = _moed_from_key(str(key))
        parsed_date, start_time, raw = _parse_exam_datetime(value)
        if parsed_date is None:
            continue
        items.append(
            {
                "courseNumber": course_number,
                "courseName": course_name,
                "moed": moed or str(key),
                "date": parsed_date.isoformat(),
                "startTime": start_time,
                "endTime": None,
                "raw": raw or str(value) if value else None,
                "isMissing": False,
            }
        )
    return items


def build_exam_summary(
    planned_courses: list[dict[str, Any]],
    offerings_by_course_number: dict[str, dict[str, Any]],
    *,
    include_inactive: bool = False,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for planned in planned_courses:
        is_active = planned.get("isActive", True)
        if not include_inactive and is_active is False:
            continue
        course_number = str(planned.get("courseNumber") or "")
        course_name = str(planned.get("courseTitle") or "")
        offering = offerings_by_course_number.get(course_number)
        entries.extend(
            exams_from_offering(
                offering,
                course_number=course_number,
                course_name=course_name,
            )
        )

    sortable = [
        entry
        for entry in entries
        if entry.get("date") is not None
    ]
    sortable.sort(key=lambda item: (item["date"], item.get("startTime") or "", item["courseNumber"]))

    warnings: list[dict[str, Any]] = []
    by_date: dict[str, list[str]] = {}
    for entry in sortable:
        date_key = str(entry["date"])
        by_date.setdefault(date_key, []).append(entry["courseNumber"])

    for date_key, course_numbers in by_date.items():
        unique = sorted(set(course_numbers))
        if len(unique) > 1:
            warnings.append(
                {
                    "type": "same_day_exams",
                    "date": date_key,
                    "courseNumbers": unique,
                    "message": f"Multiple exams on {date_key}: {', '.join(unique)}",
                }
            )

    return {
        "exams": sortable,
        "warnings": warnings,
        "totalExams": len(sortable),
        "missingCount": 0,
    }


def active_planned_courses(planned_courses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [course for course in planned_courses if course.get("isActive", True) is not False]
