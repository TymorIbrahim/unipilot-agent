"""Lesson event utilities for per-group schedule selection."""

from __future__ import annotations

import re
from typing import Any

from app.planning.weekly_schedule import normalize_schedule_group, parse_time_range

SLOT_TYPE_ALIASES: dict[str, tuple[str, ...]] = {
    "lecture": ("lecture", "הרצאה", "lec"),
    "tutorial": ("tutorial", "תרגול", "recitation", "lesson", "תר"),
    "lab": ("lab", "מעבדה", "laboratory"),
    "project": ("project", "פרויקט", "workshop", "סדנה"),
}

SLOT_ORDER = ("lecture", "tutorial", "lab", "project", "workshop", "other")
LEGACY_SLOT_KEYS = ("lecture", "tutorial", "lab", "project")


def _canonical_slot_type(slot_type: str) -> str:
    normalized = slot_type.strip().lower()
    if not normalized:
        return "other"
    for canonical, aliases in SLOT_TYPE_ALIASES.items():
        if normalized in aliases or any(alias.lower() in normalized for alias in aliases):
            return canonical
    return normalized


def group_schedule_by_type(
    schedule_groups: list[dict[str, Any]],
) -> dict[str, list[tuple[int, dict[str, Any]]]]:
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, group in enumerate(schedule_groups or []):
        normalized = normalize_schedule_group(group)
        canonical = _canonical_slot_type(normalized.get("slotType") or "")
        grouped.setdefault(canonical, []).append((index, group))
    return grouped


def normalize_lesson_type(raw: str) -> str:
    return _canonical_slot_type(raw) if raw else "other"


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-") or "na"


def split_time_range(time_range: str) -> tuple[str, str]:
    normalized = time_range.replace("–", "-").replace("—", "-")
    parsed = parse_time_range(normalized)
    if not parsed:
        return "", ""
    start_total, end_total = parsed
    start = f"{start_total // 60:02d}:{start_total % 60:02d}"
    end = f"{end_total // 60:02d}:{end_total % 60:02d}"
    return start, end


def extract_group_label(group: dict[str, Any]) -> str | None:
    for key in ("group", "groupNumber", "קבוצה", "מס.", "number", "Number"):
        value = group.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def extract_instructor(group: dict[str, Any]) -> str | None:
    for key in ("instructor", "מרצה/מתרגל", "lecturer", "ta", "Instructor"):
        value = group.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def extract_location(group: dict[str, Any]) -> str | None:
    building = group.get("building") or group.get("בניין") or group.get("Building")
    room = group.get("room") or group.get("חדר") or group.get("Room")
    parts = [str(part).strip() for part in (building, room) if part is not None and str(part).strip()]
    return " ".join(parts) if parts else None


def extract_notes(group: dict[str, Any]) -> str | None:
    for key in ("notes", "הערות", "Notes"):
        value = group.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def build_lesson_event_id(
    *,
    course_number: str,
    academic_year: int,
    semester_code: int,
    lesson_type: str,
    group_label: str | None,
    day: str,
    start_time: str,
    end_time: str,
    location: str | None = None,
) -> str:
    parts = [
        course_number,
        str(academic_year),
        str(semester_code),
        normalize_lesson_type(lesson_type),
        _slug(group_label or "0"),
        _slug(day),
        start_time.replace(":", ""),
        end_time.replace(":", ""),
    ]
    if location:
        parts.append(_slug(location))
    return "-".join(parts)


def lesson_option_from_group(
    group: dict[str, Any],
    *,
    course_number: str,
    academic_year: int,
    semester_code: int,
    index: int,
) -> dict[str, Any]:
    normalized = normalize_schedule_group(group)
    lesson_type = normalize_lesson_type(normalized.get("slotType") or "")
    group_label = extract_group_label(group)
    start_time, end_time = split_time_range(normalized.get("timeRange") or "")
    location = extract_location(group)
    instructor = extract_instructor(group)
    notes = extract_notes(group)
    incomplete = not normalized.get("day") or not start_time or not end_time

    event_id = build_lesson_event_id(
        course_number=course_number,
        academic_year=academic_year,
        semester_code=semester_code,
        lesson_type=lesson_type,
        group_label=group_label or str(index),
        day=normalized.get("day") or "",
        start_time=start_time or "0000",
        end_time=end_time or "0000",
        location=location,
    )

    return {
        "eventId": event_id,
        "type": lesson_type,
        "group": group_label,
        "index": index,
        "day": normalized.get("day") or "",
        "startTime": start_time,
        "endTime": end_time,
        "timeRange": normalized.get("timeRange") or "",
        "slotTypeLabel": normalized.get("slotType") or lesson_type,
        "instructor": instructor,
        "location": location,
        "notes": notes,
        "incomplete": incomplete,
        "rawGroup": group,
    }


def extract_lesson_options_from_offering(
    offering: dict[str, Any] | None,
    *,
    course_number: str | None = None,
) -> list[dict[str, Any]]:
    if not offering:
        return []
    number = str(course_number or offering.get("courseNumber") or "")
    academic_year = int(offering.get("academicYear") or 0)
    semester_code = int(offering.get("semesterCode") or 0)
    schedule_groups = offering.get("scheduleGroups") or []
    return [
        lesson_option_from_group(
            group,
            course_number=number,
            academic_year=academic_year,
            semester_code=semester_code,
            index=index,
        )
        for index, group in enumerate(schedule_groups)
    ]


def sync_selected_groups_from_events(
    selected_lesson_events: list[dict[str, Any]] | None,
) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {key: [] for key in LEGACY_SLOT_KEYS}
    for event in selected_lesson_events or []:
        lesson_type = normalize_lesson_type(str(event.get("type") or "other"))
        if lesson_type not in groups:
            groups[lesson_type] = []
        group_label = event.get("group")
        if group_label is None:
            continue
        label = str(group_label)
        if label not in groups[lesson_type]:
            groups[lesson_type].append(label)
    return groups


def migrate_legacy_selected_groups(
    planned_course: dict[str, Any],
    schedule_groups: list[dict[str, Any]],
    *,
    course_number: str,
    academic_year: int,
    semester_code: int,
) -> list[dict[str, Any]]:
    """Convert legacy index-based selectedGroups to selectedLessonEvents."""
    selected_groups = planned_course.get("selectedGroups") or {}
    if not isinstance(selected_groups, dict):
        return []

    grouped = group_schedule_by_type(schedule_groups)
    migrated: list[dict[str, Any]] = []

    for slot_key in LEGACY_SLOT_KEYS:
        selection = selected_groups.get(slot_key)
        if selection is None:
            continue
        bucket = grouped.get(slot_key, [])
        if not bucket:
            continue

        if isinstance(selection, int):
            if 0 <= selection < len(bucket):
                option = lesson_option_from_group(
                    bucket[selection][1],
                    course_number=course_number,
                    academic_year=academic_year,
                    semester_code=semester_code,
                    index=bucket[selection][0],
                )
                migrated.append(
                    {
                        "eventId": option["eventId"],
                        "type": option["type"],
                        "group": option.get("group"),
                    }
                )
            continue

        if isinstance(selection, list):
            for group_label in selection:
                for _, group in bucket:
                    option = lesson_option_from_group(
                        group,
                        course_number=course_number,
                        academic_year=academic_year,
                        semester_code=semester_code,
                        index=0,
                    )
                    if str(option.get("group") or "") == str(group_label):
                        migrated.append(
                            {
                                "eventId": option["eventId"],
                                "type": option["type"],
                                "group": option.get("group"),
                            }
                        )
                        break

    return migrated


def normalize_planned_course_lessons(
    planned_course: dict[str, Any],
    *,
    offering: dict[str, Any] | None = None,
    academic_year: int | None = None,
    semester_code: int | None = None,
) -> dict[str, Any]:
    """Ensure selectedLessonEvents and selectedGroups are consistent on read."""
    course = dict(planned_course)
    course_number = str(course.get("courseNumber") or "")
    schedule_groups = (offering or {}).get("scheduleGroups") or []

    year = academic_year or int((offering or {}).get("academicYear") or 0)
    term = semester_code or int((offering or {}).get("semesterCode") or 0)

    selected_events = course.get("selectedLessonEvents")
    if not selected_events and schedule_groups and year and term:
        migrated = migrate_legacy_selected_groups(
            course,
            schedule_groups,
            course_number=course_number,
            academic_year=year,
            semester_code=term,
        )
        if migrated:
            course["selectedLessonEvents"] = migrated

    if course.get("selectedLessonEvents"):
        course["selectedGroups"] = sync_selected_groups_from_events(
            course.get("selectedLessonEvents")
        )
    elif course.get("selectedGroups") is None:
        course["selectedGroups"] = {key: [] for key in LEGACY_SLOT_KEYS}

    return course


def filter_groups_by_lesson_selection(
    schedule_groups: list[dict[str, Any]],
    *,
    selected_lesson_events: list[dict[str, Any]] | None = None,
    selected_groups: dict[str, Any] | None = None,
    course_number: str | None = None,
    academic_year: int | None = None,
    semester_code: int | None = None,
) -> list[dict[str, Any]]:
    """Return only schedule groups chosen by the user."""
    if not schedule_groups:
        return []

    if selected_lesson_events:
        selected_ids = {
            str(event.get("eventId"))
            for event in selected_lesson_events
            if event.get("eventId")
        }
        if not selected_ids:
            return []

        offering_stub = {
            "courseNumber": course_number,
            "academicYear": academic_year,
            "semesterCode": semester_code,
            "scheduleGroups": schedule_groups,
        }
        options = extract_lesson_options_from_offering(
            offering_stub,
            course_number=course_number,
        )
        by_id = {option["eventId"]: option["rawGroup"] for option in options}
        selected: list[dict[str, Any]] = []
        for event_id in selected_ids:
            group = by_id.get(event_id)
            if group is not None:
                selected.append(group)
        return selected

    if not selected_groups:
        return []

    explicit_values = [
        value
        for key, value in selected_groups.items()
        if key in LEGACY_SLOT_KEYS and value is not None
    ]
    if not explicit_values:
        return []

    grouped = group_schedule_by_type(schedule_groups)
    selected: list[dict[str, Any]] = []

    for slot_key in LEGACY_SLOT_KEYS:
        selection = selected_groups.get(slot_key)
        if selection is None:
            continue
        bucket = grouped.get(slot_key, [])
        if not bucket:
            continue

        if isinstance(selection, int):
            if 0 <= selection < len(bucket):
                selected.append(bucket[selection][1])
            continue

        if isinstance(selection, list):
            for group_label in selection:
                for _, group in bucket:
                    option = lesson_option_from_group(
                        group,
                        course_number=str(course_number or ""),
                        academic_year=int(academic_year or 0),
                        semester_code=int(semester_code or 0),
                        index=0,
                    )
                    if str(option.get("group") or "") == str(group_label):
                        selected.append(group)
                        break
            continue

        if isinstance(selection, str):
            for _, group in bucket:
                normalized = normalize_schedule_group(group)
                if normalized.get("slotType") == selection:
                    selected.append(group)
                    break

    return selected


def validate_lesson_selection(
    selected_lesson_events: list[dict[str, Any]],
    available_options: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    available_ids = {str(option["eventId"]) for option in available_options}

    for event in selected_lesson_events:
        event_id = str(event.get("eventId") or "").strip()
        if not event_id:
            errors.append("Each selected lesson event must include eventId")
            continue
        if event_id in seen:
            errors.append(f"Duplicate selected lesson event: {event_id}")
        seen.add(event_id)
        if event_id not in available_ids:
            errors.append(f"Selected lesson event is not available: {event_id}")

    return errors


def build_lesson_selection_warnings(
    planned_course: dict[str, Any],
    offering: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if planned_course.get("isActive") is False:
        return []

    course_number = str(planned_course.get("courseNumber") or "")
    course_id = planned_course.get("courseId")
    options = extract_lesson_options_from_offering(offering, course_number=course_number)
    selected = planned_course.get("selectedLessonEvents") or []
    warnings: list[dict[str, Any]] = []

    if offering is None:
        return warnings

    if not options:
        warnings.append(
            {
                "courseNumber": course_number,
                "courseId": course_id,
                "type": "no_lesson_options",
                "message": "No lesson options available for this course.",
            }
        )
        return warnings

    if not selected:
        warnings.append(
            {
                "courseNumber": course_number,
                "courseId": course_id,
                "type": "no_lesson_selected",
                "message": (
                    "Choose lecture/tutorial/lab groups to place this course in the schedule."
                ),
            }
        )
        return warnings

    available_ids = {option["eventId"] for option in options}
    options_by_id = {option["eventId"]: option for option in options}

    for event in selected:
        event_id = str(event.get("eventId") or "")
        if event_id not in available_ids:
            warnings.append(
                {
                    "courseNumber": course_number,
                    "courseId": course_id,
                    "type": "stale_lesson_event",
                    "eventId": event_id,
                    "message": "This selected lesson is no longer available in the latest course data.",
                }
            )
            continue
        option = options_by_id[event_id]
        if option.get("incomplete"):
            warnings.append(
                {
                    "courseNumber": course_number,
                    "courseId": course_id,
                    "type": "incomplete_lesson_data",
                    "eventId": event_id,
                    "message": "Selected lesson has incomplete time or day data.",
                }
            )

    return warnings
