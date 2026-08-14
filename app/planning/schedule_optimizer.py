"""Conflict-aware course selection and weekly schedule optimization."""

from __future__ import annotations

import itertools
from typing import Any

from app.planning.exam_summary import build_exam_summary, exams_from_offering
from app.planning.lesson_events import extract_lesson_options_from_offering, normalize_lesson_type
from app.planning.prerequisite_resolver import canonical_course_number
from app.planning.weekly_schedule import parse_time_range


def _planned_number_key(course_number: str) -> str:
    if not course_number:
        return ""
    return canonical_course_number(course_number) or course_number


def _offering_for_planned_number(
    offerings_by_number: dict[str, dict[str, Any]],
    course_number: str,
) -> dict[str, Any] | None:
    if not course_number:
        return None
    direct = offerings_by_number.get(course_number)
    if direct is not None:
        return direct
    canonical = canonical_course_number(course_number)
    if canonical:
        return offerings_by_number.get(canonical)
    return None


def _option_slot(option: dict[str, Any]) -> dict[str, Any] | None:
    parsed = parse_time_range(str(option.get("timeRange") or "").replace("–", "-").replace("—", "-"))
    if not option.get("day") or parsed is None:
        return None
    start_minutes, end_minutes = parsed
    return {
        "day": str(option["day"]),
        "startMinutes": start_minutes,
        "endMinutes": end_minutes,
        "courseNumber": str(option.get("courseNumber") or ""),
        "eventId": str(option.get("eventId") or ""),
        "type": str(option.get("type") or "other"),
        "group": option.get("group"),
    }


def slots_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left["day"] != right["day"]:
        return False
    return left["startMinutes"] < right["endMinutes"] and right["startMinutes"] < left["endMinutes"]


def _group_options_by_type(options: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for option in options:
        if option.get("incomplete"):
            continue
        lesson_type = normalize_lesson_type(str(option.get("type") or "other"))
        grouped.setdefault(lesson_type, []).append(option)
    return grouped


def _lesson_events_from_options(options: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "eventId": str(option["eventId"]),
            "type": str(option.get("type") or "other"),
            "group": option.get("group"),
        }
        for option in options
        if option.get("eventId")
    ]


def pick_lessons_for_course(
    options: list[dict[str, Any]],
    *,
    occupied_slots: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    """Pick one lesson per type with the fewest overlaps against occupied slots."""
    grouped = _group_options_by_type(options)
    if not grouped:
        return []

    type_keys = sorted(grouped.keys())
    combinations = itertools.product(*[grouped[key] for key in type_keys])
    best: list[dict[str, Any]] | None = None
    best_score: tuple[int, int] | None = None

    for combo in combinations:
        candidate_slots = [_option_slot(option) for option in combo]
        if any(slot is None for slot in candidate_slots):
            continue
        valid_slots = [slot for slot in candidate_slots if slot is not None]
        overlap_count = 0
        for slot in valid_slots:
            for occupied in occupied_slots:
                if slots_overlap(slot, occupied):
                    overlap_count += 1
                    break
        internal_conflicts = 0
        for left_index, left in enumerate(valid_slots):
            for right in valid_slots[left_index + 1 :]:
                if slots_overlap(left, right):
                    internal_conflicts += 1
        if internal_conflicts > 0:
            continue
        score = (overlap_count, sum(slot["startMinutes"] for slot in valid_slots))
        if best_score is None or score < best_score:
            best_score = score
            best = list(combo)

    if best is None:
        return None
    return _lesson_events_from_options(best)


def _exam_entries_for_course(
    offering: dict[str, Any] | None,
    *,
    course_number: str,
    course_title: str,
) -> list[dict[str, Any]]:
    return exams_from_offering(
        offering,
        course_number=course_number,
        course_name=course_title,
    )


def _offering_is_schedulable(
    offering: dict[str, Any] | None,
    *,
    course_number: str,
) -> tuple[bool, list[dict[str, Any]]]:
    if not offering:
        return False, []
    options = extract_lesson_options_from_offering(offering, course_number=course_number)
    return bool(options), options


def exams_conflict(existing: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> bool:
    existing_dates = {str(entry["date"]) for entry in existing if entry.get("date")}
    for entry in candidate:
        date_key = str(entry.get("date") or "")
        if date_key and date_key in existing_dates:
            return True
    return False


def _merge_selection_state(
    target: dict[str, Any],
    batch: dict[str, Any],
) -> None:
    """Sync scalar carry-over fields after an in-place batch selection."""
    target["totalCredits"] = batch["totalCredits"]
    target["occupiedSlots"] = batch["occupiedSlots"]
    target["examEntries"] = batch["examEntries"]
    target["localSatisfied"] = batch["localSatisfied"]
    target["plannedCourseNumbers"] = batch.get("plannedCourseNumbers") or set()


def _empty_selection_state(
    *,
    satisfied_course_ids: set[str],
) -> dict[str, Any]:
    return {
        "selectedCourses": [],
        "skippedDueToWorkload": [],
        "skippedDueToConflicts": [],
        "skippedDueToUnavailable": [],
        "totalCredits": 0.0,
        "occupiedSlots": [],
        "examEntries": [],
        "localSatisfied": set(satisfied_course_ids),
        "plannedCourseNumbers": set(),
    }


def _seed_planned_conflicts_from_offering(
    state: dict[str, Any],
    *,
    planned: dict[str, Any],
    course_number: str,
    offerings_by_number: dict[str, dict[str, Any]],
) -> None:
    """Reserve exam and weekly slots from draft picks even when the course is inactive."""
    offering = _offering_for_planned_number(offerings_by_number, course_number)
    if not offering:
        return

    course_title = str(planned.get("courseTitle") or "")
    state["examEntries"].extend(
        _exam_entries_for_course(
            offering,
            course_number=course_number,
            course_title=course_title,
        )
    )
    selected_events = planned.get("selectedLessonEvents") or []
    if not selected_events:
        return

    options = extract_lesson_options_from_offering(offering, course_number=course_number)
    selected_ids = {str(event.get("eventId") or "") for event in selected_events}
    for option in options:
        if str(option.get("eventId") or "") not in selected_ids:
            continue
        slot = _option_slot({**option, "courseNumber": course_number})
        if slot:
            state["occupiedSlots"].append(slot)








def _dedupe_selected_courses(courses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_ids: set[str] = set()
    unique: list[dict[str, Any]] = []
    for course in courses:
        course_id = str(course.get("courseId") or "")
        if not course_id or course_id in seen_ids:
            continue
        seen_ids.add(course_id)
        unique.append(course)
    return unique


def optimize_schedule_for_planned_courses(
    planned_courses: list[dict[str, Any]],
    *,
    offerings_by_number: dict[str, dict[str, Any]],
    academic_year: int,
    semester_code: int,
) -> dict[str, Any]:
    """Assign lesson events across existing planned courses without conflicts."""
    occupied_slots: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for planned in planned_courses:
        if planned.get("isActive", True) is False:
            continue
        course_number = str(planned.get("courseNumber") or "")
        offering = _offering_for_planned_number(offerings_by_number, course_number)
        options = extract_lesson_options_from_offering(
            offering,
            course_number=course_number,
        )
        if not options:
            skipped.append(
                {
                    "courseNumber": course_number,
                    "reason": "No published offering schedule for this semester",
                }
            )
            continue

        selected_lessons = pick_lessons_for_course(options, occupied_slots=occupied_slots)
        if selected_lessons is None:
            skipped.append(
                {
                    "courseNumber": course_number,
                    "reason": "No conflict-free lesson combination found",
                }
            )
            continue

        selections.append(
            {
                "courseNumber": course_number,
                "selectedLessonEvents": selected_lessons,
            }
        )
        for option in options:
            if any(event["eventId"] == option.get("eventId") for event in selected_lessons):
                slot = _option_slot({**option, "courseNumber": course_number})
                if slot:
                    occupied_slots.append(slot)

    exam_summary = build_exam_summary(
        [
            {
                **planned,
                "selectedLessonEvents": next(
                    (
                        item["selectedLessonEvents"]
                        for item in selections
                        if item["courseNumber"] == str(planned.get("courseNumber") or "")
                    ),
                    planned.get("selectedLessonEvents"),
                ),
            }
            for planned in planned_courses
            if planned.get("isActive", True) is not False
        ],
        offerings_by_number,
    )

    return {
        "selections": selections,
        "skippedCourses": skipped,
        "examSummary": exam_summary,
    }
