"""Weekly schedule conflict detection and week view (Phase 16.1)."""

from __future__ import annotations

import re
from typing import Any

HEBREW_DAY_ORDER = [
    "ראשון",
    "שני",
    "שלישי",
    "רביעי",
    "חמישי",
    "שישי",
    "שבת",
]

ENGLISH_DAY_ORDER = [
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
]

TIME_RANGE_PATTERN = re.compile(
    r"^\s*(\d{1,2}):(\d{2})\s*[-–—]\s*(\d{1,2}):(\d{2})\s*$"
)


def _day_sort_key(day: str) -> tuple[int, str]:
    normalized = day.strip()
    if normalized in HEBREW_DAY_ORDER:
        return (0, str(HEBREW_DAY_ORDER.index(normalized)))
    if normalized in ENGLISH_DAY_ORDER:
        return (1, str(ENGLISH_DAY_ORDER.index(normalized)))
    return (2, normalized)


def normalize_schedule_group(group: dict[str, Any]) -> dict[str, str]:
    day = str(
        group.get("יום")
        or group.get("day")
        or group.get("Day")
        or ""
    ).strip()
    time_range = str(
        group.get("שעה")
        or group.get("time")
        or group.get("Time")
        or ""
    ).strip()
    slot_type = str(
        group.get("סוג")
        or group.get("type")
        or group.get("Type")
        or ""
    ).strip()
    return {"day": day, "timeRange": time_range, "slotType": slot_type}


def parse_time_range(time_range: str) -> tuple[int, int] | None:
    match = TIME_RANGE_PATTERN.match(time_range.replace("–", "-").replace("—", "-"))
    if not match:
        return None
    start_hour, start_minute, end_hour, end_minute = (int(value) for value in match.groups())
    start_total = start_hour * 60 + start_minute
    end_total = end_hour * 60 + end_minute
    if end_total <= start_total:
        return None
    return start_total, end_total


def _schedule_slots(entry: dict[str, Any]) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    for group in entry.get("scheduleGroups") or []:
        normalized = normalize_schedule_group(group)
        parsed = parse_time_range(normalized["timeRange"])
        if not normalized["day"] or parsed is None:
            continue
        start_minutes, end_minutes = parsed
        slots.append(
            {
                "day": normalized["day"],
                "timeRange": normalized["timeRange"],
                "slotType": normalized["slotType"],
                "startMinutes": start_minutes,
                "endMinutes": end_minutes,
                "courseNumber": entry.get("courseNumber"),
                "courseTitle": entry.get("courseTitle"),
            }
        )
    return slots


def _slots_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left["day"] != right["day"]:
        return False
    return left["startMinutes"] < right["endMinutes"] and right["startMinutes"] < left["endMinutes"]


def detect_schedule_conflicts(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    all_slots: list[dict[str, Any]] = []
    for entry in entries:
        all_slots.extend(_schedule_slots(entry))

    conflicts: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str, str]] = set()

    for index, left in enumerate(all_slots):
        for right in all_slots[index + 1 :]:
            if left["courseNumber"] == right["courseNumber"]:
                continue
            if not _slots_overlap(left, right):
                continue

            pair_key = tuple(
                sorted(
                    [
                        str(left["courseNumber"]),
                        str(right["courseNumber"]),
                        f"{left['day']}|{left['timeRange']}",
                    ]
                )
            )
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            course_numbers = sorted(
                {str(left["courseNumber"]), str(right["courseNumber"])},
                key=lambda value: value or "",
            )
            conflicts.append(
                {
                    "day": left["day"],
                    "timeRange": left["timeRange"],
                    "courseNumbers": course_numbers,
                    "reason": "Overlapping schedule slots",
                }
            )

    return sorted(
        conflicts,
        key=lambda conflict: (
            _day_sort_key(str(conflict.get("day") or "")),
            str(conflict.get("timeRange") or ""),
        ),
    )


def build_week_view(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        for slot in _schedule_slots(entry):
            grouped.setdefault(slot["day"], []).append(
                {
                    "timeRange": slot["timeRange"],
                    "slotType": slot["slotType"],
                    "courseNumber": slot["courseNumber"],
                    "courseTitle": slot["courseTitle"],
                }
            )

    week_view: list[dict[str, Any]] = []
    for day in sorted(grouped.keys(), key=_day_sort_key):
        slots = sorted(
            grouped[day],
            key=lambda slot: parse_time_range(str(slot.get("timeRange") or "")) or (9999, 9999),
        )
        week_view.append({"day": day, "slots": slots})
    return week_view


def summarize_slot_types(schedule_groups: list[dict[str, Any]]) -> list[str]:
    types: set[str] = set()
    for group in schedule_groups or []:
        normalized = normalize_schedule_group(group)
        slot_type = normalized.get("slotType") or ""
        if slot_type:
            types.add(slot_type)
    return sorted(types)


def build_weekly_schedule_payload(
    entries: list[dict[str, Any]],
    *,
    custom_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    merged_entries = list(entries)
    for event in custom_events or []:
        day = str(event.get("day") or "").strip()
        start_time = str(event.get("startTime") or "").strip()
        end_time = str(event.get("endTime") or "").strip()
        if not day or not start_time or not end_time:
            continue
        merged_entries.append(
            {
                "courseId": str(event.get("id") or event.get("title") or "custom"),
                "courseNumber": "CUSTOM",
                "courseTitle": str(event.get("title") or "Custom event"),
                "scheduleGroups": [
                    {
                        "day": day,
                        "time": f"{start_time}-{end_time}",
                        "type": "custom",
                    }
                ],
                "isCustomEvent": True,
            }
        )

    conflicts = detect_schedule_conflicts(merged_entries)
    if not merged_entries:
        status = "empty"
    elif conflicts:
        status = "conflicts"
    else:
        status = "valid"

    summary = f"{len(entries)} course(s) scheduled"
    if custom_events:
        summary = f"{summary}; {len(custom_events)} custom block(s)"
    if conflicts:
        summary = f"{summary}; {len(conflicts)} conflict(s)"
    else:
        summary = f"{summary}; no conflicts"

    return {
        "status": status,
        "entries": entries,
        "customEvents": custom_events or [],
        "conflicts": conflicts,
        "weekView": build_week_view(merged_entries),
        "summary": summary,
    }
