"""Did the student pass, and with what number?

Ported from UniPilot unchanged, including the threshold. It is a REGULATION, not
a tuning knob: 55 is the Technion's passing grade, and a course below it does not
count towards a degree however many credits it carried.

Kept as its own module rather than folded into the planner because "has this
course been passed" is asked by anything that reasons about what remains, and two
answers to that question is the kind of divergence nobody notices until a student
is told they have graduated.
"""

from __future__ import annotations

from typing import Any

PASSING_GRADE_THRESHOLD = 55


def parse_numeric_grade(value: Any) -> float | None:
    """A stored grade as a number, or None when it is not one.

    Booleans are refused explicitly: `bool` is a subclass of `int` in Python, so
    `True` would otherwise read as the grade 1.0 and quietly fail the threshold.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def is_passing_numeric_grade(numeric_grade: float) -> bool:
    return numeric_grade >= PASSING_GRADE_THRESHOLD


def resolve_record_numeric_grade(record: dict[str, Any]) -> float | None:
    """The official numeric grade, falling back to `gradePoints`.

    A transcript row carries one or the other depending on how it was imported;
    preferring `grade` keeps the number the registrar published when both exist.
    """
    grade = parse_numeric_grade(record.get("grade"))
    if grade is not None:
        return grade
    return parse_numeric_grade(record.get("gradePoints"))


def is_passing_grade(record: dict[str, Any] | Any, grade_points: Any = None) -> bool:
    """True when the student passed.

    An UNGRADED record is not a pass. That is the conservative direction: counting
    it would credit a course that may still be failed, and telling a student they
    have finished when they have not is the worst error this function can make.
    """
    if isinstance(record, dict):
        grade = parse_numeric_grade(record.get("grade"))
        if grade is not None:
            return is_passing_numeric_grade(grade)
        points = parse_numeric_grade(record.get("gradePoints"))
        if points is not None:
            return is_passing_numeric_grade(points)
        return False

    numeric = parse_numeric_grade(grade_points)
    if numeric is None:
        numeric = parse_numeric_grade(record)
    if numeric is None:
        return False
    return is_passing_numeric_grade(numeric)


__all__ = [
    "PASSING_GRADE_THRESHOLD",
    "is_passing_grade",
    "is_passing_numeric_grade",
    "parse_numeric_grade",
    "resolve_record_numeric_grade",
]
