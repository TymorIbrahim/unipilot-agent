"""Unit tests for the plan post-conditions -- the deterministic oracle.

The headline case is the real 2026-07-20 winter run (winter-plan-...Z.json): six
per-course minimums, computed in isolation against a standing that already clears
the floor. It is wrong two independent ways, and both are pure arithmetic:

  - two minimums are negative (an impossible grade), and
  - earning all six at once drops the GPA to 65 against an 80 floor.

No live run, no model call -- these tests pin the checks that a future verify
step and the harness both rely on.
"""

from __future__ import annotations

import pytest

from app.agent_core.facts.postconditions import (
    GradedCourse,
    Standing,
    check_gpa_in_range,
    check_grades_in_range,
    check_joint_floor,
    check_plan,
    check_term_load,
)


def test_check_term_load_flags_a_term_over_the_ceiling():
    violations = check_term_load(83.0, "Winter")

    assert len(violations) == 1
    assert violations[0].kind == "term_load"
    assert "83 credits" in violations[0].message


@pytest.mark.parametrize("credits", [19.5, 22.5, 40.0])
def test_check_term_load_passes_a_realistic_load(credits: float):
    assert check_term_load(credits, "Winter") == []

# The fixture student's standing at the time of the winter run.
_WINTER_STANDING = Standing(total_points=5243.0, total_credits=62.5)  # gpa 83.888
_WINTER_FLOOR = 80.0
_WINTER_PLAN = [
    GradedCourse("00940314", 3.5, 10.571428571428571),
    GradedCourse("00940395", 1.5, -82.0),
    GradedCourse("00940396", 3.5, 10.571428571428571),
    GradedCourse("00940412", 4.0, 19.25),
    GradedCourse("00940594", 3.5, 10.571428571428571),
    GradedCourse("00950139", 2.5, -17.2),
]


def test_winter_run_negative_grades_are_flagged_out_of_range():
    violations = check_grades_in_range(_WINTER_PLAN)

    flagged = {v.message.split(":")[0] for v in violations}
    assert flagged == {"00940395", "00950139"}  # the two negative minimums, only those
    assert all(v.kind == "min_grade_range" for v in violations)


def test_winter_run_fails_the_joint_floor_at_gpa_65():
    violations = check_joint_floor(_WINTER_STANDING, _WINTER_PLAN, _WINTER_FLOOR)

    assert len(violations) == 1
    assert violations[0].kind == "joint_floor"
    assert "65.00" in violations[0].message  # (5243 + 22) / (62.5 + 18.5) = 65.0


def test_winter_run_fails_the_full_plan_check_on_both_counts():
    violations = check_plan(_WINTER_STANDING, _WINTER_PLAN, _WINTER_FLOOR)

    kinds = sorted({v.kind for v in violations})
    assert kinds == ["joint_floor", "min_grade_range"]


def test_a_single_course_at_its_isolated_minimum_lands_exactly_on_the_floor():
    # One 4-credit course at min_grade 19.25: (5243 + 77) / 66.5 == 80.0 exactly.
    # A plan of one course IS its own joint case, so the boundary must pass.
    plan = [GradedCourse("00940412", 4.0, 19.25)]

    assert check_joint_floor(_WINTER_STANDING, plan, _WINTER_FLOOR) == []
    assert check_plan(_WINTER_STANDING, plan, _WINTER_FLOOR) == []


def test_a_plan_with_comfortable_grades_passes_clean():
    plan = [
        GradedCourse("00940412", 4.0, 85.0),
        GradedCourse("00940314", 3.5, 90.0),
    ]

    assert check_plan(_WINTER_STANDING, plan, _WINTER_FLOOR) == []


def test_grade_above_100_is_flagged_infeasible():
    violations = check_grades_in_range([GradedCourse("00940412", 4.0, 105.0)])

    assert len(violations) == 1
    assert violations[0].kind == "min_grade_range"
    assert "exceeds 100" in violations[0].message


@pytest.mark.parametrize("gpa", [0.0, -3.0, 100.001, 838.88])
def test_a_gpa_outside_zero_to_hundred_is_flagged(gpa: float):
    violations = check_gpa_in_range(gpa)

    assert len(violations) == 1
    assert violations[0].kind == "gpa_range"


@pytest.mark.parametrize("gpa", [0.001, 83.888, 100.0])
def test_a_gpa_inside_range_is_clean(gpa: float):
    assert check_gpa_in_range(gpa) == []


def test_a_plan_with_no_completed_credits_is_unverifiable_not_a_violation():
    # No standing means no GPA exists to average against -- honest "can't check",
    # not a false alarm. (Guards the verifier against a divide-by-zero crash.)
    empty = Standing(total_points=0.0, total_credits=0.0)

    assert check_joint_floor(empty, [], _WINTER_FLOOR) == []
