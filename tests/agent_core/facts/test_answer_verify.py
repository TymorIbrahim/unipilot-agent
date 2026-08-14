"""Unit tests for the loop's verify step.

`verify_answer` reads the TYPED facts an answer was built from -- the plan comes
from the Collection behind a `:detail` slot, the standing from scalar facts -- so
these build those facts directly rather than parsing prose. The winter plan is
the real 2026-07-20 run's numbers.
"""

from __future__ import annotations

from app.agent_core.facts.answer import Answer, HeldFact
from app.agent_core.facts.answer_verify import verify_answer
from app.agent_core.facts.types import (
    Basis,
    Collection,
    Completeness,
    Record,
    Scalar,
    ScalarKind,
)

Q = ScalarKind.QUANTITY
I = ScalarKind.IDENTIFIER
_ABOVE_80 = "Plan my winter to keep my overall GPA above 80."


def _course(code: str, credits: float, grade: float) -> Record:
    return Record(
        fields={"number": Scalar(I, code), "credits": Scalar(Q, credits), "min_grade": Scalar(Q, grade)},
        basis=Basis.OFFICIAL_RECORD,
    )


def _plan(*courses: Record) -> Collection:
    return Collection(records=tuple(courses), completeness=Completeness(complete=True, total=len(courses)))


def _facts(**kv: object) -> dict[str, HeldFact]:
    return {name: HeldFact(value=value, basis=Basis.OFFICIAL_RECORD) for name, value in kv.items()}


def _answer(*used: str) -> Answer:
    return Answer(text="(rendered plan)", basis=Basis.OFFICIAL_RECORD, used=tuple(used), citations=())


_WINTER = _plan(
    _course("00940314", 3.5, 10.571428571428571),
    _course("00940395", 1.5, -82.0),
    _course("00940396", 3.5, 10.571428571428571),
    _course("00940412", 4.0, 19.25),
    _course("00940594", 3.5, 10.571428571428571),
    _course("00950139", 2.5, -17.2),
)


def test_the_real_winter_plan_is_rejected_on_range_and_joint_floor():
    facts = _facts(winter=_WINTER, total_points=Scalar(Q, 5243.0), total_credits=Scalar(Q, 62.5))

    problems = verify_answer(_answer("winter", "total_points", "total_credits"), facts, _ABOVE_80)

    kinds = sorted({p.kind for p in problems})
    assert kinds == ["joint_floor", "min_grade_range"]
    assert sum(p.kind == "min_grade_range" for p in problems) == 2  # the two negative grades


def test_a_sound_plan_passes():
    # One 4-credit course at 19.25: joint GPA lands exactly on the 80 floor.
    facts = _facts(
        winter=_plan(_course("00940412", 4.0, 19.25)),
        total_points=Scalar(Q, 5243.0),
        total_credits=Scalar(Q, 62.5),
    )

    assert verify_answer(_answer("winter", "total_points", "total_credits"), facts, _ABOVE_80) == []


def test_a_non_plan_answer_is_a_no_op():
    # A collection with no min_grade field is not a plan -- the verifier ignores it,
    # so ordinary answers are never touched.
    offerings = Collection(
        records=(Record(fields={"number": Scalar(I, "00940412"), "credits": Scalar(Q, 4.0)}, basis=Basis.OFFICIAL_RECORD),),
        completeness=Completeness(complete=True, total=1),
    )
    facts = _facts(offerings=offerings)

    assert verify_answer(_answer("offerings"), facts, _ABOVE_80) == []


def test_winter_and_spring_are_judged_as_one_plan():
    # Two collections, each sound alone, but jointly they must still hold.
    winter = _plan(_course("00940412", 4.0, 19.25))   # holds the floor alone
    spring = _plan(_course("00940314", 3.5, 19.25))   # also low; together they overshoot down
    facts = _facts(winter=winter, spring=spring, total_points=Scalar(Q, 5243.0), total_credits=Scalar(Q, 62.5))

    problems = verify_answer(_answer("winter", "spring", "total_points", "total_credits"), facts, "GPA above 80")

    # (5243 + 77 + 67.375) / (62.5 + 7.5) = 5387.375/70 = 76.96 < 80 -> joint failure.
    assert [p.kind for p in problems] == ["joint_floor"]


def test_without_standing_only_the_grade_range_is_checked():
    # No total_points/total_credits: the joint check is skipped (not guessed), but
    # an impossible grade is still caught from the plan alone.
    facts = _facts(winter=_WINTER)

    problems = verify_answer(_answer("winter"), facts, _ABOVE_80)

    assert {p.kind for p in problems} == {"min_grade_range"}


def test_standing_is_recovered_from_points_credits_names_not_only_total():
    # A live run named the standing `points`/`credits`, not total_*. `_standing`
    # found neither, silently skipped the joint check, and shipped a GPA-65 plan.
    facts = _facts(
        winter=_plan(_course("00940412", 4.0, 19.25), _course("00940395", 1.5, 0.0)),
        points=Scalar(Q, 5243.0),
        credits=Scalar(Q, 62.5),
        gpa=Scalar(Q, 83.888),
    )

    # 4cr@19.25 holds the floor alone; adding 1.5cr@0 drops the joint GPA to 78.2.
    problems = verify_answer(_answer("winter", "points", "credits", "gpa"), facts, _ABOVE_80)

    assert [p.kind for p in problems] == ["joint_floor"]


def test_standing_cross_fills_points_from_gpa_and_credits():
    # Only credits and gpa are held -- points is derived (83.888 * 62.5 = 5243).
    facts = _facts(
        winter=_plan(_course("00940412", 4.0, 19.25), _course("00940395", 1.5, 0.0)),
        credits=Scalar(Q, 62.5),
        gpa=Scalar(Q, 83.888),
    )

    problems = verify_answer(_answer("winter", "credits", "gpa"), facts, _ABOVE_80)

    assert [p.kind for p in problems] == ["joint_floor"]


def test_a_mis_named_credits_fact_is_rejected_by_the_gpa_cross_check():
    # `credits` here is 3.5 (a course's credits), not the standing. gpa disagrees
    # with points/3.5, so the standing is NOT trusted -- the joint check skips
    # rather than replaying against a phantom baseline and returning a false verdict.
    facts = _facts(
        winter=_plan(_course("00940412", 4.0, 19.25)),
        points=Scalar(Q, 5243.0),
        credits=Scalar(Q, 3.5),
        gpa=Scalar(Q, 83.888),
    )

    problems = verify_answer(_answer("winter", "points", "credits", "gpa"), facts, _ABOVE_80)

    assert problems == []  # unverifiable standing -> skipped, never a false alarm


def test_a_term_stuffed_with_the_unscheduled_overflow_is_flagged():
    # 12 courses / 48 credits in one term is the raw optimize output (placed rows
    # AND the "(unscheduled)" overflow) scored as if all placed. Grades hold the
    # floor, so only the term-load check should fire.
    big = _plan(*[_course(f"009400{i:02d}", 4.0, 80.0) for i in range(12)])
    facts = _facts(winter=big, points=Scalar(Q, 5243.0), credits=Scalar(Q, 62.5), gpa=Scalar(Q, 83.888))

    problems = verify_answer(_answer("winter", "points", "credits", "gpa"), facts, _ABOVE_80)

    assert [p.kind for p in problems] == ["term_load"]


def test_a_realistic_winter_load_is_not_flagged_for_term_size():
    plan = _plan(_course("00940412", 4.0, 80.0), _course("00940314", 3.5, 80.0))  # 7.5 credits
    facts = _facts(winter=plan, points=Scalar(Q, 5243.0), credits=Scalar(Q, 62.5), gpa=Scalar(Q, 83.888))

    problems = verify_answer(_answer("winter", "points", "credits", "gpa"), facts, _ABOVE_80)

    assert not any(p.kind == "term_load" for p in problems)


def test_a_gpa_over_100_is_flagged():
    facts = _facts(
        winter=_plan(_course("00940412", 4.0, 85.0)),
        total_points=Scalar(Q, 7000.0),  # 7000 / 62.5 = 112 -> impossible ratio
        total_credits=Scalar(Q, 62.5),
    )

    problems = verify_answer(_answer("winter", "total_points", "total_credits"), facts, _ABOVE_80)

    assert any(p.kind == "gpa_range" for p in problems)
