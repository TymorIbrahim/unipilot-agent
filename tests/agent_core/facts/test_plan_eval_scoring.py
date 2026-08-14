"""Tests for the plan-eval scorer.

The records here are INLINE and shaped exactly like the saved
`agent_planning_eval/*.json` files -- the real ones are gitignored (fixture
student data), so the regression must carry its own copy of the shape it scores.
The winter record reproduces the 2026-07-20 run verbatim.
"""

from __future__ import annotations

from tests.agent_core.facts.plan_eval_scoring import (
    extract_courses,
    extract_floor,
    extract_standing,
    score_run,
)

# The real winter run, trimmed to the fields the scorer reads.
_WINTER_ANSWER = (
    "Your current GPA is 83.888, above your target. Here is your winter plan, with the "
    "minimum grade in each course that keeps your GPA above the floor:\n\n"
    "Winter -- 18.5 credits\n"
    "- number 00940314 · name a · type unclassified · credits 3.5 · min_grade 10.571428571428571\n"
    "- number 00940395 · name b · type unclassified · credits 1.5 · min_grade -82\n"
    "- number 00940396 · name c · type unclassified · credits 3.5 · min_grade 10.571428571428571\n"
    "- number 00940412 · name d · type unclassified · credits 4 · min_grade 19.25\n"
    "- number 00940594 · name e · type unclassified · credits 3.5 · min_grade 10.571428571428571\n"
    "- number 00950139 · name f · type unclassified · credits 2.5 · min_grade -17.2"
)
_WINTER_RECORD = {
    "question": "Plan my next winter semester ... to keep my overall GPA above 80.",
    "answer": _WINTER_ANSWER,
    "transcript": [
        {"turn": 5, "action": "call", "detail": "compute(...) -> completed_points=5243.0, total_credits=62.5, gpa=83.888"},
    ],
}


def test_extracts_all_six_winter_courses():
    courses = extract_courses(_WINTER_ANSWER)

    assert [c.code for c in courses] == ["00940314", "00940395", "00940396", "00940412", "00940594", "00950139"]
    assert courses[1].credits == 1.5 and courses[1].min_grade == -82.0


def test_extracts_floor_from_the_question():
    assert extract_floor(_WINTER_RECORD["question"]) == 80.0


def test_extracts_standing_from_a_transcript_detail():
    standing = extract_standing(_WINTER_RECORD)

    assert standing is not None
    assert standing.total_points == 5243.0 and standing.total_credits == 62.5


def test_winter_run_scores_as_a_failure_on_both_checks():
    scored = score_run(_WINTER_RECORD)

    assert not scored.passed
    kinds = sorted({v.kind for v in scored.violations})
    assert kinds == ["joint_floor", "min_grade_range"]
    assert not scored.skipped  # everything was recoverable, so nothing is unscored


def test_a_clean_run_passes():
    record = {
        "question": "... keep my overall GPA above 80.",
        "answer": (
            "Your current GPA is 83.888.\nWinter -- 7.5 credits\n"
            "- number 00940412 · credits 4 · min_grade 85\n"
            "- number 00940314 · credits 3.5 · min_grade 90"
        ),
        "transcript": [{"turn": 1, "detail": "-> completed_points=5243.0, total_credits=62.5, gpa=83.888"}],
    }

    scored = score_run(record)

    assert scored.passed
    assert not scored.violations and not scored.skipped


def test_a_refusal_with_no_plan_lines_is_unscored_not_passed():
    record = {"question": "... above 80.", "answer": "I couldn't work that out with confidence."}

    scored = score_run(record)

    assert not scored.passed  # nothing to score is NOT a pass
    assert scored.skipped and not scored.violations


def test_missing_standing_skips_only_the_joint_check():
    # A plan with in-range grades but no recoverable standing: the range check
    # still runs (and passes); the joint check is honestly skipped, not assumed.
    record = {
        "question": "... above 80.",
        "answer": "GPA is 83.888.\n- number 00940412 · credits 4 · min_grade 85",
        "transcript": [],
    }

    scored = score_run(record)

    assert not scored.violations
    assert any("joint-floor" in reason for reason in scored.skipped)
    assert not scored.passed  # a skipped check means not fully verified


def test_a_single_term_over_the_credit_ceiling_is_flagged():
    # The real 27-course / 83-credit winter: the raw optimize output scored as one
    # term. The scorer must flag the impossible term size.
    lines = "\n".join(f"- number 009400{i:02d} · credits 4 · min_grade 80" for i in range(12))
    record = {
        "question": "... keep my overall GPA above 80.",
        "answer": f"Your current GPA is 83.888.\nWinter -- 48 credits\n{lines}",
        "transcript": [{"turn": 5, "detail": "compute(...) -> credits=62.5, points=5243.0, gpa=83.888"}],
    }

    scored = score_run(record)

    assert any(v.kind == "term_load" for v in scored.violations)


def test_standing_from_bare_names_not_mistaken_for_plan_line_credits():
    # A live run named the standing `credits=62.5`/`points=5243.0` (not total_*),
    # and its rendered plan lines ALSO say "credits 4". The scorer must read the
    # standing from the `=` form, never the space-separated plan-line credits --
    # grabbing 4 would replay against a phantom baseline.
    record = {
        "question": "... keep my overall GPA above 80.",
        "answer": (
            "Your current GPA is 83.888.\nWinter -- 7.5 credits\n"
            "- number 00940412 · credits 4 · min_grade 0\n"
            "- number 00940314 · credits 3.5 · min_grade 0"
        ),
        "transcript": [{"turn": 5, "detail": "compute(...) -> credits=62.5, points=5243.0, gpa=83.888"}],
    }

    standing = extract_standing(record)
    assert standing is not None
    assert (standing.total_points, standing.total_credits) == (5243.0, 62.5)

    scored = score_run(record)
    # Both courses at 0: (5243 + 0) / (62.5 + 7.5) = 74.9 < 80 -> joint failure.
    assert [v.kind for v in scored.violations] == ["joint_floor"]
