"""The fourth derivation moved out of the reasoning loop.

Three live planning runs, traced from `/api/execute`, all had one shape:

    find x3 -> compute x2-4 -> plan_term -> compute x1-2 -> answer
    └──────── deterministic: one right answer, no judgement ────────┘

At ~15s a turn that is 60-90s of plumbing around a single turn of real planning,
against a platform ceiling of 60s. The planning question could not fit in one
request, and no prompt makes a deterministic join cheaper than doing it in SQL.

So `remaining_courses` joins the curriculum to the transcript once: every course
in the student's track they have not passed, carrying title, credits and
category. It follows `passed_courses`, `track_courses.category` and
`prerequisite_edges` -- each of which removed turns the same way.

A SOURCE, not a composite. It decides nothing and composes with `select`,
`group`, `plan_term` and `optimize` exactly as `passed_courses` does. The
pre-solved `generate_semester_plan(student, track)` that `optimize.py` warns
against answers the question; this one supplies facts.
"""

from __future__ import annotations

from app.agent_core.facts.find import declared_paths
from app.agent_core.facts.sources import REGISTRY, REMAINING_COURSES


class TestItIsDeclaredWhereTheModelCanSeeIt:
    def test_it_is_in_the_registry(self) -> None:
        assert REGISTRY["remaining_courses"] is REMAINING_COURSES

    def test_it_carries_what_planning_needs_without_a_join(self) -> None:
        """The point is that no follow-up fetch is required: a planner needs the
        code, its credits and whether it is compulsory."""
        paths = set(declared_paths(REMAINING_COURSES))
        assert {"courseNumber", "credits", "category", "userId"} <= paths

    def test_it_is_filterable_by_student(self) -> None:
        assert "userId" in REMAINING_COURSES.fields

    def test_it_is_an_official_record(self) -> None:
        from app.agent_core.facts.types import Basis

        assert REMAINING_COURSES.basis is Basis.OFFICIAL_RECORD


class TestTheNotesSteerAwayFromTheOldRoute:
    def test_the_note_says_to_start_here(self) -> None:
        note = REMAINING_COURSES.field_notes["courseNumber"]
        assert "START PLANNING QUESTIONS HERE" in note

    def test_the_credits_note_repeats_the_two_meanings_of_remaining(self) -> None:
        """`remaining` means two things and the confusion has now cost three
        separate defects, so the warning travels with the column."""
        note = REMAINING_COURSES.field_notes["credits"]
        assert "50.0" in note and "25.5" in note
        assert "totalCredits" in note

    def test_the_category_note_says_how_to_bound_the_set(self) -> None:
        note = REMAINING_COURSES.field_notes["category"]
        assert "mandatory" in note and "electives only" in note

    def test_every_noted_field_is_real(self) -> None:
        assert set(REMAINING_COURSES.field_notes) <= set(declared_paths(REMAINING_COURSES))


class TestTheRecipeUsesIt:
    def test_step_one_fetches_it(self) -> None:
        from app.agent_core.facts.adapter import SYSTEM_PROMPT

        assert "1. find remaining_courses" in SYSTEM_PROMPT

    def test_the_old_multi_turn_opening_is_gone(self) -> None:
        """A recipe that still spells out the six-turn route is what the model
        follows, however good the new source is -- measured twice already."""
        from app.agent_core.facts.adapter import SYSTEM_PROMPT

        assert "remaining = (step-3 courses) difference" not in SYSTEM_PROMPT

    def test_the_recipe_steps_are_numbered_in_order(self) -> None:
        """Renumbering after collapsing five steps into one is where a
        cross-reference goes stale and points at a step that no longer exists."""
        import re

        from app.agent_core.facts.adapter import SYSTEM_PROMPT

        recipe = SYSTEM_PROMPT.split("RECIPE --", 1)[1].split("CHECKPOINT before", 1)[0]
        steps = [int(n) for n in re.findall(r"^  (\d+)\. ", recipe, re.M)]
        assert steps == sorted(steps) and steps == list(range(1, len(steps) + 1))
        for stale in ("step 5", "step 6", "step 7", "step 8"):
            assert stale not in SYSTEM_PROMPT, f"{stale} no longer exists"
