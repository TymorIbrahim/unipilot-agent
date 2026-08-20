"""A failed course earns no credit, and the model must not have to know that.

The live failure: asked "how many credits have I completed", the agent summed
`completed_courses.creditsEarned` and answered 135. The correct answer is 129.5
-- course 01040166 is graded 30, below the pass mark of 55, and still carries its
full 5.5 credits in that column.

Nothing in the fact layer could have caught it. The number was real, official,
non-empty, and read from the right table; every gate passed and should have. So
the rule is enforced in the DATABASE (a generated column, see db/schema.sql) and
the trap is named where the model picks the column.
"""

from __future__ import annotations

from app.agent_core.facts.dispatch import DispatchContext
from app.agent_core.facts.find import declared_paths
from app.agent_core.facts.loop import render_sources
from app.agent_core.facts.sources import COMPLETED_COURSES, REGISTRY


class TestTheTranscriptDeclaresCountedCredits:
    def test_credits_counted_is_declared(self) -> None:
        assert "creditsCounted" in COMPLETED_COURSES.fields

    def test_passed_is_declared(self) -> None:
        """Without it, "which courses must I re-take" has no route at all."""
        assert "passed" in COMPLETED_COURSES.fields

    def test_the_raw_column_survives(self) -> None:
        """`creditsEarned` is still the registrar's record and stays readable --
        the fix is to make the RIGHT field reachable, not to hide the other."""
        assert "creditsEarned" in COMPLETED_COURSES.fields


class TestTheTrapIsNamedWhereTheColumnIsChosen:
    def test_every_noted_field_is_a_real_field(self) -> None:
        """A note on a field that does not exist is worse than no note: it sends
        the model after a column the predicate compiler will reject.

        Checked against `declared_paths`, not `schema.fields`. The latter holds
        only TOP-LEVEL names, so it called `semesters.goalCredits` undeclared --
        a path that is real, is filterable, and is listed to the model by
        `render_sources` through this very function. Judging notes by a narrower
        set than the prompt renders would ban notes on exactly the nested fields
        that most need one."""
        for name, schema in REGISTRY.items():
            unknown = set(getattr(schema, "field_notes", {})) - set(declared_paths(schema))
            assert not unknown, f"{name} notes undeclared fields: {sorted(unknown)}"

    def test_a_note_on_an_invented_field_is_still_caught(self) -> None:
        """The guard must still do its job for a genuinely wrong name."""
        from dataclasses import replace

        schema = replace(REGISTRY["student_profiles"], field_notes={"noSuchColumn": "..."})
        assert set(schema.field_notes) - set(declared_paths(schema)) == {"noSuchColumn"}

    def test_the_notes_reach_the_prompt(self) -> None:
        rendered = render_sources(DispatchContext(schemas=REGISTRY))
        assert "creditsCounted" in rendered
        assert "SUM THIS" in rendered, "the steer must survive into what the model reads"

    def test_the_prompt_warns_that_earned_includes_failures(self) -> None:
        rendered = render_sources(DispatchContext(schemas=REGISTRY))
        note = next(
            line for line in rendered.splitlines() if line.strip().startswith("! creditsEarned")
        )
        assert "fail" in note.lower() or "FAILED" in note

    def test_a_source_without_notes_renders_unchanged(self) -> None:
        """The mechanism must cost nothing for the sources that do not use it."""
        rendered = render_sources(DispatchContext(schemas={"courses": REGISTRY["courses"]}))
        assert "!" not in rendered
