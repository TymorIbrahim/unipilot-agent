"""The profile row is read before the loop starts. It should be handed over.

`_profile_of` runs on every request -- to prove the student exists, and to pick
the right rulebook for retrieval. It was selecting two of its columns and
discarding the rest, and every planning run then opened:

    turn 1  find    -> profile          re-fetching the row we already held
    turn 2  compute -> program_slug     unpacking it
    turn 3  ...the actual work begins

Two model calls per planning question to re-derive something the server had in
hand. Widening the select costs no extra round trip; seeding the results deletes
both turns. Measured on "Plan my winter semester.": 10-11 steps became 6-9, and
126-133s became 89-122s.

What is NOT seeded matters as much. Only stable identity fields come through --
anything an ANSWER depends on (credits, grades, courses) stays behind a tool
call, so it arrives carrying a basis and a completeness the loop can reason
about, rather than appearing from nowhere.
"""

from __future__ import annotations

import pytest

from app.agent_core.facts.service import _SEEDED_PROFILE_FIELDS, _seed_profile_facts
from app.agent_core.facts.types import Basis, ScalarKind


class _Row(dict):
    """Stands in for an asyncpg Record, which is read by key and knows its keys."""


class _Context:
    def __init__(self) -> None:
        self.facts: dict = {}


FULL = _Row(
    userId="6a578a2da43a2cfe1bcc791c",
    programType="bsc",
    programSlug="track-information-systems-engineering",
    catalogYear=2025,
    currentSemesterCode="2025-2",
    maxCreditsPerSemester=18.0,
)


class TestWhatIsSeeded:
    def test_the_program_slug_arrives_as_a_fact(self) -> None:
        context = _Context()
        _seed_profile_facts(context, FULL)
        assert context.facts["program_slug"].value.value == "track-information-systems-engineering"

    def test_numbers_arrive_typed_as_quantities(self) -> None:
        context = _Context()
        _seed_profile_facts(context, FULL)
        assert context.facts["catalog_year"].value.kind is ScalarKind.QUANTITY
        assert context.facts["max_credits_per_semester"].value.value == 18.0

    def test_every_seeded_fact_carries_the_registrar_basis(self) -> None:
        context = _Context()
        _seed_profile_facts(context, FULL)
        assert all(f.basis is Basis.OFFICIAL_RECORD for f in context.facts.values())

    def test_the_query_selects_everything_it_seeds(self) -> None:
        """The seeding and the SELECT drift apart silently -- a column missing
        from the query is simply never seeded, and the turns come back."""
        import inspect

        from app.agent_core.facts import service

        sql = inspect.getsource(service._profile_of)
        for column, _, _ in _SEEDED_PROFILE_FIELDS:
            assert f'"{column}"' in sql, f"{column} is seeded but not selected"


class TestAbsenceBeatsInvention:
    def test_a_null_column_seeds_nothing(self) -> None:
        context = _Context()
        _seed_profile_facts(context, _Row(userId="x", programSlug=None, catalogYear=None))
        assert "program_slug" not in context.facts

    def test_an_empty_string_seeds_nothing(self) -> None:
        context = _Context()
        _seed_profile_facts(context, _Row(userId="x", programSlug=""))
        assert "program_slug" not in context.facts

    def test_a_missing_column_does_not_raise(self) -> None:
        """A profile row from an older schema must not end the run."""
        context = _Context()
        _seed_profile_facts(context, _Row(userId="x"))
        assert context.facts == {}


class TestNothingAnswerBearingIsSeeded:
    """An allowlist, not a keyword filter.

    The first version rejected any fact name containing "credits" and tripped on
    `max_credits_per_semester` -- which is a POLICY CAP from the profile, not
    something derived from the student's record. The distinction is the point:
    what may be seeded is a stable fact about who the student is, and what may
    not is anything about how far they have got.

    Spelled out, so adding a field is a deliberate edit here rather than a line
    that slips into the tuple.
    """

    ALLOWED = {
        "program_slug",             # which degree -- identity
        "catalog_year",             # which rulebook -- identity
        "current_semester",         # where in the calendar -- identity
        "max_credits_per_semester", # a cap the registrar sets, not credits earned
    }

    def test_the_seeded_set_is_exactly_the_allowlist(self) -> None:
        assert {name for _, name, _ in _SEEDED_PROFILE_FIELDS} == self.ALLOWED

    @pytest.mark.parametrize(
        "column",
        ["grade", "gradePoints", "creditsEarned", "creditsCounted", "passed", "courseId"],
    )
    def test_no_transcript_column_is_seeded(self, column: str) -> None:
        """Progress must arrive through a tool call, carrying a basis and a
        completeness. A credit total that appeared from nowhere is exactly the
        ungrounded number this layer exists to refuse."""
        assert column not in {col for col, _, _ in _SEEDED_PROFILE_FIELDS}
