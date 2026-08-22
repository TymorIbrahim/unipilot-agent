"""A course offered zero times cannot be found by counting offerings.

Asked which of their remaining mandatory courses is offered least often, the
deployed agent answered "00940704 and 01040065, each offered 2 times". By SQL:

    00960221 offered 0x     <- the actual answer
    00940704 offered 2x
    01040065 offered 2x
    ...
    00940412 offered 7x

A course with no row in `course_offerings` produces no group, so `group` drops
it and a ranking built on the result loses precisely the rarest courses -- the
ones the question is about.

The algebra cannot express its way out of this directly: `join` is INNER, so an
unmatched left record is discarded, and a left join would not help either
because the unmatched row would then count as one rather than zero. What IS
expressible is a `difference` of the candidate list against the grouped result,
and that is what the note tells the model to do.

The same shape as `academicYear`'s note beside it -- 0 rows meaning "not
recorded" rather than "does not happen" -- which is the mistake this domain
makes most.
"""

from __future__ import annotations

from app.agent_core.facts.find import declared_paths
from app.agent_core.facts.sources import COURSE_OFFERINGS


class TestTheNoteExists:
    def test_it_is_on_the_field_the_grouping_uses(self) -> None:
        assert "courseNumber" in COURSE_OFFERINGS.field_notes

    def test_it_names_the_route_that_finds_the_zeroes(self) -> None:
        note = COURSE_OFFERINGS.field_notes["courseNumber"]
        assert "difference" in note
        assert "zero offerings" in note

    def test_it_records_the_answer_that_was_wrong(self) -> None:
        """A note saying only what to do gets read and not acted on; the ones
        that hold in this codebase carry the measurement."""
        note = COURSE_OFFERINGS.field_notes["courseNumber"]
        assert "00960221" in note and "00940704" in note

    def test_every_noted_field_is_real(self) -> None:
        assert set(COURSE_OFFERINGS.field_notes) <= set(declared_paths(COURSE_OFFERINGS))

    def test_it_reaches_the_prompt(self) -> None:
        from app.agent_core.facts.dispatch import DispatchContext
        from app.agent_core.facts.loop import render_sources
        from app.agent_core.facts.sources import REGISTRY

        rendered = render_sources(DispatchContext(schemas=REGISTRY))
        assert "OMITS EVERY COURSE THAT HAS NONE" in rendered


class TestTheSiblingRuleIsStillThere:
    def test_future_years_are_still_silence_not_evidence(self) -> None:
        """The two notes are the same mistake in two directions and must not
        crowd each other out."""
        note = COURSE_OFFERINGS.field_notes["academicYear"]
        assert "silence, not" in note
