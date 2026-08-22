"""A personal setting presented as institutional policy.

Asked "What is the maximum number of credits I am allowed to take in one
semester?", the deployed agent answered:

    Your maximum allowed load per semester is 18 credits.

in 2.4 seconds, having consulted no corpus at all. It read
`student_profiles.maxCreditsPerSemester` and reported it as the rule.

The undergraduate regulations, section 5.1, set the maximum at 29 credits
without special approval, with the faculty head's recommendation and the Dean's
approval required above that. 18 is what THIS student's record says, which is a
different fact answering a different question.

Both are true and neither substitutes for the other: 18 is what a PLAN for them
may contain, 29 is what they are ALLOWED. The distinction now travels with the
column, in the same voice as the other notes that hold -- the rule, the
measurement, and what to do instead.
"""

from __future__ import annotations

from app.agent_core.facts.dispatch import DispatchContext
from app.agent_core.facts.loop import render_sources
from app.agent_core.facts.sources import REGISTRY, STUDENT_PROFILES


class TestTheNoteSeparatesTheTwo:
    def test_it_says_the_column_is_not_the_rule(self) -> None:
        note = STUDENT_PROFILES.field_notes["maxCreditsPerSemester"]
        assert "IT IS NOT THE RULE" in note

    def test_it_gives_the_regulation_figure(self) -> None:
        note = STUDENT_PROFILES.field_notes["maxCreditsPerSemester"]
        assert "29" in note

    def test_it_records_the_answer_that_was_wrong(self) -> None:
        note = STUDENT_PROFILES.field_notes["maxCreditsPerSemester"]
        assert "18 credits" in note

    def test_it_says_which_route_answers_which_question(self) -> None:
        note = STUDENT_PROFILES.field_notes["maxCreditsPerSemester"]
        assert "search_corpus" in note
        assert "ALLOWED" in note and "PLAN" in note


class TestThePlanningRuleSurvives:
    def test_the_cap_still_governs_a_plan(self) -> None:
        """The column is still the only cap a plan is checked against -- the
        23-credit winter it caught must stay caught."""
        note = STUDENT_PROFILES.field_notes["maxCreditsPerSemester"]
        assert "a plan over it is refused" in note

    def test_the_ceiling_derivation_survives(self) -> None:
        note = STUDENT_PROFILES.field_notes["maxCreditsPerSemester"]
        assert "ceil_div" in note

    def test_it_reaches_the_prompt(self) -> None:
        rendered = render_sources(DispatchContext(schemas=REGISTRY))
        assert "IT IS NOT THE RULE" in rendered


class TestTheWarningReachesTheTurnPromptToo:
    """The field note alone did not change the answer, and the reason matters.

    The model never runs a `find` on `student_profiles` -- the cap is SEEDED, so
    it is already in "FACTS YOU HOLD" before turn one. It answers from there and
    never reads the source list where the note lives. Re-measured after adding
    the note: still "Your per-semester maximum is 18 credits", now in one step
    and zero tool calls.

    Seeding the facts is what bypassed the note, so the warning has to sit
    beside the seeded facts.
    """

    def _prompt(self) -> str:
        from app.agent_core.facts.answer import HeldFact
        from app.agent_core.facts.dispatch import DispatchContext
        from app.agent_core.facts.loop import _prompt
        from app.agent_core.facts.types import Basis, Scalar, ScalarKind

        context = DispatchContext(facts={
            "me": HeldFact(value=Scalar(ScalarKind.IDENTIFIER, "6a57"),
                           basis=Basis.OFFICIAL_RECORD),
            "max_credits_per_semester": HeldFact(
                value=Scalar(ScalarKind.QUANTITY, 18.0), basis=Basis.OFFICIAL_RECORD),
        })
        return _prompt("how many credits am I allowed?", context, [])

    def test_the_turn_prompt_says_the_seeded_facts_are_not_the_rules(self) -> None:
        assert "not the rules" in self._prompt()

    def test_it_names_the_route_to_the_rules(self) -> None:
        assert "search_corpus" in self._prompt()

    def test_it_carries_the_measurement(self) -> None:
        prompt = self._prompt()
        assert "18" in prompt and "29" in prompt

    def test_the_seeded_facts_are_still_described_as_already_held(self) -> None:
        """The warning must not crowd out the reason the block exists."""
        assert "ALREADY YOURS" in self._prompt()
