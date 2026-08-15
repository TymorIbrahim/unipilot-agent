"""A near-miss slot name should be repairable, not fatal.

A live run derived `english_requirement`, wrote
`{english_requirement_phrase}` in its answer, and was refused. The refusal
already listed every held fact and the model still did not recover: a list of a
dozen names does not point at the one that is a suffix away, and the run ended
with no answer at all for what was a typo.

Suggesting is as far as this goes. Nothing is renamed or auto-corrected -- a slot
quietly resolved to a fact the model did not mean is exactly the grounding
failure the layer exists to prevent, and it would be invisible in the answer.
"""

from __future__ import annotations

from app.agent_core.facts.answer import HeldFact, resolve_answer
from app.agent_core.facts.types import Basis, Scalar, ScalarKind


def _facts() -> dict:
    return {
        "english_requirement": HeldFact(
            value=Scalar(ScalarKind.TEXT, "2 English courses"), basis=Basis.WIKI_DERIVED
        ),
        "completed_credits": HeldFact(
            value=Scalar(ScalarKind.QUANTITY, 129.5), basis=Basis.OFFICIAL_RECORD
        ),
    }


class TestTheNearMissIsNamed:
    def test_the_live_typo_gets_a_suggestion(self) -> None:
        verdict = resolve_answer("To graduate you need {english_requirement_phrase}.", _facts(), "q")
        assert "Did you mean 'english_requirement'?" in verdict.reason

    def test_a_dropped_suffix_gets_a_suggestion(self) -> None:
        verdict = resolve_answer("You have {completed_credit}.", _facts(), "q")
        assert "completed_credits" in verdict.reason

    def test_the_full_list_is_still_given(self) -> None:
        """The suggestion narrows; it does not replace the ground truth of what
        is actually held."""
        verdict = resolve_answer("You need {english_requirement_phrase}.", _facts(), "q")
        assert "Available:" in verdict.reason
        assert "completed_credits" in verdict.reason


class TestItNeverGuessesWildly:
    def test_an_unrelated_name_gets_no_suggestion(self) -> None:
        verdict = resolve_answer("You have {totally_unrelated_thing}.", _facts(), "q")
        assert "Did you mean" not in verdict.reason
        assert "Available:" in verdict.reason

    def test_nothing_is_auto_corrected(self) -> None:
        """The answer is still REFUSED. A slot silently resolved to a fact the
        model did not name would be the failure this layer exists to prevent."""
        verdict = resolve_answer("You need {english_requirement_phrase}.", _facts(), "q")
        assert not hasattr(verdict, "text"), "a near miss must not be accepted"

    def test_a_correct_slot_is_unaffected(self) -> None:
        verdict = resolve_answer("You need {english_requirement}.", _facts(), "q")
        assert verdict.text == "You need 2 English courses."
