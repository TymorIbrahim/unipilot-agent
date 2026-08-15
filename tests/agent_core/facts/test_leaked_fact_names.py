"""A fact NAME written as prose is a slot the model forgot to brace.

Live, from a registration request:

    "I've prepared a request to register target_course_number."

The student is shown a variable where a course code belongs. Every other check
passes: the name is not a number, so the grounding invariant has nothing to say,
and the answer's real slots all resolved.

Only names carrying an underscore are considered. A fact called `credits`
legitimately appears in "you have 129.5 credits"; `target_course_number` is not
something anyone writes by accident.
"""

from __future__ import annotations

from app.agent_core.facts.answer import HeldFact, resolve_answer
from app.agent_core.facts.types import Basis, Scalar, ScalarKind


def _facts() -> dict:
    return {
        "target_course_number": HeldFact(
            value=Scalar(ScalarKind.IDENTIFIER, "00960211"), basis=Basis.OFFICIAL_RECORD
        ),
        "credits": HeldFact(
            value=Scalar(ScalarKind.QUANTITY, 129.5), basis=Basis.OFFICIAL_RECORD
        ),
    }


class TestALeakedNameIsRefused:
    def test_the_live_failure_is_caught(self) -> None:
        verdict = resolve_answer(
            "I've prepared a request to register target_course_number.", _facts(), "q"
        )
        assert not hasattr(verdict, "text")
        assert "target_course_number" in verdict.reason

    def test_the_message_says_how_to_fix_it(self) -> None:
        verdict = resolve_answer("register target_course_number now", _facts(), "q")
        assert "braces" in verdict.reason


class TestRealAnswersAreUnaffected:
    def test_the_braced_form_renders(self) -> None:
        verdict = resolve_answer(
            "I've prepared a request to register {target_course_number}.", _facts(), "q"
        )
        assert verdict.text == "I've prepared a request to register 00960211."

    def test_an_ordinary_word_that_is_also_a_fact_name_is_fine(self) -> None:
        """`credits` is a real English word and a fact name. Flagging it would
        break every answer that mentions credits."""
        verdict = resolve_answer("You have completed {credits} credits.", _facts(), "q")
        assert verdict.text == "You have completed 129.5 credits."

    def test_a_similar_but_different_word_is_not_flagged(self) -> None:
        verdict = resolve_answer(
            "You have {credits} credits in target_course_numbers overall.", _facts(), "q"
        )
        assert hasattr(verdict, "text"), "a longer word merely containing the name is not the name"
