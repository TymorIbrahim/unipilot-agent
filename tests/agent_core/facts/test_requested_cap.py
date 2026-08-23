"""A credit limit the STUDENT named outranks the one on their profile.

Asked "I'm working part-time next semester, so keep it under 10 credits", the
agent returned a 16-credit term -- three runs out of three on the Hebrew
phrasing. Nothing caught it: the profile cap is 18, so `check_term_within_cap`
passed it, and `check_term_load`'s 40-credit ceiling passed it too. No part of
the system had any idea that 10 had been said.

`plan_term` can enforce this -- it takes `max_credits` -- and the prompt asks
the model to pass it "if the request names a different limit". That is a
judgement made once per run, and it was made wrong every time.

The half that must not break is the false-positive half: the questions this
runs against are full of bare numbers meaning something else ("finish by summer
2027", "starting from 2025-2"), and inventing a cap refuses a correct plan.
"""

from __future__ import annotations

import pytest

from app.agent_core.facts.answer_verify import _requested_cap
from app.agent_core.facts.postconditions import check_term_within_requested_cap


class TestReadingTheCapOutOfTheQuestion:
    @pytest.mark.parametrize(
        "question,expected",
        [
            # The live pair, both languages.
            ("אני עובד במשרה חלקית בסמסטר הבא, אז תשאיר לי מתחת ל-10 נקודות. מה כדאי לקחת?", 10),
            ("I'm working part-time next semester, so keep it under 10 credits. "
             "What should I take?", 10),
            ("Plan my term with no more than 12 credits", 12),
            ("תכנן לי סמסטר עם עד 8 נקודות", 8),
            ("plan a light term, at most 6 credits please", 6),
            ("תכנן סמסטר של לכל היותר 7 נקודות", 7),
        ],
    )
    def test_a_stated_limit_is_found(self, question: str, expected: float) -> None:
        assert _requested_cap(question) == expected

    def test_the_tightest_of_several_wins(self) -> None:
        """Honouring the looser of two stated limits is as wrong as neither."""
        assert _requested_cap("keep it under 10 credits, no more than 12 credits") == 10

    @pytest.mark.parametrize(
        "question",
        [
            # Bare numbers that are years, semesters, or nothing to do with a cap.
            "I want to finish by summer 2027. Starting from 2025-2, is that realistic?",
            "אני רוצה לסיים עד קיץ 2027. מתחיל מ-2025-2, זה ריאלי?",
            # A limit word and a credit noun, but no figure -- reading one out of
            # this would invent a constraint the student never set.
            "What is the maximum number of credits I am allowed to take in one semester?",
            "מה מספר הנקודות המקסימלי שמותר לי לקחת בסמסטר אחד?",
            "Plan my next two semesters, and put the heavier one first.",
            "כמה סמסטרים נשארו לי עד סיום התואר?",
            "Which three courses did I do worst in?",
        ],
    )
    def test_no_cap_is_invented(self, question: str) -> None:
        assert _requested_cap(question) is None


class TestHoldingThePlanToIt:
    def test_the_live_failure_is_caught(self) -> None:
        """16 credits against a requested 10."""
        violations = check_term_within_requested_cap(16.0, 10.0, "winter")
        assert [v.kind for v in violations] == ["term_over_requested_cap"]

    def test_the_message_names_the_move_that_fixes_it(self) -> None:
        """A reason the model cannot act on wastes the retry it costs."""
        message = check_term_within_requested_cap(16.0, 10.0, "winter")[0].message
        assert "max_credits=10" in message
        assert "plan_term" in message

    def test_a_plan_that_fits_passes(self) -> None:
        assert check_term_within_requested_cap(9.5, 10.0, "winter") == []

    def test_exactly_the_cap_passes(self) -> None:
        assert check_term_within_requested_cap(10.0, 10.0, "winter") == []

    def test_no_stated_cap_means_no_opinion(self) -> None:
        assert check_term_within_requested_cap(16.0, 0.0, "winter") == []


class TestItIsNotTheProfileCapCheck:
    """The two hold different limits and both must be able to fire.

    The profile cap is what the student can normally carry; the requested cap is
    what they have just said they can carry this term. A 16-credit term is legal
    under an 18-credit profile and illegal under a requested 10, which is
    exactly the case that shipped.
    """

    def test_a_term_legal_on_profile_can_still_break_the_request(self) -> None:
        from app.agent_core.facts.postconditions import check_term_within_cap

        assert check_term_within_cap(16.0, 18.0, "winter") == []
        assert check_term_within_requested_cap(16.0, 10.0, "winter") != []
