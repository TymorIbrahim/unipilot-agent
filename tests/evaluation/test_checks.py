"""Every wrong verdict this scorer has produced, pinned as a test.

Five correct answers were scored as failures in one session, each by a
hand-rolled regex in a throwaway probe. A scorer wrong in the pessimistic
direction is not the safe kind of wrong: it hides real regressions among false
ones, and it costs a live model run every time it lies.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation"))

from checks import claims_yes, denies_knowledge, mentions_code, scores, states_number  # noqa: E402


class TestTheVerdictsThatWereWrong:
    """The exact answers this scorer marked FAIL while they were correct."""

    def test_a_code_ending_a_sentence(self) -> None:
        assert mentions_code("It requires one of 00940224, 00940226.", "00940226")

    def test_a_decimal_ending_a_sentence(self) -> None:
        assert states_number("Your degree requires 155 credits, and you still need 25.5.", "25.5")

    def test_a_gpa_ending_a_sentence(self) -> None:
        assert states_number("Your current GPA is 72.64.", "72.64")

    def test_a_denial_phrased_as_could_not_find(self) -> None:
        assert denies_knowledge(
            "I couldn't find a recorded grade for 00960211 in your transcript."
        )

    def test_a_denial_phrased_as_could_not_confirm(self) -> None:
        assert denies_knowledge(
            "I can't assess eligibility for 00999999 because I could not confirm "
            "that course in the catalog: 0 catalog rows were found."
        )


class TestBoundariesStillHold:
    def test_a_longer_number_does_not_satisfy_its_prefix(self) -> None:
        assert not states_number("requires 155 credits", "15")

    def test_a_decimal_does_not_satisfy_its_tail(self) -> None:
        assert not states_number("completed 129.5 credits", "29.5")

    def test_a_year_does_not_satisfy_a_two_digit_check(self) -> None:
        assert not states_number("from 2025 onward", "20")

    def test_a_code_inside_an_edge_id_does_not_count(self) -> None:
        """It named an internal key, not a course a student can register for."""
        assert not mentions_code("one of 00960211->00940224, 00960211->00940226", "00940224")


class TestDenialAndAffirmationAreDistinguished:
    def test_a_plain_yes_is_an_affirmation(self) -> None:
        assert claims_yes("yes — you meet 1 of 1 prerequisite groups.")
        assert not denies_knowledge("yes — you meet 1 of 1 prerequisite groups.")

    def test_a_refusal_is_not_an_affirmation(self) -> None:
        text = "I could not confirm that course in the catalog."
        assert denies_knowledge(text)
        assert not claims_yes(text)

    def test_the_dangerous_answer_is_recognised_as_a_claim(self) -> None:
        """The vacuous-eligibility failure: a confident yes about a course that
        does not exist. It must read as a CLAIM, not as a denial."""
        text = "yes — this course has 0 prerequisite groups, and you meet 0 of them."
        assert claims_yes(text)
        assert not denies_knowledge(text)


class TestScores:
    def test_a_matching_answer_passes(self) -> None:
        passed, _ = scores("You have completed 129.5 credits.", must=("129.5",), must_not=("135",))
        assert passed

    def test_a_known_wrong_value_fails(self) -> None:
        passed, why = scores("You have completed 135 credits.", must=("129.5",), must_not=("135",))
        assert not passed and "135" in why

    def test_no_answer_fails_distinctly(self) -> None:
        passed, why = scores(None, must=("129.5",))
        assert not passed and "no answer" in why
