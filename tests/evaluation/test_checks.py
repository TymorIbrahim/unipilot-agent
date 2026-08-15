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

from checks import (  # noqa: E402
    claims_no,
    claims_yes,
    denies_knowledge,
    mentions_code,
    scores,
    states_number,
)


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
        verdict, _ = scores("You have completed 129.5 credits.", must=("129.5",), must_not=("135",))
        assert verdict == "correct"

    def test_a_known_wrong_value_fails(self) -> None:
        verdict, why = scores("You have completed 135 credits.", must=("129.5",), must_not=("135",))
        assert verdict == "wrong" and "135" in why

    def test_no_answer_fails_distinctly(self) -> None:
        verdict, why = scores(None, must=("129.5",))
        assert verdict == "wrong" and "no answer" in why


class TestRightButThinIsNotWrong:
    """Two states hid the only distinction that matters.

    `eligibility_01040174` answered "yes -- you meet 1 of 1 prerequisite groups"
    to a student who meets NONE: a student told to register for something they
    cannot take. Fixed, it answered "No. You meet 0 of 1", which is correct but
    never says WHICH prerequisite is missing -- so it still failed
    `must_contain` and the score stayed 0/3. A measurement that cannot see a
    dangerous answer become a correct one is not measuring anything.
    """

    DANGEROUS = "Yes -- you meet 1 of 1 prerequisite groups for 01040174."
    THIN = "No. You meet 0 of 1 prerequisite groups for 01040174, so no."
    FULL = "No. 01040174 needs either 01040066 or 01040166, and you passed neither."

    def _score(self, text: str):
        return scores(text, must=("01040066", "01040166"), stance="deny")

    def test_the_dangerous_answer_is_wrong(self) -> None:
        verdict, _ = self._score(self.DANGEROUS)
        assert verdict == "wrong"

    def test_the_terse_answer_is_incomplete_not_wrong(self) -> None:
        verdict, why = self._score(self.THIN)
        assert verdict == "incomplete", "a correct verdict must not score as a wrong one"
        assert "right answer" in why

    def test_the_full_answer_is_correct(self) -> None:
        verdict, _ = self._score(self.FULL)
        assert verdict == "correct"

    def test_the_three_are_distinguishable(self) -> None:
        """The whole point: fixing the defect has to move the number."""
        verdicts = [self._score(t)[0] for t in (self.DANGEROUS, self.THIN, self.FULL)]
        assert verdicts == ["wrong", "incomplete", "correct"]

    def test_stance_is_judged_before_the_numbers(self) -> None:
        """An answer that is wrong AND thin is WRONG. Reporting it as merely
        incomplete would understate it."""
        verdict, _ = scores(
            "Yes, you are eligible.", must=("01040066", "01040166"), stance="deny"
        )
        assert verdict == "wrong"

    def test_a_known_wrong_value_still_beats_everything(self) -> None:
        verdict, _ = scores("No -- the rate is 0.43.", must_not=("0.43",), stance="deny")
        assert verdict == "wrong"


class TestAClaimOfNoIsNotADenialOfKnowledge:
    """`forecast`'s documented failure inverts the ANSWER while keeping every
    number and course code intact. The two shapes look alike and must not."""

    def test_an_inverted_forecast_is_a_claim(self) -> None:
        text = "00940412 will not be offered next spring."
        assert claims_no(text)
        assert not denies_knowledge(text), "this asserts a fact; it does not decline to"

    def test_an_honest_refusal_is_not_a_negative_claim(self) -> None:
        text = "I could not determine whether 00940412 runs next spring."
        assert denies_knowledge(text)
        assert not claims_no(text), "declining to answer must not score as answering no"

    def test_a_leading_no_is_a_negative_claim(self) -> None:
        assert claims_no("No — you have not met the prerequisites for that course.")

    def test_a_denial_after_the_course_name_is_still_a_denial(self) -> None:
        """Live answer, scored FAIL while correct. Requiring "No" at the start of
        the text missed it because the model led with the course."""
        assert claims_no("For 01040174, no — you meet 0 of 1 prerequisite groups.")

    def test_the_count_alone_carries_the_refusal(self) -> None:
        """Prose gets reworded; "you meet 0 of 1" does not, and it is the refusal
        rather than a decoration on it."""
        assert claims_no("You meet 0 of 1 prerequisite groups for that course.")
        assert claims_no("The course has 1 prerequisite group, and you meet 0 of 1.")

    def test_meeting_some_is_not_a_refusal(self) -> None:
        """The boundary: 1 of 1 must not read as a denial just because it counts."""
        assert not claims_no("You meet 1 of 1 prerequisite groups for 01040174.")

    def test_a_projection_affirms_without_the_word_yes(self) -> None:
        """Marking this a non-answer is the pessimistic failure this file exists
        to prevent -- it is a correct, well-hedged forecast."""
        text = "00940412 has run every spring for the last three years, so it is expected again."
        assert claims_yes(text)
        assert not claims_no(text)


class TestStance:
    AFFIRMED = "Yes — 00940412 has been offered every spring on record."
    INVERTED = "00940412 will not be offered next spring."

    def test_an_affirmation_passes_an_affirm_stance(self) -> None:
        verdict, _ = scores(self.AFFIRMED, must=("00940412",), stance="affirm")
        assert verdict == "correct"

    def test_the_inversion_fails_even_though_every_number_matches(self) -> None:
        """The whole reason stance exists: `must_contain` cannot tell these
        apart, because the wrong answer names the same course as the right one."""
        by_numbers, _ = scores(self.INVERTED, must=("00940412",))
        assert by_numbers == "correct", "precondition: the numeric check alone cannot catch this"

        verdict, why = scores(self.INVERTED, must=("00940412",), stance="affirm")
        assert verdict == "wrong" and "no" in why

    def test_silence_on_the_question_fails_an_affirm_stance(self) -> None:
        text = "00940412 is worth 4 credits and belongs to your track."
        verdict, why = scores(text, must=("00940412",), stance="affirm")
        assert verdict == "wrong" and "never affirms" in why

    def test_a_deny_stance_wants_the_negative(self) -> None:
        verdict, _ = scores("No — that course is not offered in winter.", stance="deny")
        assert verdict == "correct"
        verdict, _ = scores("Yes, go ahead and take it.", stance="deny")
        assert verdict == "wrong"

    def test_no_stance_leaves_scoring_unchanged(self) -> None:
        verdict, _ = scores(self.INVERTED, must=("00940412",), stance=None)
        assert verdict == "correct"
