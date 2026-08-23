"""Every wrong verdict this scorer has produced, pinned as a test.

Five correct answers were scored as failures in one session, each by a
hand-rolled regex in a throwaway probe. A scorer wrong in the pessimistic
direction is not the safe kind of wrong: it hides real regressions among false
ones, and it costs a live model run every time it lies.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation"))

from checks import (  # noqa: E402
    claims_no,
    stated_period_count,
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


class TestAContradictionIsNotAPass:
    """Scored CORRECT on a live run, and it is neither correct nor wrong-and-clear.

        "You are eligible for 01040174, because you meet 0 of 1 prerequisite
         groups. To make it yes, pass any one of 01040066, 01040166."

    `claims_no` matched "meet 0 of", the deny stance was satisfied, and nothing
    looked further. A reader who takes the first clause and a reader who takes
    the second get opposite advice.
    """

    SHIPPED = (
        "You are eligible for 01040174, because you meet 0 of 1 prerequisite groups. "
        "To make it yes, pass any one of 01040066, 01040166."
    )

    def test_the_shipped_contradiction_is_wrong(self) -> None:
        verdict, why = scores(self.SHIPPED, must=("01040066", "01040166"), stance="deny")
        assert verdict == "wrong"
        assert "affirms and denies" in why

    def test_a_coherent_denial_still_passes(self) -> None:
        verdict, _ = scores(
            "No - 01040174 needs any one of 01040066, 01040166, and you meet 0 of 1 groups.",
            must=("01040066", "01040166"),
            stance="deny",
        )
        assert verdict == "correct"

    def test_not_eligible_is_not_read_as_an_affirmation(self) -> None:
        """The obvious way to break this: "you are not eligible" contains
        "eligible"."""
        verdict, _ = scores(
            "You are not eligible; you meet 0 of 1 groups. Pass 01040066 or 01040166.",
            must=("01040066", "01040166"),
            stance="deny",
        )
        assert verdict == "correct"

    def test_a_clean_affirmation_is_untouched(self) -> None:
        verdict, _ = scores(
            "Yes - you meet 1 of 1 prerequisite groups; it needs any one of 00940224, 00940226.",
            must=("00940224", "00940226"),
        )
        assert verdict == "correct"


class TestMeetingNoneIsNotAnAffirmation:
    """`_AFFIRMS` carried a bare `you meet`, so "you meet 0 of 1 prerequisite
    groups" -- a refusal -- read as an affirmation. Invisible until the
    contradiction check asked whether both fired at once, at which point every
    correct denial scored as self-contradictory."""

    def test_meeting_zero_does_not_affirm(self) -> None:
        assert not claims_yes("you meet 0 of 1 prerequisite groups.")
        assert claims_no("you meet 0 of 1 prerequisite groups.")

    def test_meeting_some_still_affirms(self) -> None:
        assert claims_yes("You meet 1 of 1 prerequisite groups.")
        assert claims_yes("You have met 2 of 2 groups.")

    def test_the_bare_word_no_longer_counts(self) -> None:
        """"meet" on its own says nothing about the verdict."""
        assert not claims_yes("Let us meet the requirements listed below.")


class TestAGiveUpIsNotAThinAnswer:
    """Scored `incomplete` -- "right answer, but never states 2" -- for a run
    that answered nothing:

        "I wasn't able to work that out from your records with confidence."

    Every question in the ground truth is answerable from the data by
    construction, so declining one is a failure. Filing it beside genuinely
    correct-but-terse answers flatters the score in the direction that matters.
    """

    GAVE_UP = "I wasn't able to work that out from your records with confidence."

    def test_the_refusal_is_wrong_not_thin(self) -> None:
        verdict, why = scores(self.GAVE_UP, must=("2",))
        assert verdict == "wrong"
        assert "declined" in why

    def test_it_is_caught_with_no_must_contain_at_all(self) -> None:
        """The check started inside the `missing` branch, so a question with an
        empty `must_contain` never reached it and a give-up scored CORRECT.
        `semesters_to_graduate` needs that empty list -- 2 is a floor and 3 is
        also right -- which is exactly where it went unnoticed."""
        verdict, why = scores(self.GAVE_UP, must=(), must_not=("1.42",))
        assert verdict == "wrong" and "declined" in why

    def test_a_genuinely_thin_answer_is_still_thin(self) -> None:
        verdict, _ = scores(
            "No. You meet 0 of 1 prerequisite groups.",
            must=("01040066", "01040166"),
            stance="deny",
        )
        assert verdict == "incomplete"

    def test_a_correct_answer_is_unaffected(self) -> None:
        verdict, _ = scores("You need 2 English-language courses.", must=("2",))
        assert verdict == "correct"

    def test_a_denial_of_eligibility_is_not_a_denial_of_knowledge(self) -> None:
        """The boundary that matters: "you have passed neither" and "you are not
        eligible" are ANSWERS, and must not be swept up as give-ups."""
        for answer in (
            "No -- 01040174 needs 01040066, 01040166; you have passed neither.",
            "You are not eligible. You need any one of 01040066, 01040166.",
        ):
            verdict, why = scores(
                answer, must=("01040066", "01040166"), stance="deny"
            )
            assert verdict == "correct", f"{answer!r} -> {why}"


class TestEveryRealAnswerFromTheEvalRuns:
    """A corpus, not a rule. Each of these came out of a live run today, and
    every scorer change gets checked against all of them.

    Two were misread, and both cost a correct answer its mark:

      "No -- you meet 0 of 1 prerequisite groups. To make it yes, pass any one
       of 01040066, 01040166."

    scored "affirms and denies in the same answer", because a bare \\byes\\b
    matched inside "To make it yes" -- which is the phrasing the SYSTEM PROMPT
    now explicitly asks for. A rule that generates prose the scorer punishes is
    the worst kind of drift.

      "The course 00940412 is in the catalog, and next spring is forecast to
       offer it."

    scored "never affirms". It affirms, in the passive.
    """

    AFFIRMATIONS = [
        "Yes — 00940412 has been offered every spring on record.",
        "yes.",
        "Eligible: yes.",
        "The course is offered next spring: yes.",
        "For the course you asked about: yes.",
        "yes — you meet 1 of 1 prerequisite groups.",
        "Yes. 00960211 has 1 prerequisite group, and the requirement is any one of "
        "00940224, 00940226.",
        "The course 00940412 is in the catalog, and next spring is forecast to offer it.",
        "00940412 is forecast to be offered next spring: yes.",
    ]

    DENIALS = [
        "No — you meet 0 of 1 prerequisite groups. To make it yes, pass any one of "
        "01040066, 01040166.",
        "No. 01040174 requires any one of 01040066, 01040166, and you have passed neither.",
        "You are not eligible. You need any one of 01040066, 01040166.",
        "For 01040174, no — you meet 0 of 1 prerequisite groups.",
        "No — 01040174 needs 01040066, 01040166; you have passed neither.",
    ]

    @pytest.mark.parametrize("answer", AFFIRMATIONS)
    def test_an_affirmation_reads_as_one_and_only_one(self, answer: str) -> None:
        assert claims_yes(answer), "affirmation not recognised"
        assert not claims_no(answer), "affirmation also read as a denial"

    @pytest.mark.parametrize("answer", DENIALS)
    def test_a_denial_reads_as_one_and_only_one(self, answer: str) -> None:
        assert claims_no(answer), "denial not recognised"
        assert not claims_yes(answer), "denial also read as an affirmation"

    IMPOSSIBILITIES = [
        "Because that minimum is above the maximum possible grade, the target is not reachable.",
        "Since 138.57 is above the maximum possible grade, the GPA target is not reachable.",
        "you would need at least 112.32 in every remaining course, which is not achievable.",
        "That target cannot be met with the remaining credits.",
        "Reaching 85 is impossible from here.",
    ]

    @pytest.mark.parametrize("answer", IMPOSSIBILITIES)
    def test_an_unreachable_target_reads_as_a_denial(self, answer: str) -> None:
        """The GPA question scored 0/3 "never states the negative" on three
        answers that all said so -- in words no pattern here covered."""
        assert claims_no(answer), "an impossibility is a denial"
        assert not claims_yes(answer)

    def test_the_hypothetical_yes_does_not_affirm(self) -> None:
        """Isolated, because the prompt asks for exactly this shape."""
        assert not claims_yes("To make it yes, pass any one of 01040066, 01040166.")


class TestCountingSemesters:
    """Two runs, identical data, minutes apart -- "2 semesters" and "4
    semesters" -- and both scored CORRECT, because this question's
    `must_contain` was empty and nothing else could see a count.

    Empty was the honest choice at the time: 2 is a floor, 3 is legitimate
    slippage once offerings are applied, and a bare number cannot express that.
    A bare number is also wrong in both directions here -- `states_number` is
    bounded against digits, not meaning, so a required "2" is satisfied by any
    2-credit course in the plan listing, and a forbidden "4" fails a correct
    answer that happens to list a 4-credit course.
    """

    def test_the_leading_verdict_is_the_count(self) -> None:
        assert stated_period_count("It will take 2 semesters to graduate.") == 2

    def test_the_four_semester_run_reads_as_four(self) -> None:
        assert stated_period_count("You will need 4 semesters to graduate.") == 4

    def test_a_later_enumeration_does_not_overwrite_the_verdict(self) -> None:
        """Answers lead with the verdict and then list the terms, and those rows
        carry numerals of their own."""
        answer = (
            "It will take 2 semesters to graduate.\n\n"
            "- term 2026-1 · credits 16 · 6 courses\n"
            "- term 2026-2 · credits 12 · 4 courses\n"
        )
        assert stated_period_count(answer) == 2

    def test_at_least_is_still_a_count(self) -> None:
        assert stated_period_count("At least 2 semesters: you need 25.5 more credits.") == 2

    def test_words_count_too(self) -> None:
        assert stated_period_count("You will need two more semesters.") == 2

    def test_terms_count_as_semesters(self) -> None:
        assert stated_period_count("Three more terms should finish it.") == 3

    def test_an_answer_naming_no_count_says_so(self) -> None:
        assert stated_period_count("You still need 25.5 credits.") is None

    def test_credits_are_not_mistaken_for_a_count(self) -> None:
        assert stated_period_count("- term spring · credits 4 · 00970414") is None

    RANGE = {"periods": (2, 3)}

    def test_the_floor_passes(self) -> None:
        verdict, _ = scores("It will take 2 semesters. You need 25.5 more credits.",
                            must=(25.5,), **self.RANGE)
        assert verdict == "correct"

    def test_the_legitimate_three_passes(self) -> None:
        """Offerings can stretch the floor, and the derivation says so."""
        verdict, _ = scores("It will take 3 semesters. You need 25.5 more credits.",
                            must=(25.5,), **self.RANGE)
        assert verdict == "correct"

    def test_four_is_wrong(self) -> None:
        verdict, why = scores("You will need 4 semesters. You need 25.5 more credits.",
                              must=(25.5,), **self.RANGE)
        assert verdict == "wrong"
        assert "4 semesters" in why

    def test_one_is_wrong(self) -> None:
        """Below the floor is the dangerous direction -- it under-promises the
        work and a student would plan around it."""
        verdict, _ = scores("You can finish in 1 semester. You need 25.5 more credits.",
                            must=(25.5,), **self.RANGE)
        assert verdict == "wrong"

    def test_naming_no_count_is_incomplete_not_wrong(self) -> None:
        verdict, _ = scores("You still need 25.5 credits to graduate.",
                            must=(25.5,), **self.RANGE)
        assert verdict == "incomplete"

    def test_a_give_up_is_still_caught_first(self) -> None:
        verdict, why = scores("I wasn't able to work that out from your records.",
                              must=(25.5,), **self.RANGE)
        assert verdict == "wrong"
        assert "declined" in why


class TestTheCopulaYes:
    """Live answer, scored FAIL while correct:

        "The course exists in the catalog, and the forecast for next spring
         is yes."

    `_AFFIRMS` accepted "yes" only at a start or after punctuation, deliberately:
    a bare \\byes\\b matched the hypothetical the prompt asks for, "To make it
    yes, pass 01040066", turning a correct DENIAL into a self-contradiction. The
    copula falls between the two -- and it is how the model most often ends an
    affirmative forecast.
    """

    def test_the_live_answer_affirms(self) -> None:
        assert claims_yes("The course exists in the catalog, and the forecast for "
                          "next spring is yes.")

    def test_the_hypothetical_still_does_not(self) -> None:
        assert not claims_yes("To make it yes, pass any one of 01040066, 01040166.")

    def test_a_denial_carrying_the_hypothetical_is_still_only_a_denial(self) -> None:
        answer = "No -- you meet 0 of 1. To make it yes, pass 01040066."
        assert claims_no(answer)
        assert not claims_yes(answer), "would score as affirming and denying at once"

    def test_the_other_copulas(self) -> None:
        for answer in ("The answer is yes.", "Offered next spring: the forecast was yes.",
                       "It will be yes if the pattern holds."):
            assert claims_yes(answer), answer


class TestAnAsideIsNotARefusal:
    """The agent began annotating a prerequisite the catalog does not carry:

        "Yes -- you are eligible. 00960211 requires any one of 00940224 (Data
         Structures and Algorithms), 00940226 (not in the course catalog)."

    A confident, correct, MORE informative answer -- and `\\bnot (in|...)\\b`
    matched inside the parenthetical, so all three runs of
    `eligibility_00960211` scored "declined to answer a question the data
    supports". 3/3 became 0/3 with nothing wrong with the agent.

    A note attached to ONE item is not the answer refusing the question, so the
    denial has to be in the answer's own voice.
    """

    def test_the_annotated_answer_does_not_read_as_a_refusal(self) -> None:
        from evaluation.checks import denies_knowledge

        assert not denies_knowledge(
            "Yes — you are eligible. 00960211 requires any one of 00940224 "
            "(Data Structures and Algorithms), 00940226 (not in the course catalog)."
        )

    def test_a_denial_in_the_answers_own_voice_still_counts(self) -> None:
        from evaluation.checks import denies_knowledge

        for text in (
            "00999999 is not in the catalog, so I cannot say anything about it.",
            "No such record exists for that course.",
            "I could not determine who teaches that course.",
        ):
            assert denies_knowledge(text), text


class TestTheContractionThatNeverMatched:
    """`can ?n[o']?t` matches "cannot" and "can not" and never "can't".

    "I can't confirm whether you met the deadline" is a textbook denial and read
    as none -- the pessimistic direction, which hides real regressions in a
    crowd of false ones.
    """

    def test_cant_is_a_denial(self) -> None:
        from evaluation.checks import denies_knowledge

        assert denies_knowledge("I can't confirm whether you met the deadline.")

    def test_couldnt_is_a_denial(self) -> None:
        from evaluation.checks import denies_knowledge

        assert denies_knowledge("I couldn't find a recorded grade.")

    def test_the_uncontracted_forms_still_work(self) -> None:
        from evaluation.checks import denies_knowledge

        assert denies_knowledge("I cannot determine who teaches that course.")
        assert denies_knowledge("I could not confirm that course.")
