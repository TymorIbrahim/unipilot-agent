"""An answer must not count zero met groups and then declare eligibility.

Shipped live, and scored as a CORRECT answer:

    "No. You checked 1 prerequisite group and met 0, so you are eligible to
     take 01040174."

Both halves are in one sentence and they are opposites. Worse than a plainly
wrong answer: a reader who skims the front takes "No", a reader who skims the
end takes "eligible", and the eval's three-state scorer passed it because
`claims_no` matched the leading word and stopped looking.

That is why this is a POST-CONDITION and not a scorer rule. The scorer grades a
run after the fact; this refuses the answer before a student sees it.
"""

from __future__ import annotations

import pytest

from app.agent_core.facts.postconditions import (
    check_eligibility_is_not_self_contradictory as _check,
)

ASKED_ONE = "Am I eligible to take 01040174?"
ASKED_TWO = "Am I eligible for 00960211 and 01040174?"


def check(answer: str, question: str = ASKED_ONE):
    return _check(answer, question)


class TestTheLiveContradiction:
    def test_the_exact_answer_is_refused(self) -> None:
        assert check(
            "No. You checked 1 prerequisite group and met 0, so you are eligible to take 01040174."
        )

    def test_the_refusal_says_what_to_do(self) -> None:
        violations = check("You meet 0 of 1 prerequisite groups, so you are eligible.")
        assert "NOT eligible" in violations[0].message
        assert "name the prerequisite" in violations[0].message


class TestCoherentAnswersPass:
    @pytest.mark.parametrize(
        "answer",
        [
            "No — you meet 0 of 1 prerequisite groups, so you are not eligible.",
            "Eligible: no. You meet 0 of 1 prerequisite groups.",
            "Yes — you meet 1 of 1 prerequisite groups, so you are eligible.",
            "You are eligible to take 00960211; it needs any one of 00940224 or 00940226.",
            "You have completed 129.5 credits.",
            "",
        ],
        ids=["denies", "denies-colon", "affirms", "affirms-with-alts", "unrelated", "empty"],
    )
    def test_it_does_not_fire(self, answer: str) -> None:
        assert not check(answer)


class TestItIsScopedByTheQuestion:
    """The first version scoped by the ANSWER and disabled itself.

    It skipped any answer naming more than one course code, to spare a genuine
    two-course answer. But a good eligibility answer ALWAYS names the target and
    the prerequisites that would satisfy it -- three codes or more -- so the
    guard never fired once, and this shipped and scored PASS:

        "You are eligible for 01040174, because you meet 0 of 1 prerequisite
         groups. To make it yes, pass any one of 01040066, 01040166."
    """

    def test_the_answer_that_shipped_is_caught(self) -> None:
        assert check(
            "You are eligible for 01040174, because you meet 0 of 1 prerequisite groups. "
            "To make it yes, pass any one of 01040066, 01040166."
        )

    def test_naming_the_prerequisites_does_not_buy_an_exemption(self) -> None:
        """The regression in one line: extra codes in the answer are what a GOOD
        answer has, so they must not switch the check off."""
        assert check(
            "You are eligible, though you meet 0 of 1 groups; pass 01040066 or 01040166."
        )

    def test_two_courses_asked_about_are_left_alone(self) -> None:
        """Which clause owns which verdict needs a parser this does not have,
        and blocking a correct answer is the worse error."""
        assert not check(
            "You are eligible for 00960211, but you meet 0 of 1 groups for 01040174.",
            ASKED_TWO,
        )

    def test_no_course_in_the_question_stands_aside(self) -> None:
        assert not check("You are eligible, and you meet 0 of 1 groups.", "Am I eligible?")
