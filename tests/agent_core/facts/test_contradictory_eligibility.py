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

from app.agent_core.facts.postconditions import check_eligibility_is_not_self_contradictory as check


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


class TestItDoesNotBlockAMultiCourseAnswer:
    def test_two_courses_with_opposite_verdicts_are_fine(self) -> None:
        """The false positive the first version produced. These halves describe
        DIFFERENT courses, and refusing it ends the run with nothing rather than
        with something imperfect."""
        assert not check(
            "You are eligible for 00960211, but you meet 0 of 1 groups for 01040174."
        )

    def test_the_limit_is_deliberate(self) -> None:
        """Stated as a test so it is a known gap rather than a surprise: a
        contradiction inside a multi-course answer is NOT caught, because
        deciding which clause owns which code needs a parser this does not have.
        """
        assert not check(
            "For 00960211 and 01040174 you meet 0 groups, so you are eligible."
        )
