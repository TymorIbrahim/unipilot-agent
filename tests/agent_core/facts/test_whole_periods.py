"""A semester cannot be part-taken.

Asked how many semesters remained, the agent answered "you have 1.42 semesters
remaining at your current max load" -- 25.5 credits over an 18-credit cap,
reported as the raw quotient. The arithmetic is right and the sentence is not.

Wrong in the OPTIMISTIC direction, which is what makes it worth a gate: a student
reading 1.42 hears "nearly done in one more term" when they need two. Both runs
of that question answered this way, so it is the shape of the answer rather than
a slip.
"""

from __future__ import annotations

from app.agent_core.facts.postconditions import check_periods_are_whole


class TestFractionalPeriodsAreRefused:
    def test_the_live_failure_is_flagged(self) -> None:
        violations = check_periods_are_whole(
            "you have 1.42 semesters remaining at your current max load"
        )
        assert violations and violations[0].kind == "fractional_period"

    def test_it_says_to_round_up_and_gives_the_number(self) -> None:
        """Rounding DOWN loses a term the student must still attend."""
        message = check_periods_are_whole("1.42 semesters remaining")[0].message
        assert "at least 2" in message

    def test_terms_and_years_count_too(self) -> None:
        assert check_periods_are_whole("about 2.5 years remain")
        assert check_periods_are_whole("3.5 terms to go")


class TestContinuousQuantitiesAreUntouched:
    def test_credits_may_be_fractional(self) -> None:
        assert not check_periods_are_whole(
            "You have completed 129.5 credits and need 25.5 more."
        )

    def test_a_gpa_may_be_fractional(self) -> None:
        assert not check_periods_are_whole("Your GPA is 72.64.")

    def test_a_whole_period_passes(self) -> None:
        assert not check_periods_are_whole("It will take at least 2 semesters.")

    def test_a_credit_figure_beside_a_period_word_is_not_confused(self) -> None:
        assert not check_periods_are_whole("18.0 credits per semester")
