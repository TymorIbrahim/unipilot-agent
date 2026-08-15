"""The scorer decides whether the agent is right, so it has to be right first.

Its first version marked three correct answers wrong. `mentions` guarded number
boundaries with `(?![\\d.])` so that 155 would not satisfy a check for 15 -- and
that same guard rejected "requires one of 00940224, 00940226." because the
sentence ended in a full stop. A broken yardstick is worse than none: it reported
8/12 when the answers were 11/12, and the three it condemned were perfect.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation"))

from run_eval import mentions, score  # noqa: E402


class TestNumbersEndingASentence:
    def test_a_course_code_before_a_full_stop_counts(self) -> None:
        assert mentions("It requires one of 00940224, 00940226.", "00940226")

    def test_a_decimal_before_a_full_stop_counts(self) -> None:
        assert mentions("you still need 25.5.", "25.5")

    def test_a_number_before_a_comma_counts(self) -> None:
        assert mentions("requires 00940224, and you passed it", "00940224")


class TestNumberBoundariesStillHold:
    def test_a_longer_number_does_not_satisfy_its_prefix(self) -> None:
        assert not mentions("Your degree requires 155 credits", "15")

    def test_a_decimal_does_not_satisfy_its_tail(self) -> None:
        """129.5 must not be read as evidence of 29.5."""
        assert not mentions("You have completed 129.5 credits", "29.5")

    def test_a_year_does_not_satisfy_a_two_digit_check(self) -> None:
        """The must_not_contain trap: '20' inside '2025' scored a right answer
        wrong the other way round."""
        assert not mentions("from 2025 onward", "20")

    def test_the_bare_number_still_matches(self) -> None:
        assert mentions("you still need 20 credits", "20")


class TestVerdicts:
    _question = {"must_contain": ["129.5"], "must_not_contain": ["135"]}

    def test_a_matching_answer_is_correct(self) -> None:
        verdict, _ = score("You have completed 129.5 credits.", self._question)
        assert verdict == "correct"

    def test_the_known_wrong_value_is_caught(self) -> None:
        verdict, why = score("You have completed 135 credits.", self._question)
        assert verdict == "wrong"
        assert "135" in why

    def test_a_missing_value_is_caught(self) -> None:
        verdict, _ = score("You have completed some credits.", self._question)
        assert verdict == "wrong"

    def test_a_run_that_gave_up_is_not_scored_as_wrong_content(self) -> None:
        """A give-up and a wrong number are different failures and are counted
        apart -- one is a loop problem, the other a grounding problem."""
        verdict, _ = score(None, self._question)
        assert verdict == "no-answer"
