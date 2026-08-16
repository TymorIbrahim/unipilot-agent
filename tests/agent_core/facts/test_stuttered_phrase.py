"""A TEXT slot that already contained the sentence the model wrote around it.

Shipped, and scored as a PASS because the number it had to state was present:

    "You cannot graduate without completing Cannot graduate without completing
     2 English courses."

`interpret` returned the requirement as a whole clause -- "Cannot graduate
without completing 2 English courses" -- and the model framed it with the same
clause. Correct, grounded, unreadable.

The same shape as the bool repair beside it: a slot renders a value that reads
correctly alone and badly once the model writes the words too. It edits an
answer that has already passed grounding, so the one thing it must never do is
change what the answer CLAIMS -- it drops a redundancy and nothing else.
"""

from __future__ import annotations

import pytest

from app.agent_core.facts.answer import _tidy_affirmations as tidy


class TestTheLiveStutter:
    def test_the_shipped_answer_is_repaired(self) -> None:
        assert tidy(
            "You cannot graduate without completing Cannot graduate without completing "
            "2 English courses."
        ) == "You cannot graduate without completing 2 English courses."

    def test_the_claim_is_unchanged(self) -> None:
        """The number the answer exists to state has to survive the edit."""
        out = tidy(
            "You cannot graduate without completing Cannot graduate without completing "
            "2 English courses."
        )
        assert "2 English courses" in out

    def test_case_does_not_hide_it(self) -> None:
        assert tidy("you need any one of You need any one of 00940224.") == (
            "you need any one of 00940224."
        )


class TestOrdinaryProseIsLeftAlone:
    @pytest.mark.parametrize(
        "text",
        [
            "You have completed 129.5 credits.",
            "The requirement is 2 English-language courses.",
            "No -- 01040174 needs any one of 01040066, 01040166.",
            "Your winter plan totals 16 credits.",
            "",
        ],
        ids=["credits", "requirement", "eligibility", "plan", "empty"],
    )
    def test_it_does_not_fire(self, text: str) -> None:
        assert tidy(text) == text

    def test_a_two_word_repeat_is_left_alone(self) -> None:
        """Below the threshold on purpose: "had had" and "that that" are ordinary
        English, and this edits answers that already passed grounding."""
        assert tidy("It is what it is and that that is fine.") == (
            "It is what it is and that that is fine."
        )

    def test_a_separated_repeat_is_left_alone(self) -> None:
        """Only an IMMEDIATE repeat is a stutter. Saying the same thing twice in
        one answer may be deliberate emphasis."""
        text = "You need 2 English courses. To graduate you need 2 English courses."
        assert tidy(text) == text


class TestItComposesWithTheBoolRepairs:
    def test_a_doubled_yes_still_collapses(self) -> None:
        assert tidy("Yes — yes.") == "Yes."

    def test_a_stranded_yes_still_clears(self) -> None:
        assert "yes eligible" not in tidy("You are yes eligible to take 00960211.")
