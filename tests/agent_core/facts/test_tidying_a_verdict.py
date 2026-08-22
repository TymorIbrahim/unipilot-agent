"""The repair that edits a grounded answer, and the two ways it changed meaning.

`_tidy_affirmations` drops a redundant word when a BOOL slot lands badly in a
sentence -- "You are {eligible} eligible" renders "You are yes eligible". Its
docstring states the constraint it operates under: it edits an answer that has
already passed grounding, so "the one thing it must never do is change what the
answer CLAIMS."

It did, twice, and both were found by asking the deployed agent questions the
eval does not cover.

1. A DELETED NEGATION. Asked for a grade in a course the student never took:

       I found the course in the catalog, but there are 0 transcript attempts
       for it, so there is grade on record.

   The model wrote "there is no grade on record". The rule matched `is no <word>`
   and deleted the "no". It excluded conjunctions -- "and", "or", "but" -- but
   "grade" is not a conjunction, and no blacklist of conjunctions can separate a
   determiner from a stranded bool.

2. An INVERTED VERDICT, present from the first version. "You are no eligible for
   01040174" -- the bool rendering of an INELIGIBLE result -- became "You are
   eligible for 01040174", which tells a student to register for something they
   cannot take. Deleting the word is a tidy-up for "yes" and a reversal for
   "no", because "no" is the word doing the negating.
"""

from __future__ import annotations

import pytest

from app.agent_core.facts.answer import _tidy_affirmations


class TestItNeverDeletesANegation:
    @pytest.mark.parametrize("answer", [
        "I found the course in the catalog, but there are 0 transcript attempts "
        "for it, so there is no grade on record.",
        "There is no record of that course.",
        "There are no prerequisites for it.",
        "There is no offering next spring.",
        "There was no attempt recorded in 2024.",
    ])
    def test_a_determiner_survives(self, answer: str) -> None:
        assert _tidy_affirmations(answer) == answer


class TestItNeverInvertsAVerdict:
    def test_the_negative_becomes_not_rather_than_nothing(self) -> None:
        assert _tidy_affirmations("You are no eligible for 01040174.") == (
            "You are not eligible for 01040174."
        )

    def test_with_an_adverb_between(self) -> None:
        assert _tidy_affirmations("You are already no eligible for 01040174.") == (
            "You are already not eligible for 01040174."
        )

    def test_the_affirmative_still_just_loses_the_word(self) -> None:
        assert _tidy_affirmations("You are yes eligible for 00960324.") == (
            "You are eligible for 00960324."
        )

    def test_the_adverb_case_it_was_written_for(self) -> None:
        assert _tidy_affirmations("You are already yes eligible for 00960324.") == (
            "You are already eligible for 00960324."
        )


class TestItLeavesRealPredicatesAlone:
    @pytest.mark.parametrize("answer", [
        "The answer is yes and the course is open.",
        "Eligible: yes.",
        "The forecast for next spring is yes.",
        "You are yes for 00960211.",  # odd, but TRUE -- better than "you are for"
    ])
    def test_untouched(self, answer: str) -> None:
        assert _tidy_affirmations(answer) == answer

    def test_the_doubled_word_is_still_collapsed(self) -> None:
        assert _tidy_affirmations("Yes -- yes, you can take it.") == "Yes, you can take it."

    def test_the_stuttered_phrase_is_still_collapsed(self) -> None:
        answer = ("You cannot graduate without completing Cannot graduate without "
                  "completing 2 English courses.")
        assert _tidy_affirmations(answer).count("raduate without completing") == 1
