"""A true/false slot lands badly in a sentence that already says the word.

Comparisons yield BOOL, and a BOOL slot renders as the bare word "yes" or "no".
On its own that reads correctly -- "Eligible: yes." -- but the model tends to
write the word AND slot the fact, producing "Yes -- yes." and "You are yes
eligible". Measured on the eligibility question, roughly a third of otherwise
correct answers came out one of those two ways.

The prompt asks for neither and mostly gets its way. This catches the rest, and
is deliberately narrow: it edits an answer that has already passed grounding, so
the one thing it must never do is change what the answer CLAIMS.
"""

from __future__ import annotations

from app.agent_core.facts.answer import _tidy_affirmations


class TestTheObservedFailures:
    def test_a_doubled_yes_collapses(self) -> None:
        assert _tidy_affirmations("Yes -- yes. You meet 1 of 1 groups.") == (
            "Yes. You meet 1 of 1 groups."
        )

    def test_an_em_dash_doubling_collapses(self) -> None:
        assert _tidy_affirmations("Yes — yes.") == "Yes."

    def test_a_stranded_yes_between_copula_and_adjective_is_dropped(self) -> None:
        assert _tidy_affirmations("You are yes eligible to take 00960211.") == (
            "You are eligible to take 00960211."
        )

    def test_it_works_for_no_as_well(self) -> None:
        assert _tidy_affirmations("No, no. You still need 25.5 credits.") == (
            "No. You still need 25.5 credits."
        )


class TestGoodPhrasingIsLeftAlone:
    def test_a_leading_slot_is_untouched(self) -> None:
        text = "yes — you meet 1 of 1 prerequisite groups."
        assert _tidy_affirmations(text) == text

    def test_a_labelled_slot_is_untouched(self) -> None:
        text = "Eligible: yes. Prerequisite alternatives: 00940224, 00940226."
        assert _tidy_affirmations(text) == text

    def test_a_trailing_slot_is_untouched(self) -> None:
        text = "You are eligible: yes."
        assert _tidy_affirmations(text) == text

    def test_the_word_yes_in_ordinary_prose_survives(self) -> None:
        text = "Yes, you can take it, and no prerequisites are outstanding."
        assert _tidy_affirmations(text) == text


class TestItNeverChangesTheClaim:
    def test_a_yes_is_never_turned_into_a_no(self) -> None:
        for text in ("Yes -- yes.", "You are yes eligible.", "Eligible: yes."):
            assert "no" not in _tidy_affirmations(text).lower().replace("no prerequisite", "")

    def test_a_no_is_never_turned_into_a_yes(self) -> None:
        assert "yes" not in _tidy_affirmations("No -- no.").lower()

    def test_numbers_are_never_touched(self) -> None:
        text = "Yes -- yes. You have 129.5 of 155 credits and need 25.5 more."
        tidied = _tidy_affirmations(text)
        for number in ("129.5", "155", "25.5"):
            assert number in tidied
