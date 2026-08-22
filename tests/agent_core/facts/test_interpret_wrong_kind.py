"""The page was right and the expected KIND was wrong, and the refusal said
neither.

"How many times can I retake a course?" has no number in the regulations at all
-- 5.3d says a failed mandatory course may be re-registered with "no time
limit". Asked for a QUANTITY, `interpret` correctly finds none, and the refusal
used to end:

    Reading 'regulations-undergraduate' again will return the same thing --
    a different source is needed.

So the model went looking for another source. It had the right one. Measured
across a policy set: 27 of these, each followed by another `search_corpus` under
a fresh fact name -- which also slips past the repeated-derivation guard,
because the name differs. One question spent 32 steps and 129 seconds this way.

The fifth unfollowable refusal found this week. The pattern does not vary: a
message that cannot be acted on costs more than no message, because the model's
next move is to try again.
"""

from __future__ import annotations

from app.agent_core.facts.prose import Passage, _cannot_determine
from app.agent_core.facts.types import ScalarKind

PAGE = Passage(slug="regulations-undergraduate", title="Regulations",
               excerpt="...no time limit...", score=1.0)
QUESTION = "how many times can I retake a course"


def _message(expect: ScalarKind) -> str:
    return _cannot_determine(PAGE, QUESTION, expect, why="it contains no such value").message


class TestItOffersTheOtherKind:
    def test_a_failed_quantity_suggests_text(self) -> None:
        message = _message(ScalarKind.QUANTITY)
        assert 'expect: "text"' in message
        assert "no time limit" in message, "show the shape of a phrase answer"

    def test_a_failed_text_suggests_quantity(self) -> None:
        assert 'expect: "quantity"' in _message(ScalarKind.TEXT)

    def test_it_names_the_same_page_to_re_read(self) -> None:
        mentions = _message(ScalarKind.QUANTITY).count("regulations-undergraduate")
        assert mentions >= 2, "the page to ask again is the one it just read"


class TestItNoLongerSendsTheModelAway:
    def test_a_different_source_is_not_asserted(self) -> None:
        message = _message(ScalarKind.QUANTITY)
        assert "a different source is needed" not in message

    def test_looking_elsewhere_is_conditional(self) -> None:
        message = _message(ScalarKind.QUANTITY)
        assert "Only look elsewhere if" in message

    def test_it_still_says_re_reading_the_same_way_is_futile(self) -> None:
        """The original point survives -- the same request gets the same answer."""
        assert "will return the same thing" in _message(ScalarKind.QUANTITY)
