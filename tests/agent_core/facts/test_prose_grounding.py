"""The grounding check used to accept the model's own evidence for its own claim.

`interpret` and `interpret_list` verify that an extracted value APPEARS in the
text -- that is the whole grounding guarantee, and the reason a computed number
cannot enter the fact set. But the text they checked against was
`quote or passage.excerpt`, and the quote comes back from the model in the same
reply as the value. So a model that invented both passed: given the code
00999999 with the quote "00999999 is approved", against a passage that never
mentions it, `extract_list` kept the value and `interpret` accepted it.

The value is now checked against the PASSAGE. The quote stays in the citation so
a reader can see what was claimed; it no longer decides whether the claim is real.
"""

from __future__ import annotations

from app.agent_core.facts.prose import Passage, interpret, interpret_list
from app.agent_core.facts.types import ScalarKind

PASSAGE = Passage(
    slug="track-x",
    title="Track X",
    excerpt="Faculty electives: 00960600, 00960617 and 00970325 are approved.",
    score=1.0,
)


class _ListExtractor:
    def __init__(self, pairs):
        self.pairs = pairs

    async def extract_all(self, passage, question, expect):
        return list(self.pairs)


class _Extractor:
    def __init__(self, value, quote):
        self.value, self.quote = value, quote

    async def extract(self, passage, question, expect):
        return self.value, self.quote


class TestAFabricatedQuoteCannotGroundAValue:
    async def test_extract_list_drops_the_invented_code(self) -> None:
        result = await interpret_list(
            _ListExtractor([
                ("00960600", "Faculty electives: 00960600"),
                ("00999999", "00999999 is approved"),  # neither is in the passage
            ]),
            PASSAGE,
            "every elective code",
            expect=ScalarKind.IDENTIFIER,
        )
        values = [record.fields["value"].value for record in result.value.records]
        assert "00999999" not in values, "a code the passage never mentions must not enter the set"
        assert values == ["00960600"]

    async def test_interpret_refuses_the_invented_value(self) -> None:
        result = await interpret(
            _Extractor("00999999", "00999999 is approved"),
            PASSAGE,
            "the elective code",
            expect=ScalarKind.IDENTIFIER,
        )
        assert not hasattr(result, "value"), "a fabricated quote must not ground a value"
        assert "does not appear" in result.message


class TestTheCheckIsNotOverTightened:
    async def test_a_real_value_with_no_quote_is_still_accepted(self) -> None:
        """An extractor that returns no quote is untidy, not dishonest. The value
        is in the passage, which is what the guarantee is actually about."""
        result = await interpret(
            _Extractor("00960617", ""), PASSAGE, "the code", expect=ScalarKind.IDENTIFIER
        )
        assert hasattr(result, "value")
        assert result.value.value == "00960617"

    async def test_a_real_value_with_a_sloppy_quote_is_still_accepted(self) -> None:
        """A quote that does not match the passage verbatim -- reformatted, or
        pointing at the wrong sentence -- must not discard a real value."""
        result = await interpret(
            _Extractor("00970325", "some loosely remembered wording"),
            PASSAGE,
            "the code",
            expect=ScalarKind.IDENTIFIER,
        )
        assert hasattr(result, "value")
        assert result.value.value == "00970325"
