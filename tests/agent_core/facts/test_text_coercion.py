"""A number read out of a passage is text.

`_as_kind` returned None for anything non-str when TEXT was asked for, and the
refusal that followed was expensive. Asked "what is the English language
requirement I have to satisfy to graduate", the model expects a sentence and the
extractor returns 2:

    "'regulations-undergraduate' does not answer '...' with a text: the value
     found (2) is not a text. Reading 'regulations-undergraduate' again will
     return the same thing -- a different source is needed."

So the model went looking for another source, and there is no other source. Runs
that hit it took 11-15 steps against a usual 4.

Safe to widen because TEXT is the loosest kind and `_appears_in` has ALREADY
confirmed the value occurs in the passage before this is reached -- the
grounding guarantee is upstream of the type, not enforced by it. A caller that
wants arithmetic asks for QUANTITY, which still refuses a non-number.
"""

from __future__ import annotations

import pytest

from app.agent_core.facts.prose import _as_kind
from app.agent_core.facts.types import ScalarKind


class TestTextAcceptsANumber:
    @pytest.mark.parametrize(
        "raw, expected",
        [(2, "2"), (2.0, "2"), (2.5, "2.5"), (0, "0"), (155, "155")],
        ids=["int", "whole-float", "real-float", "zero", "large"],
    )
    def test_a_number_becomes_its_text(self, raw, expected: str) -> None:
        assert _as_kind(raw, ScalarKind.TEXT) == expected

    def test_a_whole_float_loses_its_trailing_zero(self) -> None:
        """"2 English courses" reads properly; "2.0 English courses" does not."""
        assert _as_kind(2.0, ScalarKind.TEXT) == "2"

    def test_a_string_is_still_passed_through(self) -> None:
        assert _as_kind("2 English-language courses", ScalarKind.TEXT) == (
            "2 English-language courses"
        )


class TestItStaysNarrow:
    @pytest.mark.parametrize("raw", [True, False], ids=["true", "false"])
    def test_a_bool_is_still_refused(self, raw: bool) -> None:
        """"True" is not what anyone reading a passage meant, and bool is an int
        in Python -- so it has to be excluded on purpose."""
        assert _as_kind(raw, ScalarKind.TEXT) is None

    @pytest.mark.parametrize("raw", [None, ["a"], {"a": 1}], ids=["none", "list", "dict"])
    def test_a_structure_is_still_refused(self, raw) -> None:
        assert _as_kind(raw, ScalarKind.TEXT) is None


class TestTheOtherKindsAreUntouched:
    def test_quantity_still_refuses_prose(self) -> None:
        assert _as_kind("ninety", ScalarKind.QUANTITY) is None

    def test_quantity_still_refuses_a_bool(self) -> None:
        assert _as_kind(True, ScalarKind.QUANTITY) is None

    def test_quantity_still_reads_a_numeric_string(self) -> None:
        assert _as_kind("2", ScalarKind.QUANTITY) == 2.0

    def test_bool_still_reads_yes_and_no(self) -> None:
        assert _as_kind("yes", ScalarKind.BOOL) is True
        assert _as_kind("no", ScalarKind.BOOL) is False
