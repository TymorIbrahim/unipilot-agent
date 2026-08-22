"""A tool that offered an option its own validator had to reject.

`interpret` accepted every `ScalarKind`, `bool` among them, and listed it as
available in the error text. Every extracted value must APPEAR in the passage it
cites -- that is the whole grounding of the prose side -- and "True" appears in
no regulation. So a model asking a page a yes/no question got:

    interpretation of 'regulations-undergraduate' returned True, which does not
    appear in the passage

19 times across the measured runs: the largest single cause of wasted turns
after the wrong-KIND refusal, and structurally unwinnable. "Can I retake a
course I already passed?" is the natural shape for a bool, and it could never
succeed.

A yes/no about a page is a JUDGEMENT, not an extraction. The phrase is what the
page holds; the verdict is what the answer says around it, which the grounding
rules already allow -- only numbers must be slots.
"""

from __future__ import annotations

import asyncio

import pytest

from app.agent_core.facts.dispatch import DispatchContext, dispatch
from app.agent_core.facts.prose import Passage


class _Extractor:
    async def extract(self, *args, **kwargs):
        return ("x", "x")

    async def extract_all(self, *args, **kwargs):
        return [("x", "x")]


def _context() -> DispatchContext:
    return DispatchContext(
        extractor=_Extractor(),
        passages={"regs": Passage(slug="regs", title="R", excerpt="some text", score=1.0)},
    )


def _call(tool: str, expect: str):
    return asyncio.run(dispatch(
        {"tool": tool, "as": "x",
         "args": {"slug": "regs", "question": "can I retake?", "expect": expect}},
        _context()))


@pytest.mark.parametrize("tool", ["interpret", "extract_list"])
class TestBoolIsRefusedWithAReason:
    def test_it_is_a_defect(self, tool: str) -> None:
        assert _call(tool, "bool").defects

    def test_it_explains_why_it_could_never_work(self, tool: str) -> None:
        message = list(_call(tool, "bool").defects.values())[0].message
        assert "must APPEAR in the passage" in message

    def test_it_names_what_to_do_instead(self, tool: str) -> None:
        message = list(_call(tool, "bool").defects.values())[0].message
        assert 'expect "text"' in message
        assert "say yes or no in your own words" in message


class TestTheUsableKindsStillWork:
    @pytest.mark.parametrize("expect", ["text", "quantity", "identifier", "date"])
    def test_the_kind_itself_is_accepted(self, expect: str) -> None:
        """A quantity or date may still fail on the VALUE -- the stub returns
        "x" -- but never on the KIND being unavailable."""
        outcome = _call("interpret", expect)
        for defect in outcome.defects.values():
            assert "is not available here" not in defect.message

    def test_bool_is_no_longer_listed_as_available(self) -> None:
        """The old message advertised the trap in the act of reporting it."""
        message = list(_call("interpret", "nonsense").defects.values())[0].message
        assert "bool" not in message
        assert "text" in message and "quantity" in message
