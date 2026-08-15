"""A proposal with no grounds must say so in words the model can act on.

`propose` raises a written explanation for empty grounds and the dispatcher
catches it -- but the basis passed to it is `min()` over those same grounds, and
arguments are evaluated before the call. So the model received
"min() iterable argument is empty" and the explanation was unreachable.

Defect messages are the model's only route out of its own mistake: the loop
feeds them back as observations, and one that names no fix buys a repeated call.
"""

from __future__ import annotations

from app.agent_core.facts.answer import HeldFact
from app.agent_core.facts.dispatch import DispatchContext, dispatch
from app.agent_core.facts.types import Basis, Scalar, ScalarKind


def _context() -> DispatchContext:
    return DispatchContext(
        facts={
            "prereqs_met": HeldFact(
                value=Scalar(ScalarKind.BOOL, True), basis=Basis.OFFICIAL_RECORD
            )
        }
    )


class TestEmptyGrounds:
    async def test_it_explains_rather_than_leaking_a_python_error(self) -> None:
        result = await dispatch(
            {"tool": "propose", "as": "reg",
             "args": {"action": "register", "target": "00960211", "grounds": []}},
            _context(),
        )
        message = result.defects["reg"].message
        assert "min()" not in message, "a Python internal is not something the model can act on"
        assert "grounds" in message
        assert result.proposal is None

    async def test_the_message_names_the_action_it_refused(self) -> None:
        result = await dispatch(
            {"tool": "propose", "as": "reg",
             "args": {"action": "register", "target": "00960211", "grounds": []}},
            _context(),
        )
        assert "00960211" in result.defects["reg"].message


class TestGroundsMustBeHeld:
    async def test_an_unheld_ground_is_refused_and_the_held_ones_listed(self) -> None:
        result = await dispatch(
            {"tool": "propose", "as": "reg",
             "args": {"action": "register", "target": "00960211", "grounds": ["imaginary"]}},
            _context(),
        )
        assert "imaginary" in result.defects["reg"].message
        assert "prereqs_met" in result.defects["reg"].message, "say what IS available"
        assert result.proposal is None


class TestAValidProposal:
    async def test_it_is_described_and_nothing_happens(self) -> None:
        result = await dispatch(
            {"tool": "propose", "as": "reg",
             "args": {"action": "register", "target": "00960211",
                      "grounds": ["prereqs_met"], "payload": {"semester": "spring-2026"}}},
            _context(),
        )
        assert result.proposal is not None
        assert "register 00960211" in result.proposal.summary()
        assert result.proposal.basis is Basis.OFFICIAL_RECORD
        assert not result.proposal.speculative
