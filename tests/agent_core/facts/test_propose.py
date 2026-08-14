"""`propose` -- phase 8 of docs/agent/tools_implementation_plan.md.

The only tool that can change the world, and it cannot change it by itself. An
effect is not a derivation, which is why this is a primitive; a proposal is not
an execution, which is why it takes two steps.

Most of these tests assert that something does NOT happen. That is the point:
the safety property is the absence of a write, and absence is only verifiable if
you look for it deliberately.
"""

from __future__ import annotations

import dataclasses

import pytest

from app.agent_core.facts.propose import (
    Confirmation,
    Proposal,
    UnconfirmedError,
    confirm,
    execute,
    propose,
)
from app.agent_core.facts.ledger import InMemoryLedger
from app.agent_core.facts.types import Basis, Scalar, ScalarKind


@pytest.fixture
def ledger() -> InMemoryLedger:
    """A fresh ledger per test. The ledger is now an explicit dependency rather
    than module state, so isolation is a consequence of the design instead of a
    cleanup step someone has to remember."""
    return InMemoryLedger()

Q = ScalarKind.QUANTITY
I = ScalarKind.IDENTIFIER


class _SpyExecutor:
    def __init__(self):
        self.calls = []

    async def apply(self, proposal: Proposal):
        self.calls.append(proposal)
        return {"ok": True}


def _proposal(basis: Basis = Basis.OFFICIAL_RECORD) -> Proposal:
    return propose(
        action="register",
        target="00960211",
        payload={"semester": Scalar(I, "spring-2026")},
        grounds=("eligibility_check", "offering_pattern"),
        basis=basis,
    )


class TestProposingIsNotDoing:
    async def test_proposing_never_touches_the_executor(self) -> None:
        executor = _SpyExecutor()
        _proposal()
        assert executor.calls == []

    async def test_a_proposal_is_inert_data(self) -> None:
        proposal = _proposal()
        assert isinstance(proposal, Proposal)
        assert proposal.action == "register"
        assert proposal.target == "00960211"

    def test_an_ungrounded_proposal_is_refused(self) -> None:
        """An action with nothing behind it is the one case where refusing is
        obviously right: there is no fact to show the person being asked."""
        with pytest.raises(ValueError):
            propose(action="register", target="x", payload={}, grounds=(), basis=Basis.OFFICIAL_RECORD)


class TestConfirmationIsRequired:
    async def test_executing_without_a_confirmation_raises(self, ledger) -> None:
        executor = _SpyExecutor()
        with pytest.raises(UnconfirmedError):
            await execute(_proposal(), None, executor, ledger)  # type: ignore[arg-type]
        assert executor.calls == []

    async def test_a_confirmation_for_a_different_proposal_is_refused(self, ledger) -> None:
        executor = _SpyExecutor()
        mine, other = _proposal(), propose(
            action="drop", target="00940224", payload={}, grounds=("g",), basis=Basis.OFFICIAL_RECORD
        )
        with pytest.raises(UnconfirmedError):
            await execute(mine, confirm(other, by="student"), executor, ledger)
        assert executor.calls == []

    async def test_a_confirmation_does_not_survive_editing_the_proposal(self, ledger) -> None:
        """The bait-and-switch case. Confirmation binds to the CONTENT, not to a
        name -- otherwise a payload could be swapped between the moment a person
        agrees and the moment the write happens, and their agreement would still
        appear to cover it."""
        executor = _SpyExecutor()
        original = _proposal()
        confirmation = confirm(original, by="student")
        tampered = dataclasses.replace(original, payload={"semester": Scalar(I, "winter-2027")})

        with pytest.raises(UnconfirmedError):
            await execute(tampered, confirmation, executor, ledger)
        assert executor.calls == []

    async def test_a_valid_confirmation_executes_exactly_once(self, ledger) -> None:
        executor = _SpyExecutor()
        proposal = _proposal()
        await execute(proposal, confirm(proposal, by="student"), executor, ledger)
        assert len(executor.calls) == 1

    async def test_a_confirmation_cannot_be_replayed(self, ledger) -> None:
        """One agreement, one write. Without this a captured confirmation is a
        standing licence to repeat the action."""
        executor = _SpyExecutor()
        proposal = _proposal()
        confirmation = confirm(proposal, by="student")
        await execute(proposal, confirmation, executor, ledger)
        with pytest.raises(UnconfirmedError):
            await execute(proposal, confirmation, executor, ledger)
        assert len(executor.calls) == 1


class TestWhatThePersonSees:
    def test_a_proposal_states_its_grounds(self) -> None:
        """Whoever is asked to confirm has to be able to see why. A proposal
        that cannot show its reasons is asking for trust, not consent."""
        proposal = _proposal()
        assert proposal.grounds == ("eligibility_check", "offering_pattern")

    def test_a_proposal_built_on_a_hypothetical_says_so(self) -> None:
        """Registering off the back of a simulated plan is legitimate -- you
        plan, then act. But the person confirming must know the ground under it
        is a what-if, so it is flagged rather than blocked."""
        proposal = _proposal(basis=Basis.SIMULATED)
        assert proposal.speculative is True

    def test_a_proposal_from_official_records_is_not_flagged(self) -> None:
        assert _proposal(basis=Basis.OFFICIAL_RECORD).speculative is False

    def test_the_summary_names_the_action_and_target(self) -> None:
        summary = _proposal().summary()
        assert "register" in summary and "00960211" in summary
