"""`propose` -- the only write. Phase 8 of docs/agent/tools_implementation_plan.md.

An effect is not a derivation, which is why this is a primitive rather than an
operator. A proposal is not an execution, which is why it takes two steps and a
person in between.

The safety property is structural rather than procedural. `execute` cannot run
without a `Confirmation`, and a `Confirmation` is bound to the FINGERPRINT of the
exact proposal it was given -- so agreeing to one thing can never authorise
another, and an agreement cannot be replayed into a second write.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from app.agent_core.facts.ledger import ConfirmationLedger
from app.agent_core.facts.types import Basis, Scalar


class UnconfirmedError(Exception):
    """An execution was attempted without a valid, unused confirmation for
    exactly this proposal."""


class Executor(Protocol):
    async def apply(self, proposal: "Proposal") -> Any: ...


@dataclass(frozen=True)
class Proposal:
    action: str
    target: str
    payload: Mapping[str, Scalar]
    grounds: tuple[str, ...]
    basis: Basis

    @property
    def speculative(self) -> bool:
        """Is this built on a hypothetical rather than a record?

        Flagged, not blocked: acting on a simulated plan is normal -- you plan,
        then register. But whoever confirms has to know the ground under it is a
        what-if, because the action itself will be entirely real.
        """
        return self.basis is Basis.SIMULATED

    def fingerprint(self) -> str:
        """A stable digest of everything that determines what will happen.

        Confirmation binds to this rather than to a name or an id, so a payload
        cannot be swapped between the moment a person agrees and the moment the
        write lands.
        """
        parts = [self.action, self.target, self.basis.name]
        parts += [f"{name}={_render(value)}" for name, value in sorted(self.payload.items())]
        parts += sorted(self.grounds)
        return hashlib.sha256("␟".join(parts).encode("utf-8")).hexdigest()

    def summary(self) -> str:
        """What the person is being asked to agree to, in one line."""
        details = ", ".join(f"{name}={_render(value)}" for name, value in sorted(self.payload.items()))
        caveat = "  [based on a simulated plan]" if self.speculative else ""
        return f"{self.action} {self.target}" + (f" ({details})" if details else "") + caveat


@dataclass(frozen=True)
class Confirmation:
    fingerprint: str
    by: str
    at: datetime


def propose(
    action: str,
    target: str,
    payload: Mapping[str, Scalar],
    grounds: Sequence[str],
    basis: Basis,
) -> Proposal:
    """Describe an intended change. Nothing happens as a result of calling this.

    `grounds` is required. An action with no facts behind it cannot be shown to
    the person being asked, and asking someone to approve something they cannot
    inspect is asking for trust rather than consent.
    """
    if not grounds:
        raise ValueError(
            f"cannot propose to {action} {target} with no grounds: a proposal must name the facts "
            "that justify it, or the person confirming has nothing to judge."
        )
    return Proposal(
        action=action,
        target=target,
        payload=dict(payload),
        grounds=tuple(grounds),
        basis=basis,
    )


def confirm(proposal: Proposal, by: str) -> Confirmation:
    """Record a person's agreement to one specific proposal."""
    return Confirmation(fingerprint=proposal.fingerprint(), by=by, at=datetime.now(timezone.utc))


async def execute(
    proposal: Proposal,
    confirmation: Confirmation | None,
    executor: Executor,
    ledger: ConfirmationLedger,
) -> Any:
    """Apply a proposal, only with a matching and unused confirmation.

    `ledger` is required rather than defaulted. Where spent confirmations are
    recorded decides whether the one-write guarantee survives a restart, and a
    default would let that be decided by accident.
    """
    if confirmation is None:
        raise UnconfirmedError(
            f"refusing to {proposal.action} {proposal.target}: no confirmation. "
            "Every write in this system requires a person to agree to it first."
        )

    fingerprint = proposal.fingerprint()
    if confirmation.fingerprint != fingerprint:
        raise UnconfirmedError(
            f"refusing to {proposal.action} {proposal.target}: the confirmation was given for a "
            "different proposal. Agreement binds to the exact content that was shown, so an edited "
            "payload needs a fresh confirmation."
        )

    # One agreement, one write. A reusable confirmation is a standing licence to
    # repeat the action, which is not what anyone agreed to.
    #
    # Spend BEFORE applying. The other order leaves a window where the write has
    # landed but the confirmation still looks unused, so a crash there permits a
    # replay of a real effect. This order costs a re-confirmation instead.
    token = f"{fingerprint}:{confirmation.by}:{confirmation.at.isoformat()}"
    if not await ledger.spend(token):
        raise UnconfirmedError(
            f"refusing to {proposal.action} {proposal.target}: this confirmation has already been "
            "used. Confirmations authorise one write each."
        )

    return await executor.apply(proposal)


def _render(value: Any) -> str:
    return str(value.value) if isinstance(value, Scalar) else str(value)


__all__ = [
    "Confirmation",
    "Executor",
    "Proposal",
    "UnconfirmedError",
    "confirm",
    "execute",
    "propose",
]
