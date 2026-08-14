"""Confirmation ledgers -- does the one-write guarantee actually survive?

`propose` refuses a replayed confirmation. Whether that refusal holds depends
entirely on where spent confirmations are recorded, so the two properties worth
testing are the two that in-process memory does NOT have:

  - it survives a restart
  - two concurrent attempts cannot both win

The Mongo tests skip loudly when no database is reachable, because a green run
that never checked durability says nothing about durability.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from tests.agent_core.ise_student_fixture import (  # noqa: F401 -- autouse fixture
    _fresh_mongo_client_per_test,
)
from motor.motor_asyncio import AsyncIOMotorClient

from app.agent_core.facts.ledger import InMemoryLedger, MongoLedger


class TestInMemoryLedger:
    async def test_a_token_is_spendable_once(self) -> None:
        ledger = InMemoryLedger()
        assert await ledger.spend("t") is True
        assert await ledger.spend("t") is False

    async def test_different_tokens_are_independent(self) -> None:
        ledger = InMemoryLedger()
        assert await ledger.spend("a") is True
        assert await ledger.spend("b") is True

    async def test_it_does_not_survive_a_restart(self) -> None:
        """Documenting the limitation as a TEST rather than a comment, so nobody
        adopts it for production believing otherwise."""
        first = InMemoryLedger()
        await first.spend("t")
        restarted = InMemoryLedger()
        assert await restarted.spend("t") is True, "in-memory ledgers forget; this is why they are not durable"


@pytest.fixture
async def database():
    # Through the agent's own settings, not a bespoke env var.
    from app.db.mongo import get_database

    try:
        handle = await get_database()
        await handle.command("ping")
        client = handle.client
    except Exception as exc:  # noqa: BLE001
        pytest.skip(
            f"NOT VERIFIED: no database ({type(exc).__name__}). "
            "Confirmation-ledger DURABILITY and atomicity are UNCHECKED in this run."
        )
    db = client["unipilot_ledger_test"]
    await db["spent_confirmations"].delete_many({})
    yield db
    await client.drop_database("unipilot_ledger_test")
    client.close()


class TestMongoLedger:
    async def test_a_token_is_spendable_once(self, database) -> None:
        ledger = MongoLedger(database)
        assert await ledger.spend("t") is True
        assert await ledger.spend("t") is False

    async def test_it_survives_a_restart(self, database) -> None:
        """The property in-memory cannot have. A fresh ledger object against the
        same store is exactly what a process restart looks like."""
        await MongoLedger(database).spend("t")
        restarted = MongoLedger(database)
        assert await restarted.spend("t") is False

    async def test_concurrent_spends_produce_exactly_one_winner(self, database) -> None:
        """The double-submit shape. A read-then-write ledger has a window where
        both attempts see 'unspent' and both proceed; making the token the
        primary key closes it, because the database itself does the arbitration.
        """
        ledger = MongoLedger(database)
        outcomes = await asyncio.gather(*(ledger.spend("same") for _ in range(12)))
        assert sum(1 for won in outcomes if won) == 1

    async def test_distinct_tokens_all_succeed(self, database) -> None:
        ledger = MongoLedger(database)
        outcomes = await asyncio.gather(*(ledger.spend(f"t{n}") for n in range(8)))
        assert all(outcomes)


class TestEndToEndWithPropose:
    async def test_a_replay_is_refused_across_a_restart(self, database) -> None:
        """The whole point, end to end: agreement is spent once, and a process
        restart does not hand it back."""
        from app.agent_core.facts.propose import UnconfirmedError, confirm, execute, propose
        from app.agent_core.facts.types import Basis, Scalar, ScalarKind

        class _Spy:
            def __init__(self):
                self.calls = []

            async def apply(self, proposal):
                self.calls.append(proposal)

        executor = _Spy()
        proposal = propose(
            action="register",
            target="00960211",
            payload={"semester": Scalar(ScalarKind.IDENTIFIER, "spring-2026")},
            grounds=("eligibility",),
            basis=Basis.OFFICIAL_RECORD,
        )
        confirmation = confirm(proposal, by="student")

        await execute(proposal, confirmation, executor, MongoLedger(database))
        with pytest.raises(UnconfirmedError):
            await execute(proposal, confirmation, executor, MongoLedger(database))

        assert len(executor.calls) == 1


class TestMongoConversations:
    """Durability, the same property the confirmation ledger needs: a follow-up
    must resolve after a restart, so the exchanges live in Mongo."""

    async def test_it_survives_a_restart(self, database) -> None:
        from app.agent_core.facts.conversation import Exchange, MongoConversations

        await MongoConversations(database).append("c1", "how many left?", "You need 92.5.")
        # A fresh store object against the same store is what a restart looks like.
        restarted = MongoConversations(database)
        assert await restarted.history("c1") == [Exchange("how many left?", "You need 92.5.")]

    async def test_appends_accumulate_in_order(self, database) -> None:
        from app.agent_core.facts.conversation import MongoConversations

        store = MongoConversations(database)
        await store.append("c2", "q1", "a1")
        await store.append("c2", "q2", "a2")
        history = await store.history("c2")
        assert [e.question for e in history] == ["q1", "q2"]
