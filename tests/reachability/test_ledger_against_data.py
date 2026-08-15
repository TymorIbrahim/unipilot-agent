"""Does the one-write guarantee actually survive? Asked of the real database.

`propose` is the only tool that can change anything, and it changes nothing until
a confirmation is spent. Whether a replayed confirmation is refused depends
entirely on where spent tokens are recorded, so the two properties worth testing
are the two in-process memory does NOT have:

  - it survives a restart
  - two concurrent attempts cannot both win

Ported from `tests/pending_supabase/test_ledger.py`, which asked Mongo the same
questions. `MongoLedger` spent a token with `insert_one` and caught
`DuplicateKeyError`; `SupabaseLedger` does `insert ... on conflict do nothing`
and reads the command tag. Same guarantee, same single round trip, and the same
reason it works: the token IS the primary key, so the DATABASE arbitrates and
there is no read-then-write window for a double-submit to slip through.

Until this ran, `spent_confirmations` held 0 rows -- the guarantee had never been
exercised against Postgres at all.

WRITES TO LIVE TABLES. `spent_confirmations` and `agent_conversations` are
runtime state, not seed data: `scripts/seed.py` deliberately never touches them,
and `agent_conversations` holds real exchanges from the deployed GUI. So there is
no test database to drop -- every row here is namespaced with a per-run prefix
and deleted by prefix afterwards. Never `delete_many({})`, which is what the
Mongo version could afford against a throwaway database and this cannot.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from app.agent_core.facts.conversation import Exchange, SupabaseConversations
from app.agent_core.facts.ledger import InMemoryLedger, SupabaseLedger
from app.db.postgres import get_database

pytestmark = pytest.mark.supabase

PREFIX = "pytest-ledger-"


def _token() -> str:
    """Unique per call, so a crashed run never poisons the next one."""
    return f"{PREFIX}{uuid.uuid4()}"


@pytest.fixture
async def database():
    db = await get_database()
    # A prefix is not enough on its own. `TestEndToEndWithPropose` spends tokens
    # minted by `confirm()`, which knows nothing about this fixture, and the
    # first run of this suite left two of them behind in a live table. So the
    # rows created during the test are diffed out, and the prefix is kept as the
    # belt-and-braces path for a run that crashes before the teardown.
    before = {
        row["token"] for row in await db.fetch("select token from spent_confirmations")
    }
    yield db
    after = {row["token"] for row in await db.fetch("select token from spent_confirmations")}
    created = sorted(after - before)
    if created:
        await db.execute("delete from spent_confirmations where token = any($1::text[])", created)
    await db.execute("delete from spent_confirmations where token like $1", f"{PREFIX}%")
    await db.execute("delete from agent_conversations where conversation_id like $1", f"{PREFIX}%")


class TestInMemoryLedger:
    """Kept from the Mongo suite unchanged -- it needs no database, and the third
    test is the reason the durable one exists."""

    async def test_a_token_is_spendable_once(self) -> None:
        ledger = InMemoryLedger()
        assert await ledger.spend("t") is True
        assert await ledger.spend("t") is False

    async def test_different_tokens_are_independent(self) -> None:
        ledger = InMemoryLedger()
        assert await ledger.spend("a") is True
        assert await ledger.spend("b") is True

    async def test_it_does_not_survive_a_restart(self) -> None:
        """The limitation as a TEST rather than a comment, so nobody adopts it
        for production believing otherwise."""
        first = InMemoryLedger()
        await first.spend("t")
        restarted = InMemoryLedger()
        assert await restarted.spend("t") is True, (
            "in-memory ledgers forget; this is why they are not durable"
        )


class TestSupabaseLedger:
    async def test_a_token_is_spendable_once(self, database) -> None:
        ledger = SupabaseLedger(database)
        token = _token()
        assert await ledger.spend(token) is True
        assert await ledger.spend(token) is False

    async def test_it_survives_a_restart(self, database) -> None:
        """The property in-memory cannot have. A fresh ledger object against the
        same store is exactly what a process restart looks like."""
        token = _token()
        await SupabaseLedger(database).spend(token)
        assert await SupabaseLedger(database).spend(token) is False

    async def test_concurrent_spends_produce_exactly_one_winner(self, database) -> None:
        """The double-submit shape, and the only test here that would pass a
        read-then-write implementation only by luck: both attempts see "unspent"
        and both proceed. Making the token the primary key closes the window
        because the database arbitrates rather than the process.
        """
        ledger = SupabaseLedger(database)
        token = _token()
        outcomes = await asyncio.gather(*(ledger.spend(token) for _ in range(12)))
        assert sum(1 for won in outcomes if won) == 1, (
            f"{sum(outcomes)} of 12 concurrent spends won; exactly one may"
        )

    async def test_distinct_tokens_all_succeed(self, database) -> None:
        ledger = SupabaseLedger(database)
        outcomes = await asyncio.gather(*(ledger.spend(_token()) for _ in range(8)))
        assert all(outcomes)

    async def test_the_row_is_really_there(self, database) -> None:
        """Guards the command-tag read. `spend` returns True from parsing
        "INSERT 0 1", so a bug in that parse looks identical to a working
        ledger until the replay -- check the row, not just the return."""
        token = _token()
        await SupabaseLedger(database).spend(token)
        assert await database.fetchval(
            "select count(*) from spent_confirmations where token = $1", token
        ) == 1


class TestEndToEndWithPropose:
    async def test_a_replay_is_refused_across_a_restart(self, database) -> None:
        """The whole point, end to end: an agreement is spent once, and a process
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

        await execute(proposal, confirmation, executor, SupabaseLedger(database))
        with pytest.raises(UnconfirmedError):
            await execute(proposal, confirmation, executor, SupabaseLedger(database))

        assert len(executor.calls) == 1, "the action was applied twice"

    async def test_a_replayed_confirmation_is_not_applied_concurrently(self, database) -> None:
        """The double-submit at the level a student would cause it: the same
        confirmation posted twice at once, not sequentially."""
        from app.agent_core.facts.propose import confirm, execute, propose
        from app.agent_core.facts.types import Basis, Scalar, ScalarKind

        class _Spy:
            def __init__(self):
                self.calls = []

            async def apply(self, proposal):
                self.calls.append(proposal)

        executor = _Spy()
        proposal = propose(
            action="register",
            target="00960212",
            payload={"semester": Scalar(ScalarKind.IDENTIFIER, "spring-2026")},
            grounds=("eligibility",),
            basis=Basis.OFFICIAL_RECORD,
        )
        confirmation = confirm(proposal, by="student")
        ledger = SupabaseLedger(database)

        await asyncio.gather(
            *(execute(proposal, confirmation, executor, ledger) for _ in range(6)),
            return_exceptions=True,
        )
        assert len(executor.calls) == 1, (
            f"applied {len(executor.calls)} times from six concurrent confirmations"
        )


class TestConversationsAreDurable:
    """The same property the ledger needs, for a different reason: a follow-up
    must resolve after a restart, so exchanges live in Postgres rather than in
    the process that happened to serve the first turn."""

    async def test_it_survives_a_restart(self, database) -> None:
        cid = f"{PREFIX}{uuid.uuid4()}"
        await SupabaseConversations(database).append(cid, "how many left?", "You need 25.5.")
        restarted = SupabaseConversations(database)
        assert await restarted.history(cid) == [Exchange("how many left?", "You need 25.5.")]

    async def test_appends_accumulate_in_order(self, database) -> None:
        cid = f"{PREFIX}{uuid.uuid4()}"
        store = SupabaseConversations(database)
        await store.append(cid, "q1", "a1")
        await store.append(cid, "q2", "a2")
        assert [e.question for e in await store.history(cid)] == ["q1", "q2"]

    async def test_order_holds_within_one_millisecond(self, database) -> None:
        """`seq` is an identity column precisely so ordering is insert order and
        not a clock. Appends inside one millisecond used to be orderable only by
        luck."""
        cid = f"{PREFIX}{uuid.uuid4()}"
        store = SupabaseConversations(database)
        for n in range(6):
            await store.append(cid, f"q{n}", f"a{n}")
        assert [e.question for e in await store.history(cid)] == [f"q{n}" for n in range(6)]

    async def test_threads_do_not_bleed_into_each_other(self, database) -> None:
        store = SupabaseConversations(database)
        mine, theirs = f"{PREFIX}{uuid.uuid4()}", f"{PREFIX}{uuid.uuid4()}"
        await store.append(mine, "mine", "a")
        await store.append(theirs, "theirs", "b")
        assert [e.question for e in await store.history(mine)] == ["mine"]

    async def test_an_unknown_thread_is_empty_not_an_error(self, database) -> None:
        assert await SupabaseConversations(database).history(f"{PREFIX}never-used") == []
