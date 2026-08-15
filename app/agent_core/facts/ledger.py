"""Confirmation ledgers -- durability for `propose`'s one-write guarantee.

A confirmation authorises exactly one write. Enforcing that needs somewhere to
record that it has been used, and where that record lives decides whether the
guarantee actually holds:

  in process   a restart forgets every spent confirmation, so a captured one
               becomes replayable. Narrows the window; does not close it.
  durable      survives restarts, and -- because the check and the record are a
               single atomic insert -- survives two concurrent attempts too.

The ordering in `execute` is spend-then-apply, deliberately. If the write landed
first and the ledger entry second, a crash between them would leave a spent
confirmation looking unused. Spending first means a crash costs the user a
re-confirmation, which is the failure worth having.
"""

from __future__ import annotations

from typing import Any, Protocol


class ConfirmationLedger(Protocol):
    async def spend(self, token: str) -> bool:
        """Record `token` as used. True if this call was the one that spent it."""
        ...


class InMemoryLedger:
    """Non-durable. Correct within one process and worthless across a restart.

    Named plainly rather than treated as the default, so choosing it is a
    decision someone made rather than one that happened to them.
    """

    def __init__(self) -> None:
        self._spent: set[str] = set()

    async def spend(self, token: str) -> bool:
        if token in self._spent:
            return False
        self._spent.add(token)
        return True

    def clear(self) -> None:
        self._spent.clear()


class SupabaseLedger:
    """Durable, and atomic by construction.

    The token IS the primary key, so spending is a single
    `insert ... on conflict do nothing` and the DATABASE arbitrates: exactly one
    of two concurrent attempts inserts a row, and `rowcount` says which. There is
    no read-then-write window for both to slip through -- which a select followed
    by an insert would have, and which is exactly the shape of a double-submit.

    Replaces `MongoLedger`, which spent a token with `insert_one` and caught
    `DuplicateKeyError`. Same guarantee, same single round trip; the only reason
    it could not stay is that it imported `pymongo`, a package this deployment
    does not carry, so the one place the guarantee lived would have raised
    ImportError the first time a confirmation was ever spent.
    """

    def __init__(self, database: Any, table: str = "spent_confirmations") -> None:
        self._database = database
        self._table = table

    async def spend(self, token: str) -> bool:
        # `execute` returns Postgres' command tag -- "INSERT 0 1" when the row
        # was written, "INSERT 0 0" when the token was already spent. Reading the
        # count is what makes this a test AND a write in one statement.
        tag = await self._database.execute(
            f"insert into {self._table} (token) values ($1) on conflict (token) do nothing",
            token,
        )
        return str(tag).strip().endswith("1")


__all__ = ["ConfirmationLedger", "InMemoryLedger", "SupabaseLedger"]
