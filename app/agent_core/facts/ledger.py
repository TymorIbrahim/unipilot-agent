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


class MongoLedger:
    """Durable, and atomic by construction.

    The token IS the `_id`, so spending is a single `insert_one` and a duplicate
    key error means it was already spent. There is no read-then-write window for
    two concurrent attempts to slip through -- which a `find` followed by an
    `insert` would have, and which is exactly the shape of a double-submit.
    """

    def __init__(self, database: Any, collection: str = "spent_confirmations") -> None:
        self._collection = database[collection]

    async def spend(self, token: str) -> bool:
        from pymongo.errors import DuplicateKeyError

        try:
            await self._collection.insert_one({"_id": token})
            return True
        except DuplicateKeyError:
            return False


__all__ = ["ConfirmationLedger", "InMemoryLedger", "MongoLedger"]
