"""Conversation memory -- just enough to make a follow-up self-contained.

A student asks a hard question, the agent does what it can and says "if you want,
I can take the next step..."; the student replies "yes, continue". For that to
work the follow-up run has to know what "continue" REFERS to.

What is stored is deliberately narrow: the prior QUESTIONS and ANSWERS, as text.
Not the derived facts. Re-injecting a fact grounded in an earlier run as though
it were still true is the memory-contamination failure this whole layer is built
to avoid -- a grade or a course list can change between two requests, and a
stale one wearing the shape of a fresh fact is worse than none. So history is
CONTEXT the model reads to interpret the follow-up, and every fact is still
re-derived and re-grounded from the student's live records each turn.

Two implementations, the same split as the confirmation ledger: in-process for a
test, durable for production, chosen explicitly rather than defaulted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

MAX_TURNS_REMEMBERED = 6
"""How many prior exchanges a follow-up carries. Enough to continue a line of
questioning; bounded so the prompt cannot grow without limit across a long chat."""


@dataclass(frozen=True)
class Exchange:
    """One prior question and the answer that was given to it."""

    question: str
    answer: str


class ConversationStore(Protocol):
    async def history(self, conversation_id: str) -> list[Exchange]:
        """The prior exchanges of this conversation, oldest first."""
        ...

    async def append(self, conversation_id: str, question: str, answer: str) -> None:
        """Record one more exchange."""
        ...


class InMemoryConversations:
    """Non-durable. Correct within one process, forgotten on restart.

    Named plainly so adopting it for production is a decision, not an accident --
    a restart loses every conversation, so "continue" would stop resolving.
    """

    def __init__(self) -> None:
        self._by_id: dict[str, list[Exchange]] = {}

    async def history(self, conversation_id: str) -> list[Exchange]:
        return list(self._by_id.get(conversation_id, ()))[-MAX_TURNS_REMEMBERED:]

    async def append(self, conversation_id: str, question: str, answer: str) -> None:
        self._by_id.setdefault(conversation_id, []).append(Exchange(question, answer))


class SupabaseConversations:
    """Durable. A follow-up resolves after a restart because the exchanges live
    in Postgres.

    A ROW PER EXCHANGE, where `MongoConversations` pushed onto an array inside
    one document per conversation. Two things improve by it: appending is a plain
    INSERT with no read-modify-write, so two overlapping requests on one thread
    cannot lose an exchange the way `$push` on a stale document could; and
    `history` asks for its six rows in the database rather than fetching a
    conversation of unbounded length to slice the tail off it.

    `seq` is an identity column, so ordering is the insert order rather than
    anything derived from a clock -- two exchanges appended in the same
    millisecond still come back in the order they happened.
    """

    def __init__(self, database: Any, table: str = "agent_conversations") -> None:
        self._database = database
        self._table = table

    async def history(self, conversation_id: str) -> list[Exchange]:
        # Newest-first with a LIMIT, then reversed: the tail is what
        # MAX_TURNS_REMEMBERED means, and taking it in SQL keeps a long thread
        # from crossing the wire to be thrown away.
        rows = await self._database.fetch(
            f"select question, answer from {self._table} "
            "where conversation_id = $1 order by seq desc limit $2",
            conversation_id,
            MAX_TURNS_REMEMBERED,
        )
        return [Exchange(str(row["question"]), str(row["answer"])) for row in reversed(rows)]

    async def append(self, conversation_id: str, question: str, answer: str) -> None:
        await self._database.execute(
            f"insert into {self._table} (conversation_id, question, answer) values ($1, $2, $3)",
            conversation_id,
            question,
            answer,
        )


def render_history(history: list[Exchange]) -> str:
    """The conversation-so-far block for the turn prompt, or empty when new.

    Marked clearly as PRIOR context, not fact: the model may read it to resolve
    a follow-up like "continue", but it must still derive every value fresh --
    an answer quoted here is last turn's prose, not a grounded fact.
    """
    if not history:
        return ""
    lines = ["CONVERSATION SO FAR (prior turns, for context only -- re-derive every fact fresh):"]
    for exchange in history:
        lines.append(f"  Student asked: {exchange.question}")
        lines.append(f"  You answered:  {exchange.answer}")
    return "\n".join(lines)


__all__ = [
    "ConversationStore",
    "Exchange",
    "InMemoryConversations",
    "MAX_TURNS_REMEMBERED",
    "SupabaseConversations",
    "render_history",
]
