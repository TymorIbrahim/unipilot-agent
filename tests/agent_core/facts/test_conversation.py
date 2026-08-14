"""Conversation memory -- follow-ups resolve, facts stay fresh.

Two properties matter: the store round-trips, and -- the load-bearing one -- what
is carried forward is TEXT, not facts, so a follow-up cannot inherit a stale
grounded value as though it were still true.
"""

from __future__ import annotations

import pytest

from app.agent_core.facts.conversation import (
    Exchange,
    InMemoryConversations,
    MAX_TURNS_REMEMBERED,
    render_history,
)


class TestInMemoryStore:
    async def test_it_round_trips_an_exchange(self) -> None:
        store = InMemoryConversations()
        await store.append("c1", "how many credits left?", "You need 92.5.")
        history = await store.history("c1")
        assert history == [Exchange("how many credits left?", "You need 92.5.")]

    async def test_conversations_are_independent(self) -> None:
        store = InMemoryConversations()
        await store.append("a", "q-a", "ans-a")
        await store.append("b", "q-b", "ans-b")
        assert await store.history("a") == [Exchange("q-a", "ans-a")]
        assert await store.history("b") == [Exchange("q-b", "ans-b")]

    async def test_an_unknown_conversation_is_empty_not_an_error(self) -> None:
        assert await InMemoryConversations().history("never-seen") == []

    async def test_only_the_most_recent_turns_are_kept(self) -> None:
        """Bounded so a long chat cannot grow the prompt without limit."""
        store = InMemoryConversations()
        for n in range(MAX_TURNS_REMEMBERED + 4):
            await store.append("c", f"q{n}", f"a{n}")
        history = await store.history("c")
        assert len(history) == MAX_TURNS_REMEMBERED
        assert history[-1] == Exchange(f"q{MAX_TURNS_REMEMBERED + 3}", f"a{MAX_TURNS_REMEMBERED + 3}")


class TestRendering:
    def test_empty_history_renders_nothing(self) -> None:
        assert render_history([]) == ""

    def test_it_marks_history_as_context_not_fact(self) -> None:
        """The load-bearing instruction: an answer quoted here is prior prose,
        not a grounded fact, and every value must be re-derived."""
        text = render_history([Exchange("plan my terms", "your track has 49 courses")])
        assert "re-derive every fact fresh" in text
        assert "plan my terms" in text and "49 courses" in text
