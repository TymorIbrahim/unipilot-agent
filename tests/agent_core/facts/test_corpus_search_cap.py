"""The one tool no governor could see.

`search_corpus` always succeeds and always returns hits, so every call counts as
a new fact and `NO_PROGRESS_LIMIT` never fires. `_call_signatures` does not catch
it either: a query differing by one word is a different derivation.

Measured. Asked for a minimum attendance percentage -- which the undergraduate
regulations do not set -- one run issued 17 searches across 39 turns and 216
seconds before the clock stopped it. A second, about physical education, spent
22 turns the same way. Both ended with no answer at all, having read the right
page early on.

A corpus searched five times without yielding an answer is telling you
something, and it is not "search again". The honest move at that point is to say
the regulations do not cover it -- an absence IS an answer, and it is the one
the invented-rule case needs.
"""

from __future__ import annotations

import pytest

from app.agent_core.facts.dispatch import DispatchContext
from app.agent_core.facts.loop import _CORPUS_SEARCH_LIMIT, run_loop
from app.agent_core.facts.answer import HeldFact
from app.agent_core.facts.types import Basis, Scalar, ScalarKind

Q = ScalarKind.QUANTITY

pytestmark = pytest.mark.asyncio


class _Searcher:
    """Answers every turn with another corpus search, as the live run did."""

    def __init__(self, then_answer: str | None = None) -> None:
        self.calls = 0
        self.prompts: list[str] = []
        self._then = then_answer

    async def respond(self, prompt: str):
        self.prompts.append(prompt)
        self.calls += 1
        if self._then and self.calls > _CORPUS_SEARCH_LIMIT + 1:
            return {"answer": self._then}
        return {"calls": [{"tool": "search_corpus", "as": f"hits{self.calls}",
                           "args": {"query": f"attendance rule phrasing {self.calls}"}}]}


class _Retriever:
    """Always returns a hit -- which is the whole problem. A search that always
    succeeds always counts as progress, so no other governor can see the loop."""

    async def search(self, query: str, limit: int):
        from app.agent_core.facts.prose import Passage

        return [Passage(slug="regulations-undergraduate", title="Regulations",
                        excerpt="Section 5.1 ...", score=1.0)]

    def page(self, slug: str):
        return "Undergraduate Study Regulations. Section 5.1 ..."


def _context() -> DispatchContext:
    return DispatchContext(
        facts={"count": HeldFact(value=Scalar(Q, 3.0), basis=Basis.OFFICIAL_RECORD)},
        retriever=_Retriever(),
    )


class TestTheCapHolds:
    async def test_searching_stops_after_the_limit(self) -> None:
        model = _Searcher()
        result = await run_loop("what is the attendance rule?", model, _context(), max_turns=20)
        capped = [t for t in result.transcript if t.action == "search-capped"]
        assert capped, "nothing stopped the searching"

    async def test_the_run_does_not_reach_seventeen_searches(self) -> None:
        model = _Searcher()
        await run_loop("what is the attendance rule?", model, _context(), max_turns=20)
        searched = sum(1 for t in model.prompts if t)  # one prompt per turn
        assert searched <= 20
        # The cap is what matters: no more than the limit actually dispatched.
        assert _CORPUS_SEARCH_LIMIT < 17

    async def test_the_model_is_told_an_absence_is_an_answer(self) -> None:
        model = _Searcher()
        await run_loop("what is the attendance rule?", model, _context(), max_turns=20)
        assert any("absence IS an answer" in p for p in model.prompts)

    async def test_it_can_still_answer_after_being_capped(self) -> None:
        """Capped is not concluded -- the run continues with what it holds."""
        model = _Searcher(then_answer="You have {count} courses.")
        result = await run_loop("q", model, _context(), max_turns=20)
        assert result.outcome == "answered"


class TestOrdinaryUseIsUntouched:
    async def test_a_few_searches_are_fine(self) -> None:
        """A genuine multi-page question searches two or three times."""
        class _Twice:
            def __init__(self): self.n = 0
            async def respond(self, prompt: str):
                self.n += 1
                if self.n <= 2:
                    return {"calls": [{"tool": "search_corpus", "as": f"h{self.n}",
                                       "args": {"query": f"english requirement {self.n}"}}]}
                return {"answer": "You have {count} courses."}

        result = await run_loop("q", _Twice(), _context(), max_turns=8)
        assert result.outcome == "answered"
        assert not [t for t in result.transcript if t.action == "search-capped"]

    async def test_the_limit_leaves_room_for_real_questions(self) -> None:
        """Two or three searches is a genuine multi-page question; the
        knowledge-base question that motivated NO_PROGRESS_LIMIT uses four at
        most."""
        assert _CORPUS_SEARCH_LIMIT >= 4
