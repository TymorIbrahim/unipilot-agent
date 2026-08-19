"""The time budget must bound when the run ENDS, not when its last turn starts.

Checked only as `elapsed >= budget`, the bound says nothing about the turn it
lets through. A live "how many semesters will it take me to graduate" run began
a turn at 235s of a 240s budget and returned at 267s.

That overrun is the one failure that escapes every other guarantee here: the
platform kills the request at 300s, and a killed request answers with the
platform's error rather than the four fields `/api/execute` promises on both
paths. Everything else this loop does -- refusing an ungrounded number, handing
back a violated post-condition -- assumes there is still a response to put it in.

The reserve is the longest turn this run has actually taken. Nothing else
available is better evidence for what the next one will cost, and the first turn
is always allowed because before one finishes there is nothing to go on.
"""

from __future__ import annotations

import time

import pytest

from app.agent_core.facts.answer import HeldFact
from app.agent_core.facts.dispatch import DispatchContext
from app.agent_core.facts.loop import run_loop
from app.agent_core.facts.types import Basis, Scalar, ScalarKind


def _context() -> DispatchContext:
    """One held fact, because an answer standing on nothing is refused -- which
    is correct, and unrelated to what these tests measure."""
    return DispatchContext(
        facts={"me": HeldFact(value=Scalar(ScalarKind.IDENTIFIER, "student-1"),
                              basis=Basis.OFFICIAL_RECORD)}
    )


class _SlowModel:
    """Answers only after `answer_on` turns, each taking `seconds`."""

    def __init__(self, seconds: float, answer_on: int = 99) -> None:
        self.seconds = seconds
        self.answer_on = answer_on
        self.turns = 0

    async def respond(self, prompt: str):
        self.turns += 1
        time.sleep(self.seconds)  # deliberate: wall clock is what the budget reads
        if self.turns >= self.answer_on:
            return {"answer": "done for {me}."}
        return {"calls": [{"tool": "find", "as": f"x{self.turns}",
                           "args": {"source": "courses"}}]}


async def _run(model, budget: float):
    started = time.monotonic()
    result = await run_loop("q", model, _context(), max_turns=20, time_budget_s=budget)
    return result, time.monotonic() - started


class TestTheRunFinishesInsideTheWindow:
    async def test_it_does_not_start_a_turn_it_cannot_finish(self) -> None:
        """Turns of 0.2s against a 0.5s budget: without the reserve a third turn
        starts at 0.4s and returns at 0.6s, past the budget."""
        result, elapsed = await _run(_SlowModel(0.2), budget=0.5)
        assert elapsed < 0.5 + 0.2, f"overran: {elapsed:.2f}s against a 0.5s budget"
        assert result.outcome == "exhausted"

    async def test_the_reason_names_the_reserve(self) -> None:
        result, _ = await _run(_SlowModel(0.2), budget=0.5)
        assert "would overrun" in (result.reason or "")

    async def test_the_first_turn_is_always_allowed(self) -> None:
        """Nothing is known about turn length before one finishes, and a budget
        that refuses to start is worse than one that overruns."""
        model = _SlowModel(0.2, answer_on=1)
        result, _ = await _run(model, budget=0.05)
        assert model.turns == 1, "the first turn must run"
        assert result.outcome == "answered"

    async def test_a_run_inside_the_budget_is_untouched(self) -> None:
        model = _SlowModel(0.01, answer_on=3)
        result, _ = await _run(model, budget=5.0)
        assert result.outcome == "answered"
        assert model.turns == 3

    async def test_the_transcript_survives(self) -> None:
        """Concluding gracefully is the whole point -- an outer cancellation
        would throw the transcript away."""
        result, _ = await _run(_SlowModel(0.2), budget=0.5)
        assert result.transcript
        assert result.transcript[-1].action == "timeout"


class TestNoBudgetMeansNoReserve:
    async def test_an_unbudgeted_run_is_bounded_only_by_turns(self) -> None:
        model = _SlowModel(0.01, answer_on=4)
        result = await run_loop("q", model, _context(), max_turns=20, time_budget_s=None)
        assert result.outcome == "answered" and model.turns == 4
