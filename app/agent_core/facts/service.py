"""The facts layer's production entry point.

`/advise` calls this. It is the one place the fact/tool loop is assembled for a
real request: a `DispatchContext` wired from the running settings, the asking
student's identity seeded as the `me` fact, the chat adapter built, and the
loop run under a turn budget.

Everything the route needs to answer is DERIVED here from the loop's own result
-- the answer text, its confidence, the course codes it grounded, the outcome
status -- so the HTTP layer stays a thin shape-translation and never reaches
into the working set itself.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.agent_core.facts.adapter import build_adapter
from app.agent_core.facts.answer import HeldFact
from app.agent_core.facts.conversation import MongoConversations
from app.agent_core.facts.loop import MAX_TURNS, LoopResult, run_loop
from app.agent_core.facts.types import Basis, Scalar, ScalarKind
from app.agent_core.facts.wiring import build_context
from app.agent_core.loop.course_names import course_codes_in, course_display_name

# Outcome (facts loop) -> the frontend's retrieval_agent.status vocabulary.
# A declined or proposed question IS a completed, valid response: the system
# answered by declining an out-of-scope ask or by preparing a change for
# approval. A refusal or a spent budget did NOT answer, and says so.
_STATUS_BY_OUTCOME = {
    "answered": "succeeded",
    "declined": "succeeded",
    "proposed": "succeeded",
    "refused": "incomplete",
    "stalled": "incomplete",
    "exhausted": "incomplete",
}

# When the loop could not answer, the reason is diagnostic prose meant for a
# developer, not a student. This is what the student sees instead.
_COULD_NOT_ANSWER = (
    "I wasn't able to work that out from your records with confidence. Could you rephrase it, "
    "or ask about something more specific?"
)


@dataclass(frozen=True)
class Advice:
    """What the route needs, all derived from the loop result."""

    answer: str
    confidence: str
    course_ids: list[str]
    status: str
    sources: list[str]
    outcome: str


async def run_advice(
    question: str,
    user_id: str,
    *,
    settings: Any | None = None,
    on_progress: Callable[[str], None] | None = None,
    max_turns: int = MAX_TURNS,
    time_budget_s: float | None = None,
    conversation_id: str | None = None,
) -> LoopResult:
    """Run the fact loop for one student's question.

    The student's identity is the one fact GIVEN rather than derived -- the loop
    cannot ask who is asking, and every "my records" filter resolves through it.

    `time_budget_s`, when set, bounds the whole request by the wall clock and
    lets the turn count run free -- so a hard question can take as many steps as
    it needs inside the window rather than being cut off at a fixed turn count.

    `conversation_id` threads a follow-up to its predecessors: the prior
    exchanges are loaded so a message like "continue" resolves, and this run's
    answer is appended when it concludes. Only the TEXT is carried -- facts are
    re-derived fresh every run, so a follow-up is grounded in live records, not
    in a snapshot from a previous turn.
    """
    from app.db.mongo import get_database

    adapter = build_adapter(settings=settings)
    if adapter is None:
        # No credentials. A loop with no model cannot run; surface it as an
        # honest non-answer rather than crashing the route.
        return LoopResult(outcome="exhausted", reason="no language model is configured")

    database = await get_database()
    context = build_context(database, settings)
    context.facts["me"] = HeldFact(
        value=Scalar(ScalarKind.IDENTIFIER, user_id),
        basis=Basis.OFFICIAL_RECORD,
    )

    store = MongoConversations(database)
    # Scope the conversation to the ASKING student, so one student's id can never
    # load another's thread even if a client sent a guessed conversation_id.
    thread_key = f"{user_id}:{conversation_id}" if conversation_id else None
    history = await store.history(thread_key) if thread_key else []

    # With a wall-clock budget the turn count must not be the thing that stops a
    # run first, so raise it out of the way and let the clock govern.
    turns = max(max_turns, 100) if time_budget_s is not None else max_turns
    result = await run_loop(
        question,
        adapter,
        context,
        max_turns=turns,
        on_progress=on_progress,
        time_budget_s=time_budget_s,
        history=history,
    )

    # Record the exchange so the NEXT message can continue it. Only a real
    # student-facing answer is worth remembering; a bare non-answer would just
    # clutter the thread.
    if thread_key and result.answer is not None:
        await store.append(thread_key, question, result.answer.text)

    return result


def to_advice(result: LoopResult) -> Advice:
    """Map a loop result to the fields the route ships. Pure and total."""
    answer = _answer_text(result)
    return Advice(
        answer=answer,
        confidence=_confidence(result),
        course_ids=_course_ids(answer, result),
        status=_STATUS_BY_OUTCOME.get(result.outcome, "incomplete"),
        sources=_sources(result),
        outcome=result.outcome,
    )


def _answer_text(result: LoopResult) -> str:
    """The student-facing prose for every outcome the loop can reach."""
    if result.outcome == "answered" and result.answer is not None:
        return result.answer.text
    if result.outcome == "declined":
        # The model's own words for why it is out of scope.
        return result.reason or "That is outside what I can help with."
    if result.outcome == "proposed" and result.proposal is not None:
        name = course_display_name(result.proposal.target) or result.proposal.target
        return (
            f"I've prepared a request to {result.proposal.action} {name}. Nothing has been "
            "changed yet -- it needs your confirmation before anything happens."
        )
    # refused / stalled / exhausted -- the reason is diagnostic, not for a student.
    return _COULD_NOT_ANSWER


def _confidence(result: LoopResult) -> str:
    """low / medium / high, banded by the answer's weakest grounded basis.

    A non-answer is always low. An answer is only as strong as the weakest thing
    it stands on, so an interpreted or predicted fact honestly pulls the band
    down from a pure official record -- the same principle the basis ordering
    enforces everywhere else in the layer.
    """
    if result.outcome != "answered" or result.answer is None:
        return "low"
    strength = result.answer.basis.strength
    if strength >= Basis.OFFICIAL_RECORD.strength:
        return "high"
    if strength >= Basis.LLM_INTERPRETATION.strength:
        return "medium"
    return "low"


def _course_ids(answer: str, result: LoopResult) -> list[str]:
    """Course codes the answer names that a grounded fact also carries.

    Not model-authored: facts are the loop's only channel for admitted data, so
    intersecting the answer's codes against the facts keeps a hallucinated
    8-digit number out even if it reached the prose. Mirrors the V2 route's
    `_mentioned_course_ids`, which this replaces.
    """
    if not answer:
        return []
    import json

    grounded = course_codes_in(
        json.dumps([held.value for held in result.facts.values()], default=str)
    )
    return sorted(course_codes_in(answer) & grounded)


def course_references(course_ids: list[str]) -> list[dict[str, str]]:
    """Each id with its display name, falling back to the bare id."""
    return [{"id": cid, "name": course_display_name(cid) or cid} for cid in course_ids]


def _sources(result: LoopResult) -> list[str]:
    """A provenance hint per corpus search the loop ran -- the query term, taken
    from the transcript, never the passage text."""
    sources: set[str] = set()
    for turn in result.transcript:
        if turn.action == "call" and turn.detail.startswith("search_corpus("):
            # The transcript records `search_corpus({"query": "..."}) -> ...`.
            start = turn.detail.find('"query": "')
            if start != -1:
                start += len('"query": "')
                end = turn.detail.find('"', start)
                if end != -1:
                    sources.add(f"search: {turn.detail[start:end]}")
    return sorted(sources)


__all__ = ["Advice", "course_references", "run_advice", "to_advice"]
