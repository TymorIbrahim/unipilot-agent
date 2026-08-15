"""The turn loop -- phase 10 of docs/agent/tools_implementation_plan.md.

Everything below this is deterministic. This is the one place a model decides
anything, so it is also the only place the old failure modes can return, and the
governors here are aimed at those specifically:

**Wandering after the answer was found.** The old loop would reach a sufficient
fact set and then spend its remaining turns not knowing it was done. A turn that
produces no new fact and attempts no answer is not thinking -- it is stalling,
and `NO_PROGRESS_LIMIT` ends it rather than letting the wall clock do it.

**Wandering that LOOKS productive.** The guard above asks whether a fact
arrived, and a re-derivation always produces one, so renaming the output made a
lap invisible to it: ten measured runs showed the same `search_corpus` query
issued sixteen times and the same pipeline recomputed five times, every lap
resetting the no-progress counter. Progress is therefore counted against
`_call_signatures` -- what a call DERIVES, with names stripped -- so a lap round
the same derivation is what it is: no progress, reported back to the model.

**Rejection with no legal move.** The old answer boundary could reject every
formulation the model had, so it burned the budget rediscovering that. Rejections
are bounded here AND carry the reason back, so a retry differs from its
predecessor. If they run out, the loop stops and says why rather than shipping
something unverified.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.agent_core.facts.answer import Answer, HeldFact, Ungrounded, resolve_answer
from app.agent_core.facts.answer_verify import verify_answer
from app.agent_core.facts.catalog import render_catalog
from app.agent_core.facts.conversation import Exchange, render_history
from app.agent_core.facts.dispatch import DispatchContext, dispatch
from app.agent_core.facts.find import array_paths, declared_paths
from app.agent_core.facts.presentation import render_facts
from app.agent_core.facts.propose import Proposal

MAX_TURNS = 8
NO_PROGRESS_LIMIT = 5
"""Turns without a new fact before the loop gives up.

Raised from 2 after live runs: a lookup whose key must be computed first
legitimately spends a turn producing the key and another using it, and a model
that mis-shapes the first attempt needs one more. Two was tight enough to kill
runs that were converging; the guard is against WANDERING, not against thinking.

Raised again from 3 when `_call_signatures` landed. That change made a repeated
derivation count as no progress -- correctly -- but it also made this number
mean something much stricter than it used to: before it, almost nothing reset
the counter's opposite, so three was nearly unreachable. Holding it at three
turned a run that would have thrashed and eventually converged into one that
stalled and returned nothing, which is the worse outcome of the two. Measured on
the knowledge-base question, which searches several phrasings before it finds
the right page: it stalled twice while recording examples, and its successful
runs use four to six turns.
"""
REJECTION_LIMIT = 3

DEFECT_NOTE = "A step failed"
"""Prefix marking an observation as a real failure rather than a nudge.

The turn prompt's "something you attempted failed" warning keys on this. Sharing
one constant is what keeps the note and the warning from disagreeing about
whether anything actually broke.
"""


class Model(Protocol):
    async def respond(self, prompt: str) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class Turn:
    index: int
    action: str
    detail: str


@dataclass
class LoopResult:
    outcome: str
    answer: Answer | None = None
    proposal: Proposal | None = None
    reason: str | None = None
    turns: int = 0
    facts: dict[str, HeldFact] = field(default_factory=dict)
    transcript: list[Turn] = field(default_factory=list)


async def run_loop(
    question: str,
    model: Model,
    context: DispatchContext,
    max_turns: int = MAX_TURNS,
    on_progress: "Callable[[str], None] | None" = None,
    time_budget_s: float | None = None,
    history: "Sequence[Exchange]" = (),
) -> LoopResult:
    """Run until the question is answered, refused, or a budget is spent.

    `on_progress` receives one short, student-facing phrase per turn -- the
    streaming route forwards these so a long request is not silent. It is
    advisory: a caller that ignores it sees the same final answer. Nothing
    grounded flows through it, so it never carries a number or a course code.

    `history` is the PRIOR exchanges of a conversation, so a follow-up like
    "continue" resolves against what was already said. It is context, never
    fact: the model reads it to interpret the question but re-derives every value
    fresh, so a stale answer from a past turn cannot re-enter as a pseudo-fact.

    `time_budget_s` is a WALL-CLOCK bound on the whole request, checked between
    turns. It concludes the loop gracefully -- with the transcript it has, which
    an outer `wait_for` cancellation would throw away -- rather than capping the
    number of turns or model calls: a hard turn cap kills a run that was one
    step from done, where a time budget lets it think for as long as it is
    actually making progress inside the window.
    """
    result = LoopResult(outcome="exhausted", facts=context.facts)
    observations: list[str] = []
    idle_turns = 0
    rejections = 0
    seen_derivations: set[str] = set()
    started = time.monotonic()

    for turn in range(1, max_turns + 1):
        if time_budget_s is not None and time.monotonic() - started >= time_budget_s:
            result.outcome = "exhausted"
            result.reason = f"the {time_budget_s:.0f}s time budget was spent before an answer was reached"
            result.transcript.append(Turn(turn, "timeout", result.reason))
            return result

        result.turns = turn
        reply = await model.respond(_prompt(question, context, observations, history))
        if on_progress is not None:
            _report_progress(on_progress, reply)

        if "decline" in reply:
            # A decline is legitimate ONLY for a genuinely out-of-scope question
            # -- the weather -- where nothing was ever fetched. Once the model
            # has pulled the student's records, the question is in scope, and a
            # decline is the model giving up on a hard SYNTHESIS, not judging the
            # question unanswerable. A live planning run fetched 49 curriculum
            # courses, the transcript, the offerings, then declined at turn 2
            # rather than working the plan. So a post-fetch decline is refused
            # like a bad answer and sent back to keep working, not concluded.
            fetched = [
                name for name, held in context.facts.items()
                if name != "me" and not _is_empty(held.value)
            ]
            if not fetched:
                result.outcome = "declined"
                result.reason = str(reply["decline"])
                result.transcript.append(Turn(turn, "declined", result.reason))
                return result

            rejections += 1
            result.transcript.append(Turn(turn, "decline-refused", str(reply["decline"])))
            if rejections >= REJECTION_LIMIT:
                # It will not push through. Conclude honestly rather than spin --
                # the route ships a grounded partial / graceful message.
                result.outcome = "refused"
                result.reason = "declined a question it had already fetched records for"
                return result
            observations.append(
                "Do NOT decline: you have already fetched this student's records, so the question "
                "IS in scope. Take the next derivation step, or give an ANSWER stating what you "
                f"established from the facts you hold ({sorted(fetched)}) and what stayed open."
            )
            continue

        if "answer" in reply:
            verdict = resolve_answer(str(reply["answer"]), context.facts, question)
            if isinstance(verdict, Answer):
                # Grounding passed -- every number came from a fact. The verify
                # step is the SEPARATE question the invariant cannot answer:
                # are those numbers SANE, and does a min-grade plan hold its
                # floor when the courses are taken together? A violation is
                # handled exactly like a rejected answer -- bounded, with the
                # specific reason fed back -- because it is one: an answer the
                # facts do not actually support, caught a layer later.
                problems = verify_answer(verdict, context.facts, question)
                if not problems:
                    result.outcome = "answered"
                    result.answer = verdict
                    result.transcript.append(Turn(turn, "answer", verdict.text))
                    return result
                verdict = Ungrounded("; ".join(problem.message for problem in problems))

            rejections += 1
            result.transcript.append(Turn(turn, "rejected", verdict.reason))
            if rejections >= REJECTION_LIMIT:
                # Stop rather than keep asking. Repeated rejection means the
                # facts do not support any answer the model can phrase, and
                # another attempt just spends money to learn that again.
                result.outcome = "refused"
                result.reason = verdict.reason
                return result
            # The reason goes back, so the retry differs from its predecessor.
            observations.append(f"Your answer was refused: {verdict.reason}")
            continue

        calls: Sequence[Mapping[str, Any]] = reply.get("calls") or ()
        if not calls:
            idle_turns += 1
            result.transcript.append(Turn(turn, "idle", "no calls, no answer"))
            if idle_turns >= NO_PROGRESS_LIMIT:
                result.outcome = "stalled"
                result.reason = "the loop stopped making progress: no tool calls and no answer attempt"
                return result
            observations.append(
                "That turn did nothing. Either call a tool or give the answer with the facts you hold."
            )
            continue

        gained = 0
        for call in calls:
            signatures = _call_signatures(call)
            # A call whose every derivation was already made this run cannot
            # teach the loop anything: nothing writes, so re-running it returns
            # what it returned before. It is still DISPATCHED -- the fact is
            # refreshed and the model may legitimately want it under a new name
            # -- but it does not count as progress, and the model is told.
            repeated = bool(signatures) and all(s in seen_derivations for s in signatures)
            seen_derivations.update(signatures)

            outcome = await dispatch(call, context)
            context.facts.update(outcome.facts)
            # Only facts with CONTENT count. Fetching an empty collection over
            # and over is the clearest possible case of busy-but-not-progressing,
            # and counting it as progress kept a live run alive for six turns
            # while it learned nothing.
            if not repeated:
                gained += sum(1 for held in outcome.facts.values() if not _is_empty(held.value))
            elif outcome.facts:
                observations.append(
                    f"You already ran {call.get('tool')}({_brief(call.get('args'), 90)}) this run and "
                    f"it returned {', '.join(f'{n}={_describe(h.value)}' for n, h in outcome.facts.items())}. "
                    "Repeating it cannot change the result. Use what you hold: take the next "
                    "derivation step, or ANSWER with the facts you have."
                )
            if outcome.proposal is not None:
                # A proposal is TERMINAL: an action request's correct outcome is
                # a change described for a person to approve, and once described
                # the agent's job is done. Making the model narrate it in a
                # second turn cost two live runs -- one re-proposed eight times
                # chasing a success signal `propose` never returns, another
                # proposed cleanly then had its narration refused for slotting an
                # ObjectId. The proposal's own `action`/`target` are already
                # readable, so the loop concludes from them directly, with no
                # second turn to get wrong.
                result.proposal = outcome.proposal
                result.outcome = "proposed"
                result.transcript.append(Turn(turn, "proposed", f"{outcome.proposal.action} {outcome.proposal.target}"))
                return result
            for name, defect in outcome.defects.items():
                observations.append(f"{DEFECT_NOTE} -- '{name}': {defect.message}")
            # The transcript records WHY, not just how many. A first live run
            # showed six failed calls as "1 defect(s)" each, which said nothing
            # about what went wrong -- and a transcript that cannot explain a
            # failure makes the next run a guess.
            # Say what each fact CONTAINS, not just that one arrived. A `find`
            # that matched nothing and a `find` that matched fifty are both
            # "1 fact(s)", and the difference is the whole story.
            produced = ", ".join(
                f"{name}={_describe(held.value)}" for name, held in outcome.facts.items()
            )
            # The ARGS matter as much as the result. A `find` returning zero rows
            # is entirely explained by what it asked for, and a transcript
            # without that leaves the next run guessing.
            summary = f"{call.get('tool')}({_brief(call.get('args'))}) -> {produced or '0 facts'}"
            if outcome.defects:
                reasons = "; ".join(f"{n}: {d.message}" for n, d in outcome.defects.items())
                summary = f"{summary}, {len(outcome.defects)} defect(s) -- {reasons}"
            result.transcript.append(Turn(turn, "call", summary))

        # A turn that fetched nothing new has not moved, whatever it attempted.
        idle_turns = 0 if gained else idle_turns + 1
        if idle_turns >= NO_PROGRESS_LIMIT:
            result.outcome = "stalled"
            result.reason = "the loop stopped making progress: repeated calls produced no new facts"
            return result

    result.reason = f"the turn budget of {max_turns} was spent without an answer"
    return result


_PROGRESS_BY_TOOL = {
    "find": "Looking up your records…",
    "search_corpus": "Searching the policies…",
    "interpret": "Reading the relevant policy…",
    "compute": "Working through the details…",
    "traverse": "Tracing the prerequisite chain…",
    "forecast": "Checking the offering history…",
    "optimize": "Putting a plan together…",
    "propose": "Preparing that for your approval…",
}


def _report_progress(on_progress: "Callable[[str], None]", reply: Mapping[str, Any]) -> None:
    """One reassuring phrase for the turn, from the FIRST tool it reached for.

    Deliberately generic -- it says what KIND of work is happening, never a
    result, so no grounded value can leak through the advisory channel. A reply
    that answers or declines needs no phrase; the answer itself is next.
    """
    calls = reply.get("calls") or ()
    tool = calls[0].get("tool") if calls else None
    phrase = _PROGRESS_BY_TOOL.get(tool) if tool else None
    if phrase is not None:
        on_progress(phrase)


def _is_empty(value: Any) -> bool:
    records = getattr(value, "records", None)
    return records is not None and len(records) == 0


def _call_signatures(call: Mapping[str, Any]) -> tuple[str, ...]:
    """How a call DERIVES its result, with every name stripped out.

    This is what `gained` is counted against, and the names have to go: the
    wandering this guards against re-derives a value it already holds under a
    fresh name, and a signature that included the name would call every lap new.
    Measured over ten live runs, the waste was almost entirely literal repeats --
    the same `search_corpus` query issued six times, the same `completed_numbers`
    pipeline recomputed five times, `compute`+`interpret` on the same slug four
    times before the run gave up.

    `compute` is decomposed per PIPELINE rather than signed whole, because the
    repeat hides at that granularity: a turn re-deriving `completed_numbers`
    alongside one genuinely new pipeline is a different call each time but the
    same derivation, and signing the call as a unit would miss it.
    """
    tool = str(call.get("tool"))
    args = call.get("args") if isinstance(call.get("args"), Mapping) else {}

    if tool == "compute":
        pipelines = args.get("pipelines")
        if isinstance(pipelines, Sequence) and not isinstance(pipelines, (str, bytes)):
            return tuple(
                "compute:" + json.dumps(
                    {k: v for k, v in pipeline.items() if k != "name"},
                    sort_keys=True, default=str,
                )
                for pipeline in pipelines
                if isinstance(pipeline, Mapping)
            )

    # `as` is the caller's name for the result, not part of the derivation.
    return (f"{tool}:" + json.dumps(
        {k: v for k, v in args.items() if k != "as"}, sort_keys=True, default=str
    ),)


def _brief(args: Any, limit: int = 180) -> str:
    rendered = json.dumps(args, default=str, ensure_ascii=False) if args else "{}"
    return rendered if len(rendered) <= limit else rendered[: limit - 1] + "\u2026"


def _render_sources(context: DispatchContext) -> str:
    """What `find` can read, and the fields each source has.

    Measured on the first live run: the model spent three turns guessing source
    names (`students`, `degree_requirements`, `profile`) and field names
    (`student_id`, `id`). Every guess came back naming the real alternatives and
    it did converge -- but discovering the schema by rejection costs a turn per
    wrong guess, and the schema is knowable up front.
    """
    if not context.schemas:
        return "## data sources\n  (none configured)"

    lines = []
    nested = False
    for name, schema in sorted(context.schemas.items()):
        arrays = array_paths(schema)
        nested = nested or bool(arrays)
        fields = ", ".join(f"{p}[]" if p in arrays else p for p in declared_paths(schema))
        lines.append(f"  {name}\n     key: {schema.key}\n     fields: {fields}")
        for local, foreign in getattr(schema, "joins", ()):
            lines.append(f"     joins: {local} -> {foreign}")
        # Rendered HERE, next to the field it qualifies, because this is where
        # the column gets chosen. The same warning in a prompt paragraph did not
        # stop a live run summing `creditsEarned` over failed courses.
        for field, note in sorted(getattr(schema, "field_notes", {}).items()):
            lines.append(f"     ! {field}: {note}")

    # Nested paths are listed in full for the same reason the source names are:
    # a field the model cannot see is a field it will guess at. Listing only
    # top-level names would show `semesters` beside the scalars and hide both
    # that it is an array and everything inside it.
    note = (
        "\n  A field marked [] holds an array. Filter it by its inner path "
        "(`semesters.order`), and use compute's `unnest` to get one record per element."
        if nested
        else ""
    )
    return "## data sources for `find`\n" + "\n".join(lines) + note


def _describe(value: Any) -> str:
    records = getattr(value, "records", None)
    if records is None:
        return f"{getattr(value, 'value', value)}"
    complete = getattr(value, "completeness", None)
    suffix = "" if complete is None or complete.complete else " TRUNCATED"
    return f"[{len(records)} records{suffix}]"


def _prompt(
    question: str,
    context: DispatchContext,
    observations: Sequence[str],
    history: "Sequence[Exchange]" = (),
) -> str:
    """The turn prompt.

    Facts are rendered as shapes rather than payloads, so the prompt grows with
    the NUMBER of facts rather than their size -- otherwise one large fetch
    crowds out everything the model needs to reason with.
    """
    values = {name: held.value for name, held in context.facts.items()}
    conversation = render_history(list(history))
    conversation_block = f"{conversation}\n\n" if conversation else ""
    recent = observations[-4:]
    notes = "\n".join(f"  - {note}" for note in recent) or "  (none)"
    # A derivation that half-failed is the dangerous case: a live run had its
    # subtraction fail, answered from the un-subtracted operand, and the answer
    # passed every check while being wrong. The model has to be told that
    # answering over a broken step is a decision, not an oversight.
    #
    # Only when something ACTUALLY failed, though. This used to fire on any
    # note at all, so an idle turn or a refused answer -- neither of which is a
    # failed step -- told the model a step had failed and warned it about citing
    # a partial result that did not exist. An instruction that describes a state
    # the system is not in is worse than none: it is confidently wrong about the
    # one thing the model cannot check.
    warning = (
        "\n\nSOMETHING YOU ATTEMPTED FAILED LAST TURN (above). Either fix it, or -- if you "
        "answer anyway -- make sure the facts you cite are the ones you actually meant, not a "
        "partial result of the step that failed."
        if any(note.startswith(DEFECT_NOTE) for note in recent)
        else ""
    )
    # `me` arrives seeded by the caller and reads as an opaque id. Naming it
    # costs one line and saves the model inferring what to filter by.
    whose = (
        "  (`me` is the id of the student asking -- filter their records by it)\n"
        if "me" in context.facts
        else ""
    )
    return (
        f"{conversation_block}"
        f"QUESTION: {question}\n\n"
        f"{render_catalog(context)}\n\n"
        f"{_render_sources(context)}\n\n"
        f"FACTS YOU HOLD:\n{render_facts(values)}\n{whose}\n"
        f"NOTES FROM LAST TURN:\n{notes}{warning}\n\n"
        "Reply with either {\"calls\": [...]} or {\"answer\": \"...\"}. "
        "Every number in an answer must be a {fact_name} slot; typed digits are refused."
    )


__all__ = ["LoopResult", "MAX_TURNS", "Model", "Turn", "run_loop"]
