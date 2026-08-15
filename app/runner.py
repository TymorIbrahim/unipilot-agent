"""The seam between the HTTP layer and the reasoning core.

`main.py` knows only this module's `run_agent`, and this module knows only the
core's entry point. That keeps the spec-mandated response shape from leaking
into the reasoning code, and keeps the reasoning code's own vocabulary
(outcomes, facts, working sets) from leaking into the HTTP contract.

Two things happen here that happen nowhere else:

**Tracing is installed.** The chat client is wrapped in `TracedChat` before it
reaches the core, and the wrapped client is what the core builds both its
reasoning adapter AND its prose extractor from. That is what makes `steps`
complete: recording at the client rather than at the call sites means a model
call cannot be made without being recorded, and a call site added later is
traced the day it is written.

**The clock is enforced.** Vercel kills the invocation at 300s with no output at
all. The loop is given a budget below that and concludes GRACEFULLY when it
expires -- returning the answer it has grounded so far, with the trace that
produced it. A partial honest answer scores; a killed request does not.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from app.tracing.modules import REASONING_LOOP
from app.tracing.recorder import StepRecorder, TracedChat

logger = logging.getLogger(__name__)

# The primary demo student: ISE track, 44 completed courses, and the record the
# term planner was already validated against end to end. A bare `{"prompt": ...}`
# with no student named resolves here.
DEFAULT_STUDENT_ID = "6a578a2da43a2cfe1bcc791c"


@dataclass(frozen=True)
class AgentResult:
    """What the HTTP layer needs, and nothing about how it was produced."""

    ok: bool
    answer: str | None
    error: str | None
    steps: list[dict[str, Any]] = field(default_factory=list)


async def run_agent(
    prompt: str, *, student_id: str | None = None, conversation_id: str | None = None
) -> AgentResult:
    """Run one request end to end and return its answer plus its full trace."""
    from app.agent_core.facts.service import run_advice, to_advice
    from app.agent_core.reasoning.llm_client import build_chat_llm
    from app.config import get_settings

    recorder = StepRecorder()
    settings = get_settings()
    student = student_id or DEFAULT_STUDENT_ID

    question = (prompt or "").strip()
    if not question:
        # Caught here rather than in the loop: an empty prompt is a client
        # mistake with a precise cause, and the loop would spend model calls
        # discovering that it has nothing to answer.
        return AgentResult(
            ok=False, answer=None, error="prompt is empty", steps=recorder.as_list()
        )

    chat = build_chat_llm(settings=settings)
    if chat is None:
        return AgentResult(
            ok=False,
            answer=None,
            error="the agent is not configured: no LLM API key is set",
            steps=recorder.as_list(),
        )

    # Every model call the run makes now flows through this one object. The
    # reasoning turns record as ReasoningLoop; the extractor re-stamps the same
    # client as Interpreter / ListInterpreter at its own call sites.
    traced = TracedChat(chat, recorder, REASONING_LOOP)

    started = time.monotonic()
    try:
        result = await run_advice(
            question,
            student,
            settings=settings,
            time_budget_s=settings.effective_time_budget_s(),
            chat=traced,
            conversation_id=conversation_id,
        )
    except Exception as error:  # noqa: BLE001 -- reported, never raised past here
        # The steps recorded BEFORE the failure are kept. A trace that stops
        # where the run broke is the most useful artifact a failed request can
        # leave, and the spec requires `steps` on the failure path too.
        logger.exception("agent_run_failed")
        return AgentResult(
            ok=False,
            answer=None,
            error=f"{type(error).__name__}: {error}",
            steps=recorder.as_list(),
        )

    advice = to_advice(result)
    elapsed = time.monotonic() - started
    logger.info(
        "agent_run outcome=%s turns=%d steps=%d elapsed=%.1fs confidence=%s",
        result.outcome,
        result.turns,
        len(recorder),
        elapsed,
        advice.confidence,
    )

    # `to_advice` already maps every outcome to student-facing prose, including
    # the ones that did not answer -- so `response` is never null on a run that
    # completed, and `error` names the outcome only when it was not an answer.
    return AgentResult(
        ok=result.outcome in _ANSWERED_OUTCOMES,
        answer=advice.answer,
        error=None if result.outcome in _ANSWERED_OUTCOMES else _error_for(result),
        steps=recorder.as_list(),
    )


# `declined` and `proposed` ARE successful responses: the agent answered by
# declining an out-of-scope question, or by preparing a change for approval.
# Only a run that failed to produce a grounded answer reports an error.
_ANSWERED_OUTCOMES = frozenset({"answered", "declined", "proposed"})


def _error_for(result: Any) -> str:
    """Why a run did not answer, in one line, without leaking diagnostics.

    `result.reason` is written for a developer reading a transcript, so it is
    summarised rather than forwarded: the student-facing prose is already in
    `response`, and `error` exists to tell whoever is reading the trace which
    budget or guard stopped the run.
    """
    from app.agent_core.facts.service import UNKNOWN_STUDENT

    reason = str(getattr(result, "reason", "") or "")
    if reason.startswith(UNKNOWN_STUDENT):
        # A caller naming a student who does not exist is a client mistake with
        # a precise cause, not a diagnostic to be summarised away. Forwarded
        # verbatim for the same reason "prompt is empty" is.
        return reason

    reasons = {
        "refused": "the answer could not be grounded in the student's records",
        "stalled": "the agent stopped making progress before reaching an answer",
        "exhausted": "the time or turn budget was spent before an answer was reached",
    }
    return reasons.get(result.outcome, f"the run ended as '{result.outcome}'")


__all__ = ["DEFAULT_STUDENT_ID", "AgentResult", "run_agent"]
