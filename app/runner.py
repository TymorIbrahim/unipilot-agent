"""The seam between the HTTP layer and the reasoning core.

`main.py` knows only this module's `run_agent`, and this module knows only the
core's entry point. That keeps the spec-mandated response shape from leaking
into the reasoning code, and keeps the reasoning code's own vocabulary
(outcomes, facts, working sets) from leaking into the HTTP contract.

The core is being ported in from UniPilot; until it lands, this returns a
truthful "not wired yet" error in the correct shape rather than a fake answer.
An endpoint that answers convincingly before its brain is connected is the one
failure mode that would waste real debugging time later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.tracing.recorder import StepRecorder

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


async def run_agent(prompt: str, *, student_id: str | None = None) -> AgentResult:
    """Run one request end to end and return its answer plus its full trace."""
    recorder = StepRecorder()
    _student = student_id or DEFAULT_STUDENT_ID

    # TODO(port): build the dispatch context over Supabase, wrap the chat client
    # in TracedChat(recorder), and run the reasoning loop under the time budget.
    return AgentResult(
        ok=False,
        answer=None,
        error="the reasoning core has not been wired up yet",
        steps=recorder.as_list(),
    )


__all__ = ["DEFAULT_STUDENT_ID", "AgentResult", "run_agent"]
