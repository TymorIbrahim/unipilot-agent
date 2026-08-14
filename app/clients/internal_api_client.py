"""The term planner, called in process instead of over HTTP.

In UniPilot this module was an HTTP client: the agent service posted to the API
service's `/internal/term-plan` endpoint, because the planning engine lived in a
different container. Here there is no other container -- the engine was ported
alongside the agent and sits one import away.

So the module keeps its NAME and its FUNCTION SIGNATURES, and changes only what
happens inside them. `facts/dispatch.py` imports `fetch_term_plan` and
`InternalApiClientError` exactly as before and needs no edit, while the network
hop, the service token, and a whole class of "the plan service is unreachable"
failures simply stop existing.

`InternalApiClientError` is kept rather than dropped for the same reason: dispatch
catches it to turn a planner failure into a defect the model can read and react
to. Removing it would let a planning error escape as an unhandled exception and
abort the whole request, which is precisely the outcome the catch exists to
prevent.
"""

from __future__ import annotations

from typing import Any


class InternalApiClientError(RuntimeError):
    """A planner call that could not produce a plan.

    `detail` is what dispatch renders into the defect the model sees, so it is
    written to be actionable by a reader who cannot see this stack.
    """

    def __init__(self, detail: str, status_code: int | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


async def fetch_term_plan(
    *,
    user_id: str,
    semester_codes: list[str],
    candidates: list[dict[str, Any]],
    max_credits: float | None = None,
    settings: Any | None = None,
) -> dict[str, Any]:
    """Build a conflict-free term plan from agent-supplied candidates.

    Returns the same payload shape the HTTP endpoint returned -- `terms` (each
    with placedCourses, credits, weeklySchedule, examSummary), `unscheduled`, and
    `maxCredits` -- because `dispatch._placed_collection` and `_plan_summary`
    already read that shape and are not worth rewriting for a transport change.
    """
    from app.services.term_plan_service import build_term_plan

    try:
        return await build_term_plan(
            user_id=user_id,
            semester_codes=semester_codes,
            candidates=candidates,
            max_credits=max_credits,
        )
    except InternalApiClientError:
        raise
    except Exception as error:  # noqa: BLE001
        # Fail as a readable defect, never as a crash: an unhandled exception
        # here would abort a request that could still answer without a plan.
        raise InternalApiClientError(f"{type(error).__name__}: {error}") from error


async def fetch_student_user_context(*, user_id: str, settings: Any | None = None) -> dict[str, Any]:
    """Kept for signature parity; the ported agent reads student context from
    its own Supabase sources rather than through this call."""
    raise InternalApiClientError("student context is read directly from the database")


__all__ = ["InternalApiClientError", "fetch_student_user_context", "fetch_term_plan"]
