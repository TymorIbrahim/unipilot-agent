"""The budget must fit the ceiling the deployment actually has.

Established 2026-08-22 by asking the platform rather than inferring from
response timings:

    GET /v2/user               -> plan: hobby
    GET /v13/deployments/{id}  -> functions: {"api/index.py": {"maxDuration": 300}}
                                  lambdas[0].maxDuration: None

`vercel.json` asks for 300; the deployed function carries none. On Hobby a
classic serverless function is capped at 60s, and the request is killed there
mid-response:

    responseStatusCode: 0 ... agent_run outcome=answered turns=5 elapsed=48.2s

The function SUCCEEDED and the caller received nothing, because the ~13s cold
start belongs to the same 60s window and the loop's clock did not start until
after it.
"""

from __future__ import annotations

from app.config import (
    DEFAULT_TIME_BUDGET_S,
    RESPONSE_RESERVE_S,
    VERCEL_HARD_LIMIT_S,
    Settings,
)


def test_the_default_budget_fits_inside_the_ceiling() -> None:
    assert DEFAULT_TIME_BUDGET_S + RESPONSE_RESERVE_S <= VERCEL_HARD_LIMIT_S


def test_an_oversized_configured_budget_is_clamped() -> None:
    """Not a formality: `TIME_BUDGET_S=240` was live in production against a
    ceiling of 60, and every question over ~45s returned an aborted connection
    rather than an answer. The clamp is what keeps a stale env var from meaning
    "no answer"."""
    settings = Settings(time_budget_s=240.0)

    assert settings.effective_time_budget_s() == VERCEL_HARD_LIMIT_S - RESPONSE_RESERVE_S
    assert settings.effective_time_budget_s() < VERCEL_HARD_LIMIT_S


def test_a_budget_under_the_ceiling_is_left_alone() -> None:
    settings = Settings(time_budget_s=30.0)

    assert settings.effective_time_budget_s() == 30.0


def test_the_reserve_leaves_room_for_the_response() -> None:
    """A seven-step reply is ~390KB -- every step carries the 51KB system
    prompt -- and the budget governs the loop, not the write that follows it."""
    assert RESPONSE_RESERVE_S > 0
    assert VERCEL_HARD_LIMIT_S - RESPONSE_RESERVE_S > 0


def test_raising_the_ceiling_is_one_edit() -> None:
    """On a paid plan the ceiling becomes 300 and everything follows from it,
    so nothing may hardcode 60 alongside this constant."""
    settings = Settings(time_budget_s=DEFAULT_TIME_BUDGET_S)

    assert settings.effective_time_budget_s() == min(
        DEFAULT_TIME_BUDGET_S, VERCEL_HARD_LIMIT_S - RESPONSE_RESERVE_S
    )
