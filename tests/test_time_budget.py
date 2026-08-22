"""The budget must fit the ceiling the deployment actually has.

Established 2026-08-22 by asking the platform rather than inferring from
response timings:

    GET /api/health?sleep=100&kb=600  ->  100.5s, HTTP 200, 600KB delivered

A function that only sleeps runs 100 seconds and returns, so neither duration
nor response size is capped. Three earlier readings of this -- "60s execution
cap", "no cap, it is time-to-first-byte", "60s because the plan is Hobby" -- were
all inferred from how long requests happened to survive, and all were wrong.

What the budget still has to respect is the 300s ceiling, and the ~13s cold
start that belongs to the same window as the run: the loop's clock did not start
until after it, so a run that logged `elapsed=48.2s` had already spent more.
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
    """A budget above the ceiling cannot be honoured, so it must not be
    configurable by accident."""
    settings = Settings(time_budget_s=600.0)

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
