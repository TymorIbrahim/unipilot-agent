"""`plan_term` term names must be distinct, because the name IS the split key.

A placed course comes back tagged with the term name it was asked for, and that
tag is the only way the model separates one term from another afterwards. Ask
for the same name twice and two genuinely separate terms return
indistinguishable.

Live, on "how many semesters will it take me to graduate":

    plan_term(terms=["winter","spring","summer","winter","spring","summer"])
    ... compute: select term == "winter"

The planner had applied the 18-credit cap correctly to EACH term. The model then
merged both winters into one bucket and reported "Winter -- 23 credits". Nothing
in the planner was wrong; the labels destroyed the result after it.

Refused at dispatch rather than repaired, for the usual reason: renaming the
model's terms for it would silently change what it asked for.
"""

from __future__ import annotations

import pytest

from app.agent_core.facts.answer import HeldFact
from app.agent_core.facts.dispatch import DispatchContext, dispatch
from app.agent_core.facts.types import Basis, Scalar, ScalarKind


def _detail(result) -> str:
    return " ".join(str(getattr(d, "message", d)) for d in result.defects.values())


def _context() -> DispatchContext:
    return DispatchContext(
        facts={"me": HeldFact(
            value=Scalar(ScalarKind.IDENTIFIER, "6a578a2da43a2cfe1bcc791c"),
            basis=Basis.OFFICIAL_RECORD)}
    )


def _call(terms: list[str]) -> dict:
    return {"tool": "plan_term", "as": "plan",
            "args": {"terms": terms, "candidates": [{"courseNumber": "00940412"}]}}


class TestTheLiveCall:
    async def test_the_exact_terms_that_shipped_23_credits(self) -> None:
        result = await dispatch(
            _call(["winter", "spring", "summer", "winter", "spring", "summer"]), _context()
        )
        assert "repeats" in _detail(result), "duplicate term names were accepted"

    async def test_it_names_which_term_repeats(self) -> None:
        result = await dispatch(_call(["winter", "spring", "winter"]), _context())
        detail = _detail(result)
        assert "'winter'" in detail and "'spring'" not in detail

    async def test_it_offers_a_working_alternative(self) -> None:
        """A refusal the model cannot act on costs the same turn twice."""
        result = await dispatch(_call(["winter", "winter"]), _context())
        assert "2026-1" in _detail(result)

    async def test_nothing_is_planned(self) -> None:
        result = await dispatch(_call(["winter", "winter"]), _context())
        assert not result.facts

    async def test_it_is_judged_before_the_plan_service_is_needed(self) -> None:
        """Argument shape does not depend on configuration. Checking settings
        first reported "not configured" for a malformed call -- sending the
        model to fix the wrong thing, and making two tests here pass for that
        reason instead of the one they assert."""
        result = await dispatch(_call(["winter", "winter"]), _context())
        assert "not configured" not in _detail(result)


class TestValidCallsAreUntouched:
    @pytest.mark.parametrize(
        "terms",
        [["winter"], ["winter", "spring"], ["2026-1", "2026-2", "2026-3"],
         ["winter", "spring", "summer"]],
        ids=["one", "two", "year-coded", "three-distinct"],
    )
    async def test_distinct_terms_are_not_refused_here(self, terms: list[str]) -> None:
        result = await dispatch(_call(terms), _context())
        assert "repeats" not in _detail(result), f"{terms} was wrongly rejected"

    async def test_an_empty_terms_list_keeps_its_own_message(self) -> None:
        result = await dispatch(_call([]), _context())
        assert "non-empty" in _detail(result)


class TestTheCatalogSaysSo:
    def test_the_guidance_warns_about_repeated_names(self) -> None:
        """The refusal is the backstop; the catalog is what should prevent it."""
        from app.agent_core.facts.catalog import COMPOSITES

        spec = next(s for s in COMPOSITES if s.name == "plan_term")
        assert "DISTINCT" in spec.when
        assert "2026-1" in spec.when
