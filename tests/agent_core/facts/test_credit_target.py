"""Bounding the candidate set was a turn, and the turn most costly to get wrong.

The recipe's rule -- every mandatory course, then electives only until the
credits still needed are covered -- is pure arithmetic over facts already held.
There is one right answer, and the model was spending ~15s of a 45s budget
computing it.

Getting it wrong is worse than paying for it. Handing the planner the WHOLE
unfinished track produced "4 semesters" where the truth is 2: 50.0 credits of
courses scheduled against a 25.5-credit requirement, with every individual term
legal, so no per-term check could see it.
"""

from __future__ import annotations

import pytest

from app.agent_core.facts.dispatch import (
    DispatchContext,
    _bounded_by_credits,
    _resolve_number,
    _strip_internal,
)
from app.agent_core.facts.answer import HeldFact
from app.agent_core.facts.types import Basis, Scalar, ScalarKind

Q = ScalarKind.QUANTITY

# The demo student: 6 mandatory (17.5cr) and 15 electives (32.5cr), needing 25.5.
MANDATORY = [{"courseNumber": f"m{i}", "category": "mandatory", "_credits": c}
             for i, c in enumerate([1.5, 2.5, 3.0, 3.5, 3.5, 3.5])]
ELECTIVES = [{"courseNumber": f"e{i}", "category": "elective", "_credits": 2.5}
             for i in range(13)]
ALL = MANDATORY + ELECTIVES


class TestWhatItKeeps:
    def test_every_mandatory_course_survives(self) -> None:
        kept = _bounded_by_credits(ALL, 25.5)
        assert [c["courseNumber"] for c in kept if c["category"] == "mandatory"] == \
               [c["courseNumber"] for c in MANDATORY]

    def test_it_stops_once_the_target_is_covered(self) -> None:
        kept = _bounded_by_credits(ALL, 25.5)
        total = sum(c["_credits"] for c in kept)
        assert total >= 25.5, "it must cover what the degree needs"
        assert total < 25.5 + 2.5 + 0.001, "and not run far past it"

    def test_the_live_over_planning_is_prevented(self) -> None:
        """50.0 credits against a 25.5 requirement is what answered 4 semesters."""
        assert sum(c["_credits"] for c in ALL) == 50.0
        assert sum(c["_credits"] for c in _bounded_by_credits(ALL, 25.5)) <= 28.0

    def test_mandatory_alone_can_exceed_the_target(self) -> None:
        """Mandatory courses are never dropped -- they must be taken whatever
        the total comes to, so the target bounds the electives only."""
        kept = _bounded_by_credits(MANDATORY + ELECTIVES, 5.0)
        assert len([c for c in kept if c["category"] == "mandatory"]) == len(MANDATORY)
        assert [c for c in kept if c["category"] == "elective"] == []

    def test_a_target_of_zero_changes_nothing(self) -> None:
        assert _bounded_by_credits(ALL, 0.0) == ALL

    def test_a_course_with_no_credits_does_not_stall_the_count(self) -> None:
        candidates = [{"courseNumber": "x", "category": "elective"}] * 3
        assert len(_bounded_by_credits(candidates, 10.0)) == 3


class TestTheInternalFieldNeverLeaves:
    def test_credits_are_stripped_before_the_plan_service(self) -> None:
        stripped = _strip_internal(_bounded_by_credits(ALL, 25.5))
        assert all(set(c) == {"courseNumber", "category"} for c in stripped)


class TestTheTargetMayBeAFact:
    """The gap IS a held fact, seeded at run start. Requiring a typed digit
    would ask the model to launder a fact into a literal, which is the one thing
    the grounding invariant forbids."""

    def _context(self) -> DispatchContext:
        return DispatchContext(facts={
            "credits_needed": HeldFact(value=Scalar(Q, 25.5), basis=Basis.OFFICIAL_RECORD),
            "program_slug": HeldFact(value=Scalar(ScalarKind.IDENTIFIER, "track-ise"),
                                     basis=Basis.OFFICIAL_RECORD),
        })

    def test_a_fact_reference_resolves(self) -> None:
        assert _resolve_number({"fact": "credits_needed"}, self._context()) == 25.5

    def test_a_bare_number_still_works(self) -> None:
        assert _resolve_number(25.5, self._context()) == 25.5

    def test_absent_means_no_bound(self) -> None:
        assert _resolve_number(None, self._context()) is None

    def test_an_unknown_fact_is_a_defect_naming_what_is_held(self) -> None:
        from app.agent_core.facts.operators import ExpressionDefect

        result = _resolve_number({"fact": "nope"}, self._context())
        assert isinstance(result, ExpressionDefect)
        assert "credits_needed" in result.message

    def test_a_non_numeric_fact_is_a_defect(self) -> None:
        from app.agent_core.facts.operators import ExpressionDefect

        assert isinstance(_resolve_number({"fact": "program_slug"}, self._context()),
                          ExpressionDefect)


class TestItIsAdvertised:
    def test_the_catalog_example_uses_it(self) -> None:
        from app.agent_core.facts.catalog import render_catalog

        assert "credit_target" in render_catalog()

    def test_the_recipe_tells_the_model_to_pass_the_fact(self) -> None:
        from app.agent_core.facts.adapter import SYSTEM_PROMPT

        assert 'credit_target = {"fact": "credits_needed"}' in SYSTEM_PROMPT
