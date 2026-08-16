"""`{"fact": "name"}` in a scalar argument, when the name is not held.

This is the repair path for a whole family of tools -- `interpret`,
`extract_list`, `traverse`, `forecast` and `propose` all resolve a scalar
argument through `_literal` -- and it was broken by an undefined name:

    f"Available: {sorted(facts)}."          # `facts` does not exist here

So the one line that runs when the model gets an argument wrong raised
NameError, the catch-all in `dispatch` turned that into "'interpret' could not
run with those arguments", and the model had nothing to repair from. Measured on
the English-requirement question: runs that hit it took 10-11 steps against a
usual 4, and one gave up entirely rather than answering.

Every branch of `_literal` is covered here, because the bug lived in the branch
nothing exercised.
"""

from __future__ import annotations

import pytest

from app.agent_core.facts.answer import HeldFact
from app.agent_core.facts.dispatch import DispatchContext, _literal
from app.agent_core.facts.operators import ExpressionDefect
from app.agent_core.facts.types import Basis, Collection, Completeness, Scalar, ScalarKind


def _context() -> DispatchContext:
    return DispatchContext(
        facts={
            "regulations_slug": HeldFact(
                value=Scalar(ScalarKind.IDENTIFIER, "regulations-undergraduate"),
                basis=Basis.WIKI_DERIVED,
            ),
            "passages": HeldFact(
                value=Collection(records=(), completeness=Completeness(complete=True, total=0)),
                basis=Basis.WIKI_DERIVED,
            ),
        }
    )


class TestTheUnheldFactPath:
    def test_it_returns_a_defect_rather_than_raising(self) -> None:
        result = _literal({"slug": {"fact": "nope_not_here"}}, "slug", _context())
        assert isinstance(result, ExpressionDefect)

    def test_it_lists_what_is_actually_held(self) -> None:
        result = _literal({"slug": {"fact": "nope_not_here"}}, "slug", _context())
        assert "regulations_slug" in result.message

    def test_a_near_miss_is_named(self) -> None:
        """Same repair the answer boundary and `collection()` already give."""
        result = _literal({"slug": {"fact": "regulations_slugs"}}, "slug", _context())
        assert "Did you mean 'regulations_slug'?" in result.message

    def test_an_unrelated_name_gets_no_guess(self) -> None:
        result = _literal({"slug": {"fact": "totally_different"}}, "slug", _context())
        assert "Did you mean" not in result.message


class TestTheOtherBranches:
    def test_a_plain_value_passes_through(self) -> None:
        assert _literal({"slug": "regulations-undergraduate"}, "slug", _context()) == (
            "regulations-undergraduate"
        )

    def test_a_held_scalar_resolves(self) -> None:
        assert _literal({"slug": {"fact": "regulations_slug"}}, "slug", _context()) == (
            "regulations-undergraduate"
        )

    def test_a_collection_is_refused_with_a_reason(self) -> None:
        result = _literal({"slug": {"fact": "passages"}}, "slug", _context())
        assert isinstance(result, ExpressionDefect)
        assert "collection" in result.message

    def test_an_object_without_a_fact_key_is_refused(self) -> None:
        result = _literal({"slug": {"name": "x"}}, "slug", _context())
        assert isinstance(result, ExpressionDefect)
        assert "must be a value" in result.message

    def test_a_missing_argument_takes_the_default(self) -> None:
        assert _literal({}, "slug", _context(), default="fallback") == "fallback"
