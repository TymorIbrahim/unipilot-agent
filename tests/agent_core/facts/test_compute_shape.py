"""A `compute` with no pipelines returned nothing, and said nothing about it.

Live, twice in the same run, the model wrote the expression at the top level:

    compute({"ceil_div": [{"fact": "credits_needed"},
                          {"fact": "max_credits_per_semester"}]})  -> 0 facts

The expression is correct. It is simply in the wrong place -- `compute` takes a
list of NAMED pipelines. What came back was no facts and no defect, which is the
worst possible response: nothing distinguishes it from a computation that
legitimately produced no rows, so the model repeated it verbatim and the run
lost two of the three turns its budget allowed.

The third silent-or-unfollowable failure found this week, after the predicate
literal that demanded the `kind` it rejected and the refusal whose advice named
facts the model did not hold.
"""

from __future__ import annotations

import asyncio

from app.agent_core.facts.answer import HeldFact
from app.agent_core.facts.dispatch import DispatchContext, dispatch
from app.agent_core.facts.types import Basis, Scalar, ScalarKind

Q = ScalarKind.QUANTITY


def _context() -> DispatchContext:
    return DispatchContext(facts={
        "credits_needed": HeldFact(value=Scalar(Q, 25.5), basis=Basis.OFFICIAL_RECORD),
        "max_credits_per_semester": HeldFact(value=Scalar(Q, 18.0), basis=Basis.OFFICIAL_RECORD),
    })


def _run(args: dict):
    return asyncio.run(dispatch({"tool": "compute", "as": "n", "args": args}, _context()))


class TestTheMalformedCallExplainsItself:
    ARGS = {"ceil_div": [{"fact": "credits_needed"}, {"fact": "max_credits_per_semester"}]}

    def test_it_is_a_defect_rather_than_silence(self) -> None:
        outcome = _run(self.ARGS)
        assert outcome.defects, "a silent no-op is indistinguishable from an empty result"
        assert not outcome.facts

    def test_the_message_shows_the_shape_to_write(self) -> None:
        message = list(_run(self.ARGS).defects.values())[0].message
        assert '"pipelines"' in message
        assert '"name"' in message and '"value"' in message

    def test_an_empty_pipeline_list_is_caught_too(self) -> None:
        assert _run({"pipelines": []}).defects

    def test_a_missing_args_object_is_caught(self) -> None:
        assert _run({}).defects


class TestTheCorrectCallIsUntouched:
    def test_a_scalar_pipeline_still_computes(self) -> None:
        outcome = _run({"pipelines": [{"name": "n", "value": {
            "ceil_div": [{"fact": "credits_needed"}, {"fact": "max_credits_per_semester"}]}}]})
        assert not outcome.defects
        assert outcome.facts["n"].value.value == 2

    def test_a_pipeline_that_legitimately_yields_nothing_is_not_a_defect(self) -> None:
        """The case the silence was being confused with: a real pipeline whose
        filter matches no rows is a fact about the data, not a mistake."""
        from app.agent_core.facts.types import Collection, Completeness

        context = _context()
        context.facts["rows"] = HeldFact(
            value=Collection(records=(), completeness=Completeness(complete=True, total=0)),
            basis=Basis.OFFICIAL_RECORD)
        outcome = asyncio.run(dispatch({"tool": "compute", "as": "n", "args": {"pipelines": [
            {"name": "n", "source": "rows", "stages": [{"op": "distinct", "on": "x"}]}]}}, context))
        assert not outcome.defects
