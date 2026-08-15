"""A pipeline must be able to FILTER BY a sibling, not just read from one.

The catalog tells the model to prefer one `compute` carrying several pipelines
"because pipelines may reference each other's results by name". Half of that was
true: a sibling worked as a `source` and failed as a filter VALUE, because
`{"fact": ...}` references were resolved once for the whole call, up front,
against the working set alone.

So the natural formulation of the commonest advising question --

    my_numbers : the course numbers I have passed
    met_edges  : the prerequisite edges whose `requires` is IN my_numbers

came back "the filter refers to fact 'my_numbers', which is not held" with
`my_numbers` defined one line above. Measured: an eligibility run spent eight
consecutive turns rewriting that shape under eight different names before the
loop gave up on it.
"""

from __future__ import annotations

from app.agent_core.facts.answer import HeldFact
from app.agent_core.facts.dispatch import DispatchContext, dispatch
from app.agent_core.facts.types import (
    Basis,
    Collection,
    Completeness,
    Record,
    Scalar,
    ScalarKind,
)

I = ScalarKind.IDENTIFIER


def _collection(field: str, *values: str) -> Collection:
    return Collection(
        records=tuple(
            Record(fields={field: Scalar(I, value)}, basis=Basis.OFFICIAL_RECORD)
            for value in values
        ),
        completeness=Completeness(complete=True, total=len(values)),
    )


def _context() -> DispatchContext:
    return DispatchContext(
        facts={
            "mine": HeldFact(
                value=_collection("courseNumber", "00940224", "00940345"),
                basis=Basis.OFFICIAL_RECORD,
            ),
            "edges": HeldFact(
                value=_collection("requires", "00940224", "00940226", "00999999"),
                basis=Basis.OFFICIAL_RECORD,
            ),
        }
    )


def _select_by_sibling(name: str, sibling: str) -> dict:
    return {
        "name": name,
        "source": "edges",
        "stages": [{
            "op": "select",
            "predicate": {"path": "requires", "op": "in",
                          "value": {"fact": sibling, "field": "courseNumber"}},
        }],
    }


class TestFilteringByASibling:
    async def test_a_sibling_can_be_used_as_a_filter_value(self) -> None:
        context = _context()
        result = await dispatch(
            {"tool": "compute", "args": {"pipelines": [
                {"name": "copy", "source": "mine",
                 "stages": [{"op": "project", "fields": {"courseNumber": "courseNumber"}}]},
                _select_by_sibling("met", "copy"),
            ]}},
            context,
        )
        assert not result.defects, {n: d.message for n, d in result.defects.items()}
        kept = [r.fields["requires"].value for r in result.facts["met"].value.records]
        assert kept == ["00940224"], "only the edge the student has passed survives"

    async def test_declaration_order_does_not_matter(self) -> None:
        """Ordering comes from the references, and a `{"fact": ...}` inside a
        filter is a reference like any other -- otherwise the sibling may not
        have run yet and the filter fails for what is really an ordering bug."""
        context = _context()
        result = await dispatch(
            {"tool": "compute", "args": {"pipelines": [
                _select_by_sibling("met", "copy"),  # declared BEFORE what it needs
                {"name": "copy", "source": "mine",
                 "stages": [{"op": "project", "fields": {"courseNumber": "courseNumber"}}]},
            ]}},
            context,
        )
        assert not result.defects, {n: d.message for n, d in result.defects.items()}
        assert len(result.facts["met"].value.records) == 1


class TestTheGuardsStillHold:
    async def test_an_unknown_fact_is_still_refused_loudly(self) -> None:
        """Filtering by a fact that does not exist must not quietly match
        nothing -- an empty result reads as a real answer."""
        result = await dispatch(
            {"tool": "compute", "args": {"pipelines": [_select_by_sibling("met", "imaginary")]}},
            _context(),
        )
        assert "imaginary" in result.defects["met"].message
        assert "not held" in result.defects["met"].message

    async def test_a_working_set_fact_still_resolves(self) -> None:
        """The pre-existing route -- filtering by something fetched in an earlier
        call -- must keep working."""
        result = await dispatch(
            {"tool": "compute", "args": {"pipelines": [_select_by_sibling("met", "mine")]}},
            _context(),
        )
        assert not result.defects
        assert len(result.facts["met"].value.records) == 1

    async def test_one_failed_pipeline_does_not_discard_the_others(self) -> None:
        context = _context()
        result = await dispatch(
            {"tool": "compute", "args": {"pipelines": [
                {"name": "good", "source": "mine", "stages": [{"op": "aggregate", "agg": "count"}]},
                _select_by_sibling("bad", "imaginary"),
            ]}},
            context,
        )
        assert "bad" in result.defects
        assert result.facts["good"].value.value == 2, "the good pipeline still lands"
