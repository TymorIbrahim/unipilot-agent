"""Filtering on a field no record carries told a student they were ineligible.

Asked "Can I take 00960211 and 01040174 in the same semester?", the deployed
agent fetched the prerequisite edges correctly into `edges_00960211`, then ran
its group check over `course_00960211` -- the CATALOG row, which has no
`requires` and no `group`. Every comparison silently failed, the met-group count
came out 0, and the answer was:

    No -- you meet 0 prerequisite groups for 00960211 ...

The student meets 1 of 1. They are eligible. `eligibility_00960211` scores 3/3
in the eval; the same course, reached through a two-course question, came back
inverted.

A filter that cannot be APPLIED is not a filter that matched nothing, and the
difference is the whole answer. `distinct` on an absent field was already a
defect -- `select` was not, and `select` is the one that decides eligibility.
"""

from __future__ import annotations

import asyncio

import pytest

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
T = ScalarKind.TEXT


def _collection(*records: Record) -> Collection:
    return Collection(records=records,
                      completeness=Completeness(complete=True, total=len(records)))


def _context() -> DispatchContext:
    catalog = Record(fields={"courseNumber": Scalar(I, "00960211"), "title": Scalar(T, "x")},
                     basis=Basis.OFFICIAL_RECORD)
    edge = Record(fields={"course": Scalar(I, "00960211"), "group": Scalar(I, "g1"),
                          "requires": Scalar(I, "00940224")}, basis=Basis.OFFICIAL_RECORD)
    mixed = Record(fields={"course": Scalar(I, "00970135")}, basis=Basis.OFFICIAL_RECORD)
    return DispatchContext(facts={
        "course": HeldFact(value=_collection(catalog), basis=Basis.OFFICIAL_RECORD),
        "edges": HeldFact(value=_collection(edge), basis=Basis.OFFICIAL_RECORD),
        "some_have_it": HeldFact(value=_collection(edge, mixed), basis=Basis.OFFICIAL_RECORD),
        "empty": HeldFact(value=_collection(), basis=Basis.OFFICIAL_RECORD),
    })


def _select(source: str, predicate: dict):
    return asyncio.run(dispatch(
        {"tool": "compute", "as": "n", "args": {"pipelines": [
            {"name": "n", "source": source, "stages": [{"op": "select", "predicate": predicate}]}]}},
        _context()))


class TestTheWrongCollectionIsCaught:
    def test_the_live_mistake(self) -> None:
        outcome = _select("course", {"path": "requires", "op": "=", "value": "00940224"})
        assert outcome.defects, "a filter that cannot be applied must not read as 'none matched'"

    def test_the_message_says_what_went_wrong(self) -> None:
        message = list(_select("course", {"path": "group", "op": "=", "value": "g1"}).defects.values())[0].message
        assert "on no record of this collection" in message
        assert "not the same as" in message

    def test_a_nested_predicate_is_walked(self) -> None:
        """The eligibility check filters with an `and` of two comparisons, so a
        top-level-only check would miss both."""
        outcome = _select("course", {"and": [
            {"path": "courseNumber", "op": "=", "value": "00960211"},
            {"path": "group", "op": "=", "value": "g1"},
        ]})
        assert outcome.defects


class TestGenuineNonMatchesStillPass:
    def test_the_right_collection_filters_normally(self) -> None:
        outcome = _select("edges", {"path": "requires", "op": "=", "value": "00999999"})
        assert not outcome.defects
        assert len(outcome.facts["n"].value.records) == 0

    def test_a_field_on_only_some_records_is_ordinary_filtering(self) -> None:
        outcome = _select("some_have_it", {"path": "requires", "op": "=", "value": "00999999"})
        assert not outcome.defects, "partial presence is a real filter, not a mistake"

    def test_an_empty_collection_matches_nothing_without_complaint(self) -> None:
        outcome = _select("empty", {"path": "anything", "op": "=", "value": "x"})
        assert not outcome.defects

    def test_a_matching_filter_is_untouched(self) -> None:
        outcome = _select("edges", {"path": "requires", "op": "=", "value": "00940224"})
        assert not outcome.defects
        assert len(outcome.facts["n"].value.records) == 1
