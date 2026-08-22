"""A correct refusal that told the model there was nothing it could do.

Summing `credits` over a student's remaining courses fails, correctly: some of
those courses have no catalog row, so a sum covers 11 of 15 while carrying the
confidence of a sum over all 15. That is the silent-partial failure the guard
exists for.

The message then ended:

    No edit to this pipeline can fix that -- the missing values were never
    retrieved.

which is a dead end. Live, `semesters_to_graduate` hit it twice in one run and
spent the remaining turns rephrasing. There IS a legal move -- two of them --
and a refusal that does not name one is the shape of unfollowable advice this
codebase has now hit four times: the predicate literal that demanded the `kind`
it rejected, the refusal naming facts the model did not hold, the silent
`compute`, and this.
"""

from __future__ import annotations

import asyncio

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

Q = ScalarKind.QUANTITY
I = ScalarKind.IDENTIFIER


def _sum_credits(*records: Record):
    collection = Collection(records=records,
                            completeness=Completeness(complete=True, total=len(records)))
    context = DispatchContext(facts={
        "r": HeldFact(value=collection, basis=Basis.OFFICIAL_RECORD)})
    return asyncio.run(dispatch({"tool": "compute", "as": "n", "args": {"pipelines": [
        {"name": "n", "source": "r",
         "stages": [{"op": "aggregate", "agg": "sum", "field": "credits"}]}]}}, context))


WITH = Record(fields={"c": Scalar(I, "a"), "credits": Scalar(Q, 3.0)},
              basis=Basis.OFFICIAL_RECORD)
WITHOUT = Record(fields={"c": Scalar(I, "b")}, basis=Basis.OFFICIAL_RECORD)


class TestItStillRefuses:
    def test_a_partial_sum_is_not_returned(self) -> None:
        outcome = _sum_credits(WITH, WITHOUT)
        assert outcome.defects and not outcome.facts

    def test_it_says_how_many_are_missing(self) -> None:
        message = list(_sum_credits(WITH, WITHOUT).defects.values())[0].message
        assert "1 of 2" in message


class TestTheRefusalNamesALegalMove:
    def test_it_offers_the_explicit_partial(self) -> None:
        message = list(_sum_credits(WITH, WITHOUT).defects.values())[0].message
        assert "`select` the records that HAVE" in message

    def test_it_offers_counting_instead(self) -> None:
        message = list(_sum_credits(WITH, WITHOUT).defects.values())[0].message
        assert "report the COUNT instead" in message

    def test_it_no_longer_says_nothing_can_be_done(self) -> None:
        message = list(_sum_credits(WITH, WITHOUT).defects.values())[0].message
        assert "No edit to this pipeline can fix" not in message


class TestTheOrdinaryCasesAreUnchanged:
    def test_a_complete_collection_sums(self) -> None:
        outcome = _sum_credits(WITH, WITH)
        assert not outcome.defects
        assert outcome.facts["n"].value.value == 6.0

    def test_a_collection_with_no_values_at_all_still_says_so(self) -> None:
        outcome = _sum_credits(WITHOUT, WITHOUT)
        assert outcome.defects
        assert "nothing to aggregate" in list(outcome.defects.values())[0].message
