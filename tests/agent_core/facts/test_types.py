"""Phase 1 gate for docs/agent/tools_implementation_plan.md.

Two properties the whole algebra rests on:

  1. Certainty basis is a TOTAL order with `simulated` weakest (§3.7), so a
     hypothesis taints everything derived from it without needing a
     counterfactual primitive.
  2. Completeness propagates per the §4.1 table -- and in particular
     `difference` is ASYMMETRIC: an incomplete subtrahend is unsound, not
     merely partial.
"""

from __future__ import annotations

import pytest

from app.agent_core.facts.types import (
    Basis,
    Collection,
    Completeness,
    InputRole,
    Record,
    Refusal,
    Scalar,
    ScalarKind,
    completeness_after,
    weakest,
)


class TestBasisOrdering:
    def test_official_record_is_strongest_and_simulated_is_weakest(self) -> None:
        ordered = sorted(Basis, key=lambda b: b.strength, reverse=True)
        assert ordered[0] is Basis.OFFICIAL_RECORD
        assert ordered[-1] is Basis.SIMULATED

    def test_strength_is_a_total_order(self) -> None:
        strengths = [b.strength for b in Basis]
        assert len(set(strengths)) == len(strengths), "two bases share a rank; ordering is not total"

    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            (Basis.OFFICIAL_RECORD, Basis.WIKI_DERIVED, Basis.WIKI_DERIVED),
            (Basis.SIMULATED, Basis.OFFICIAL_RECORD, Basis.SIMULATED),
            (Basis.WIKI_DERIVED, Basis.WIKI_DERIVED, Basis.WIKI_DERIVED),
        ],
    )
    def test_weakest_picks_the_lower_rank(self, left: Basis, right: Basis, expected: Basis) -> None:
        assert weakest([left, right]) is expected

    def test_a_simulated_input_taints_the_result(self) -> None:
        """§2.1: this is what removes the need for a counterfactual primitive."""
        assert weakest([Basis.OFFICIAL_RECORD, Basis.OFFICIAL_RECORD, Basis.SIMULATED]) is Basis.SIMULATED

    def test_weakest_of_nothing_is_refused_rather_than_defaulted(self) -> None:
        with pytest.raises(ValueError):
            weakest([])


class TestScalarTyping:
    def test_a_course_code_stays_an_identifier_despite_looking_numeric(self) -> None:
        """The whole point of the Quantity/Identifier split (§3.1): no string-shape
        heuristic decides this, the type does."""
        code = Scalar(kind=ScalarKind.IDENTIFIER, value="00940224")
        assert code.kind is ScalarKind.IDENTIFIER
        assert not code.is_quantity

    def test_a_credit_value_is_a_quantity(self) -> None:
        credits = Scalar(kind=ScalarKind.QUANTITY, value=3.5)
        assert credits.is_quantity

    def test_a_quantity_must_hold_a_number(self) -> None:
        with pytest.raises(ValueError):
            Scalar(kind=ScalarKind.QUANTITY, value="3.5")

    def test_scalars_are_immutable(self) -> None:
        value = Scalar(kind=ScalarKind.QUANTITY, value=1)
        with pytest.raises(Exception):
            value.value = 2  # type: ignore[misc]


class TestCompletenessPropagation:
    COMPLETE = Completeness(complete=True, total=10)
    PARTIAL = Completeness(complete=False, total=73)

    def test_monotone_op_passes_incompleteness_through(self) -> None:
        result = completeness_after((InputRole.MONOTONE,), (self.PARTIAL,))
        assert isinstance(result, Completeness)
        assert result.complete is False

    def test_monotone_op_over_complete_input_stays_complete(self) -> None:
        result = completeness_after((InputRole.MONOTONE,), (self.COMPLETE,))
        assert isinstance(result, Completeness)
        assert result.complete is True

    def test_join_is_incomplete_when_either_side_is(self) -> None:
        roles = (InputRole.MONOTONE, InputRole.MONOTONE)
        result = completeness_after(roles, (self.COMPLETE, self.PARTIAL))
        assert isinstance(result, Completeness)
        assert result.complete is False

    def test_aggregate_over_incomplete_fails_closed(self) -> None:
        """A count over a truncated page is confidently wrong -- the failure the
        grounding invariant could not see."""
        result = completeness_after((InputRole.REQUIRES_ALL,), (self.PARTIAL,))
        assert isinstance(result, Refusal)
        assert "73" in result.reason, "refusal must name the true total so it is actionable"

    def test_aggregate_over_complete_input_is_allowed(self) -> None:
        result = completeness_after((InputRole.REQUIRES_ALL,), (self.COMPLETE,))
        assert isinstance(result, Completeness)

    def test_difference_tolerates_an_incomplete_minuend(self) -> None:
        """A incomplete -> the answer is partial, which is honest."""
        roles = (InputRole.MONOTONE, InputRole.REQUIRES_ALL)
        result = completeness_after(roles, (self.PARTIAL, self.COMPLETE))
        assert isinstance(result, Completeness)
        assert result.complete is False

    def test_difference_refuses_an_incomplete_subtrahend(self) -> None:
        """B incomplete -> UNSOUND, not partial. Every record missing from B is
        wrongly RETAINED, so 'requirements remaining' silently gains courses the
        student already passed. This asymmetry is the subtle one."""
        roles = (InputRole.MONOTONE, InputRole.REQUIRES_ALL)
        result = completeness_after(roles, (self.COMPLETE, self.PARTIAL))
        assert isinstance(result, Refusal)


class TestCollectionCarriesCompleteness:
    def test_a_collection_knows_whether_it_is_all_of_what_was_asked_for(self) -> None:
        page = Collection(
            records=(Record(fields={"id": Scalar(ScalarKind.IDENTIFIER, "00940224")}, basis=Basis.OFFICIAL_RECORD),),
            completeness=Completeness(complete=False, total=73),
        )
        assert page.completeness.complete is False
        assert page.completeness.total == 73

    def test_completeness_cannot_be_optimistically_assumed(self) -> None:
        """§4.1: a collection whose completeness is unknown is NOT complete."""
        unknown = Completeness.unknown()
        assert unknown.complete is False
