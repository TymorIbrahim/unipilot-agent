"""`forecast` -- phase 7 of docs/agent/tools_implementation_plan.md.

The inference boundary. COUNTING how often something happened is algebra;
EXTRAPOLATING from it is not -- no amount of selection, joining or aggregation
turns a history into a claim about a period that has not happened yet.

Which is exactly why it gets its own basis. A projection is not an observation,
and `predicted_pattern` sits below `wiki_derived` so anything computed from a
forecast is visibly weaker than anything computed from a record.
"""

from __future__ import annotations

from app.agent_core.facts.forecast import forecast
from app.agent_core.facts.operators import DataDefect, Pipeline, Stage
from app.agent_core.facts.predicate import Path
from app.agent_core.facts.runner import run_pipelines
from app.agent_core.facts.types import (
    Basis,
    Collection,
    Completeness,
    Record,
    Scalar,
    ScalarKind,
)

I = ScalarKind.IDENTIFIER
PERIOD = Path.parse("period")


def _observed(*periods: str, complete: bool = True) -> Collection:
    return Collection(
        records=tuple(
            Record(fields={"period": Scalar(I, period)}, basis=Basis.OFFICIAL_RECORD)
            for period in periods
        ),
        completeness=Completeness(complete=complete, total=len(periods)),
    )


WINTER_AND_SPRING = _observed("winter", "spring", "winter", "spring", "winter", "spring")


class TestProjection:
    def test_a_never_observed_period_is_predicted_not_to_occur(self) -> None:
        result = forecast(WINTER_AND_SPRING, period_path=PERIOD, target="summer")
        assert result.value.kind is ScalarKind.BOOL
        assert result.value.value is False

    def test_an_always_observed_period_is_predicted_to_occur(self) -> None:
        result = forecast(_observed("spring", "spring", "spring", "spring"), period_path=PERIOD, target="spring")
        assert result.value.value is True

    def test_a_regular_half_the_time_pattern_is_predicted_to_occur(self) -> None:
        result = forecast(WINTER_AND_SPRING, period_path=PERIOD, target="spring")
        assert result.value.value is True


class TestConfidence:
    def test_confidence_rises_with_the_number_of_observations(self) -> None:
        few = forecast(_observed("spring", "spring", "spring"), period_path=PERIOD, target="spring")
        many = forecast(_observed(*["spring"] * 12), period_path=PERIOD, target="spring")
        assert many.confidence > few.confidence

    def test_an_ambiguous_history_is_never_highly_confident(self) -> None:
        """Half the time is a coin flip however long it has been observed, and
        reporting 0.9 on one would be worse than useless."""
        result = forecast(WINTER_AND_SPRING, period_path=PERIOD, target="spring")
        assert result.confidence < 0.8

    def test_confidence_never_reaches_certainty(self) -> None:
        result = forecast(_observed(*["spring"] * 50), period_path=PERIOD, target="spring")
        assert result.confidence < 1.0


class TestFailClosed:
    def test_too_few_observations_refuses_rather_than_guessing(self) -> None:
        result = forecast(_observed("spring"), period_path=PERIOD, target="spring")
        assert isinstance(result, DataDefect)
        assert "1" in result.message

    def test_a_truncated_history_refuses(self) -> None:
        """Forecasting from a partial history is the completeness bug wearing a
        different hat: the pattern is computed over whatever happened to be
        fetched, and reported as though it were the whole record."""
        partial = _observed("spring", "spring", "spring", "spring", complete=False)
        result = forecast(partial, period_path=PERIOD, target="spring")
        assert isinstance(result, DataDefect)

    def test_an_empty_history_refuses(self) -> None:
        assert isinstance(forecast(_observed(), period_path=PERIOD, target="spring"), DataDefect)


class TestProvenance:
    def test_a_projection_is_weaker_than_the_records_it_came_from(self) -> None:
        result = forecast(WINTER_AND_SPRING, period_path=PERIOD, target="spring")
        assert result.basis is Basis.PREDICTED_PATTERN
        assert result.basis.strength < Basis.OFFICIAL_RECORD.strength
        assert result.basis.strength < Basis.WIKI_DERIVED.strength

    def test_a_forecast_taints_everything_derived_from_it(self) -> None:
        result = forecast(WINTER_AND_SPRING, period_path=PERIOD, target="spring")
        derived = Collection(
            records=(Record(fields={"n": Scalar(ScalarKind.QUANTITY, 1.0)}, basis=result.basis),),
            completeness=Completeness(complete=True, total=1),
        )
        pipeline = Pipeline("t", "d", (Stage("aggregate", {"op": "sum", "path": Path.parse("n")}),))
        assert run_pipelines((pipeline,), {"d": derived})["t"].basis is Basis.PREDICTED_PATTERN
