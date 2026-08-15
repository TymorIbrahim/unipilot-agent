"""`forecast` answered "will it run next spring" with the wrong denominator.

Measured on real offerings: course 00140709 ran in spring in 3 of the 3 years on
record -- every spring it could have -- and forecast said it would NOT run.
Counting `spring offerings / all offerings` asks what share of a course's
offerings were spring, which is a different question. A course offered in all
three terms scores ~0.33 for each, so the most reliably offered courses got the
most pessimistic forecast, and the error told students a course was unavailable
when it always runs.
"""

from __future__ import annotations

from app.agent_core.facts.forecast import forecast
from app.agent_core.facts.predicate import Path
from app.agent_core.facts.types import (
    Basis,
    Collection,
    Completeness,
    Record,
    Scalar,
    ScalarKind,
)

I = ScalarKind.IDENTIFIER
PERIOD = Path(("term",))
CYCLE = Path(("year",))


def _history(*pairs: tuple[str, str]) -> Collection:
    return Collection(
        records=tuple(
            Record(
                fields={"term": Scalar(I, term), "year": Scalar(I, year)},
                basis=Basis.OFFICIAL_RECORD,
            )
            for term, year in pairs
        ),
        completeness=Completeness(complete=True, total=len(pairs)),
    )


# The live shape of 00140709: three years, every term, spring in all three.
EVERY_TERM = _history(
    ("spring", "2023"), ("winter", "2023"), ("spring", "2024"), ("summer", "2024"),
    ("winter", "2024"), ("spring", "2025"), ("summer", "2025"),
)


class TestTheRegression:
    def test_a_course_that_ran_every_spring_is_forecast_to_run(self) -> None:
        result = forecast(EVERY_TERM, period_path=PERIOD, target="spring", cycle_path=CYCLE)
        assert result.rate == 1.0, "it ran in spring in every year on record"
        assert result.value.value is True

    def test_the_old_denominator_is_what_inverted_it(self) -> None:
        """Kept as documentation of the defect: without a cycle the same history
        scores below half and the answer flips."""
        result = forecast(EVERY_TERM, period_path=PERIOD, target="spring")
        assert result.rate < 0.5
        assert result.value.value is False


class TestCycleCounting:
    def test_a_term_missed_in_one_year_of_three(self) -> None:
        result = forecast(EVERY_TERM, period_path=PERIOD, target="summer", cycle_path=CYCLE)
        assert round(result.rate, 2) == 0.67, "summer ran in 2024 and 2025, not 2023"
        assert result.value.value is True

    def test_a_term_never_offered_stays_false(self) -> None:
        history = _history(("spring", "2023"), ("spring", "2024"), ("spring", "2025"))
        result = forecast(history, period_path=PERIOD, target="winter", cycle_path=CYCLE)
        assert result.rate == 0.0
        assert result.value.value is False

    def test_twice_in_one_year_is_still_one_year(self) -> None:
        """One vote per cycle. Two spring offerings in 2023 must not make 2023
        count twice and drown out the years it did not run."""
        history = _history(
            ("spring", "2023"), ("spring", "2023"), ("winter", "2024"), ("winter", "2025")
        )
        result = forecast(history, period_path=PERIOD, target="spring", cycle_path=CYCLE)
        assert round(result.rate, 2) == 0.33, "spring ran in 1 of 3 years"
        assert result.value.value is False

    def test_too_few_cycles_is_refused(self) -> None:
        """Three offerings in two years is not three observations of a yearly
        pattern -- the guard must count cycles, not rows."""
        history = _history(("spring", "2024"), ("winter", "2024"), ("spring", "2025"))
        result = forecast(history, period_path=PERIOD, target="spring", cycle_path=CYCLE)
        assert not hasattr(result, "rate"), "a 2-cycle history must be refused"
        assert "cycle" in result.message
