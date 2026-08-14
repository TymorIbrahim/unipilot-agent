"""`forecast` -- projection from a history. Phase 7 of docs/agent/tools_implementation_plan.md.

COUNTING how often something happened is algebra. EXTRAPOLATING from it is not:
no selection, join or aggregation turns a record of the past into a claim about
a period that has not happened yet. That inference step is the boundary, and it
is the whole reason this is a primitive rather than a pipeline.

It carries its own basis. A projection is not an observation, and
`predicted_pattern` sits below `wiki_derived`, so anything derived from a
forecast is visibly weaker than anything derived from a record -- automatically,
through the ordinary weakest-input rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from app.agent_core.facts.operators import DataDefect
from app.agent_core.facts.predicate import Path
from app.agent_core.facts.types import Basis, Collection, Record, Scalar, ScalarKind

MIN_OBSERVATIONS = 3
"""Below this, a rate is noise. Refusing beats reporting a pattern from two
data points with a number attached that makes it look considered."""

_SATURATES_AT = 8.0
_CEILING = 0.95


@dataclass(frozen=True)
class Forecast:
    value: Scalar
    confidence: float
    observations: int
    rate: float
    basis: Basis = Basis.PREDICTED_PATTERN


def forecast(
    observations: Collection,
    *,
    period_path: Path,
    target: str,
    min_observations: int = MIN_OBSERVATIONS,
) -> Union[Forecast, DataDefect]:
    """Predict whether `target` recurs, from the periods actually observed."""
    if not observations.completeness.complete:
        # The completeness bug wearing a different hat: a rate computed over
        # whatever happened to be fetched, reported as though it were the whole
        # history. Worse than a partial count, because the output is a claim
        # about the FUTURE and nothing downstream can tell it was based on part
        # of the past.
        return DataDefect(
            0,
            "cannot forecast from an incomplete history: the pattern would be computed over "
            f"only the records that were fetched (of {observations.completeness.total or 'unknown'} "
            "total) and reported as though it were the whole record.",
        )

    periods = [_period(period_path, record) for record in observations.records]
    present = [p for p in periods if p is not None]

    if len(present) < min_observations:
        return DataDefect(
            0,
            f"only {len(present)} usable observation(s); at least {min_observations} are needed "
            "before a rate means anything. A pattern from fewer is noise with a number attached.",
        )

    hits = sum(1 for period in present if period == target)
    rate = hits / len(present)

    return Forecast(
        value=Scalar(ScalarKind.BOOL, rate >= 0.5),
        confidence=_confidence(rate, len(present)),
        observations=len(present),
        rate=rate,
    )


def _confidence(rate: float, sample: int) -> float:
    """Confidence from how EXTREME the rate is and how much was seen.

    Both terms are needed. A rate of 1.0 from three observations is not the same
    claim as a rate of 1.0 from thirty, and a rate of 0.5 is a coin flip however
    long it has been watched -- so extremity alone would report high confidence
    on the least informative history there is.

    Never reaches 1.0: a projection is never certain, and a formula that can
    print 1.0 will eventually print it.
    """
    extremity = abs(rate - 0.5) * 2
    saturation = min(1.0, sample / _SATURATES_AT)
    return round(0.5 + (_CEILING - 0.5) * extremity * saturation, 3)


def _period(path: Path, record: Record) -> str | None:
    resolved = path.resolve(record)
    return resolved.value if isinstance(resolved, Scalar) else None


__all__ = ["MIN_OBSERVATIONS", "Forecast", "forecast"]
