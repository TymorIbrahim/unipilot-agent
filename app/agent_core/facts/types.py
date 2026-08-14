"""The fact type system -- phase 1 of docs/agent/tools_implementation_plan.md.

Everything the algebra asserts against lives here. Three ideas carry their
weight:

`Quantity` vs `Identifier` is a TYPE, not a string shape. A course code is an
identifier that happens to be digits; typing it is what makes summing course
codes a type error instead of something a leading-zero heuristic has to guess at.

Certainty travels with the RECORD's origin, not with a field-name map, because a
`union` of two collections can carry the same field name with different
provenance on each side -- unrepresentable in a map keyed by field name.

Completeness is a property of a COLLECTION, distinct from the certainty of the
facts inside it. Every record on a truncated page can be individually perfect
while a `count` over that page is confidently wrong. Per-fact certainty cannot
express that, which is why `complete`/`total` ride on the collection itself.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from numbers import Number
from typing import Any, Union


class Basis(Enum):
    """How a fact came to be known, ordered strongest to weakest.

    The order is what makes hypotheticals work without a counterfactual
    primitive (plan §2.1): `SIMULATED` is weakest, so injecting a hypothesised
    record taints every fact derived from it through the ordinary weakest-input
    rule. Nothing special-cases a what-if.

    `LIVE_MOODLE` sits just below `OFFICIAL_RECORD`: a grade or enrolment scraped
    live from Moodle comes from an official Technion system, but is obtained by
    browser observation -- a page can be stale, mis-parsed, or show a provisional
    mark the registrar has not finalised -- so it is deliberately less certain
    than the registrar's authoritative record, and anything derived by joining a
    Moodle fact with a registrar fact correctly carries the weaker Moodle basis.
    These values are strength RANKS, not labels or a wire format: nothing
    persists a basis by its integer (serialization goes through `.label`), which
    is what makes renumbering to insert a tier safe.
    """

    OFFICIAL_RECORD = 6
    LIVE_MOODLE = 5
    WIKI_DERIVED = 4
    LLM_INTERPRETATION = 3
    PREDICTED_PATTERN = 2
    SIMULATED = 1

    @property
    def strength(self) -> int:
        return self.value

    @property
    def label(self) -> str:
        """The human-readable name.

        Exists so nothing reaches for `.value`, which is the strength RANK here
        rather than a label -- an easy trap, since most enums in this codebase
        carry their string in `.value`.
        """
        return self.name.lower()


def weakest(bases: Sequence[Basis]) -> Basis:
    """The weakest basis among `bases` -- the certainty of anything derived from them.

    Empty input raises rather than defaulting: a derived fact with no inputs has
    no honest basis to claim, and silently answering `OFFICIAL_RECORD` there
    would launder an ungrounded value into the strongest tag we have.
    """
    if not bases:
        raise ValueError("weakest() of no bases: a derived fact must consume at least one input")
    return min(bases, key=lambda basis: basis.strength)


class ScalarKind(Enum):
    QUANTITY = "quantity"
    IDENTIFIER = "identifier"
    TEXT = "text"
    BOOL = "bool"
    DATE = "date"


@dataclass(frozen=True)
class Scalar:
    """A typed leaf value.

    The `QUANTITY` check is deliberately strict. Accepting the string "3.5" here
    would reintroduce exactly the ambiguity the type system exists to remove --
    once a numeric string is a quantity, "00940224" has to be excluded by
    spelling, and a course code becomes one bad heuristic away from being
    summed. Conversion belongs at admission, where the source schema is known.
    """

    kind: ScalarKind
    value: Any

    def __post_init__(self) -> None:
        if self.kind is ScalarKind.QUANTITY and not _is_number(self.value):
            raise ValueError(
                f"QUANTITY must hold a number, got {type(self.value).__name__} {self.value!r}. "
                "Convert at admission, where the source schema says whether this is a quantity."
            )
        if self.kind is ScalarKind.BOOL and not isinstance(self.value, bool):
            raise ValueError(f"BOOL must hold a bool, got {type(self.value).__name__}")
        if self.kind is ScalarKind.DATE and not isinstance(self.value, (date, datetime)):
            raise ValueError(f"DATE must hold a date, got {type(self.value).__name__}")

    @property
    def is_quantity(self) -> bool:
        return self.kind is ScalarKind.QUANTITY


def _is_number(value: Any) -> bool:
    return isinstance(value, Number) and not isinstance(value, bool)


FieldValue = Union[Scalar, "Record", "Collection"]


@dataclass(frozen=True)
class Record:
    """A record and the basis on which it is known.

    `basis` is the record's own origin. `field_basis` overrides it for
    individual fields, and exists for exactly one reason: a JOINED record has
    two origins. Without per-field overrides, summing an `official_record` field
    after joining against a wiki-derived collection would report `wiki_derived`,
    and provenance would never recover from a single join.

    This is per-RECORD, not a collection-level field map -- a collection-level
    map cannot represent a `union` where the same field name carries different
    provenance on each side.
    """

    fields: Mapping[str, FieldValue]
    basis: Basis
    field_basis: Mapping[str, Basis] = field(default_factory=dict)

    def basis_for(self, name: str) -> Basis:
        """The basis of one field: its own override, else the record's."""
        return self.field_basis.get(name, self.basis)


@dataclass(frozen=True)
class Completeness:
    """Whether a collection is ALL of what was asked for.

    `total` is the true count at the source when it is known. It is what makes a
    refusal actionable: "50 of 73" tells the caller the shape of the problem,
    where a bare "incomplete" does not.
    """

    complete: bool
    total: int | None = None

    @classmethod
    def unknown(cls) -> "Completeness":
        """Completeness that could not be established.

        Never optimistically complete. Guessing wrong in this direction produces
        a confidently wrong count rather than a visible error, which is the
        failure this whole mechanism exists to prevent.
        """
        return cls(complete=False, total=None)

    @classmethod
    def whole(cls, total: int | None = None) -> "Completeness":
        return cls(complete=True, total=total)


@dataclass(frozen=True)
class Collection:
    records: tuple[Record, ...] = ()
    completeness: Completeness = field(default_factory=Completeness.unknown)


@dataclass(frozen=True)
class Refusal:
    """A fail-closed outcome: the operation is unsound on these inputs.

    Distinct from "the expression is wrong". No edit to the pipeline fixes a
    refusal -- the data does not support the operation -- so a repair loop must
    branch on it rather than retry, or it burns its budget re-deriving a
    pipeline that was already correct.
    """

    reason: str


class InputRole(Enum):
    """How an operator's input position reacts to an incomplete collection.

    Declared per position rather than per operator because `difference` treats
    its two inputs differently, and that asymmetry is the easiest thing here to
    get silently wrong.
    """

    MONOTONE = "monotone"
    """Incompleteness passes through: fewer inputs, fewer outputs, still honest."""

    REQUIRES_ALL = "requires_all"
    """Incompleteness makes the result WRONG rather than partial. Fail closed."""


def completeness_after(
    roles: Sequence[InputRole],
    inputs: Sequence[Completeness],
) -> Completeness | Refusal:
    """Completeness of a result, or a refusal when the operation would be unsound.

    The `difference` case is the one worth stating plainly. With roles
    `(MONOTONE, REQUIRES_ALL)`:

      - an incomplete MINUEND yields a partial answer -- some of the records
        that should be considered were never seen, so some of the output is
        missing. Honest, and flagged.
      - an incomplete SUBTRAHEND is UNSOUND. Every record missing from it is
        wrongly RETAINED in the output, so "which requirements remain" silently
        gains courses the student has already passed -- and every fact in that
        answer still reports full confidence.

    Partial and wrong are not the same failure, and only one of them is
    survivable.
    """
    if len(roles) != len(inputs):
        raise ValueError(f"{len(roles)} roles for {len(inputs)} inputs")

    for position, (role, incoming) in enumerate(zip(roles, inputs)):
        if role is InputRole.REQUIRES_ALL and not incoming.complete:
            seen = "an unknown number of" if incoming.total is None else f"fewer than {incoming.total}"
            raise_reason = (
                f"input {position} is incomplete ({seen} records) and this operation "
                "requires all of them; the result would be wrong rather than partial"
            )
            return Refusal(reason=raise_reason)

    # Derived collections do not restate the source `total`: after a filter or a
    # join it no longer describes this collection. Completeness survives; the
    # count does not.
    return Completeness(complete=all(incoming.complete for incoming in inputs), total=None)


__all__ = [
    "Basis",
    "Collection",
    "Completeness",
    "FieldValue",
    "InputRole",
    "Record",
    "Refusal",
    "Scalar",
    "ScalarKind",
    "completeness_after",
    "weakest",
]
