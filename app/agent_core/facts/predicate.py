"""The predicate grammar -- phase 2 of docs/agent/tools_implementation_plan.md.

One closed grammar, evaluated in two places: pushed down to Mongo by `find`, and
in memory by `select`. Closed means no expression strings and no `eval` -- a
predicate is a small tree of typed nodes that compiles structurally to a Mongo
filter document, so there is no operator-injection surface to defend.

The two engines must agree. Where Mongo's semantics are surprising, the
in-memory engine follows Mongo rather than the other way round, because Mongo's
behaviour is the one we cannot change:

  - a comparison against a MISSING field does not match (it is not an error)
  - consequently `Not(comparison)` DOES match a record missing the field, which
    is what `$nor` returns
  - `contains` is array membership, which Mongo spells as plain equality

Where Mongo's semantics are actively unsafe, the grammar refuses to emit them
instead of imitating them: Mongo orders values ACROSS BSON types, so
`grade > "ninety"` quietly returns a result there. `validate` rejects it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Union

from app.agent_core.facts.types import Record, Scalar, ScalarKind


class PredicateTypeError(Exception):
    """A predicate that must not be evaluated -- the fault is in the expression."""


class Op(Enum):
    EQ = "="
    NE = "≠"
    LT = "<"
    LE = "≤"
    GT = ">"
    GE = "≥"
    IN = "in"
    CONTAINS = "contains"


_ORDERING_OPS = frozenset({Op.LT, Op.LE, Op.GT, Op.GE})

# Only these can be ranked. Ordering identifiers is meaningless -- course codes
# have no order -- and permitting it invites summing them next.
_ORDERABLE_KINDS = frozenset({ScalarKind.QUANTITY, ScalarKind.DATE})

_MONGO_OPS = {
    Op.EQ: "$eq",
    Op.NE: "$ne",
    Op.LT: "$lt",
    Op.LE: "$lte",
    Op.GT: "$gt",
    Op.GE: "$gte",
}

_EXPR_OPS = {Op.EQ: "$eq", Op.NE: "$ne", Op.LT: "$lt", Op.LE: "$lte", Op.GT: "$gt", Op.GE: "$gte"}

MISSING = object()
"""Absence sentinel. Shared with the runner so both agree on what 'missing' is."""


@dataclass(frozen=True)
class Path:
    segments: tuple[str, ...]

    @classmethod
    def parse(cls, dotted: str) -> "Path":
        parts = tuple(part for part in dotted.split(".") if part)
        if not parts:
            raise PredicateTypeError(f"empty path: {dotted!r}")
        return cls(segments=parts)

    @property
    def dotted(self) -> str:
        return ".".join(self.segments)

    def resolve(self, record: Record) -> Any:
        """The value at this path, or `MISSING`. Never raises on absence.

        EXACT FIELD NAME FIRST, then segment walking. Qualified names and nested
        paths both spell themselves with a dot, and `join` deliberately produces
        flat fields called `left.credits`. Walking segments first would hunt for
        a nested record named `left`, find nothing, and make a join's own output
        unreadable by the stage immediately after it.

        A qualified name is a NAME -- the dot belongs to it -- where a path is a
        route. When both could apply, the name wins: deterministic, and it keeps
        `join` composable.

        This is the ONLY implementation of the rule. The runner delegates here
        rather than keeping its own copy, because two resolvers for one rule is
        the same silent-drift trap as two predicate engines.
        """
        exact = record.fields.get(self.dotted, MISSING)
        if exact is not MISSING:
            return exact

        current: Any = record
        for segment in self.segments:
            if isinstance(current, Record):
                current = current.fields.get(segment, MISSING)
            elif isinstance(current, dict):
                current = current.get(segment, MISSING)
            else:
                return MISSING
            if current is MISSING:
                return MISSING
        return current


@dataclass(frozen=True)
class FactRef:
    """A criterion whose value comes from a HELD FACT rather than being typed.

    The gap the first live runs exposed: a model must filter `userId` by the
    student's id, which it holds as a fact, and there was no way to say so. It
    wrote the fact's NAME as a string literal, which matched nothing -- silently,
    because a filter that matches nothing is a legitimate empty result.

    Strictly better than the literal that §3.2 already permits in criterion
    position: a literal is typed by the model, this is grounded in a fact.
    Resolved at dispatch, where the working set is in scope.

    With `field` set, it names not one value but the SET of values a collection
    fact holds at that field -- a SEMI-JOIN. "Fetch the courses whose `_id` is in
    my completed transcript" is `_id IN {"fact": "completed", "field":
    "courseId"}`. Two live evals stalled for want of exactly this: the model held
    a collection of ids and had no way to fetch the records they pointed at
    except by pulling the entire 2,613-row catalog and joining in memory. The
    set is resolved from the held fact at dispatch, so the model never types the
    ids and the grounding is preserved.
    """

    name: str
    field: str | None = None


@dataclass(frozen=True)
class Comparison:
    path: Path
    op: Op
    value: Union[Scalar, Path, FactRef, tuple[Scalar, ...]]


@dataclass(frozen=True)
class And:
    terms: tuple["Predicate", ...]


@dataclass(frozen=True)
class Or:
    terms: tuple["Predicate", ...]


@dataclass(frozen=True)
class Not:
    term: "Predicate"


@dataclass(frozen=True)
class Always:
    """The constant-true predicate.

    Required, not decorative: `join` expresses Cartesian product only through a
    constant-true predicate, and without product the basis is not relationally
    complete (plan §3.3).
    """


Predicate = Union[Comparison, And, Or, Not, Always]


def validate(predicate: Predicate) -> None:
    """Raise `PredicateTypeError` on a predicate that must not be evaluated.

    Only checks what is knowable WITHOUT data -- essentially, that an ordering
    operator is applied to an orderable literal. Whether a field actually holds
    a collection, or whether two compared fields share a kind, needs the
    collection's types and belongs to the pipeline type checker (phase 3).
    """
    if isinstance(predicate, Always):
        return
    if isinstance(predicate, Not):
        validate(predicate.term)
        return
    if isinstance(predicate, (And, Or)):
        if not predicate.terms:
            raise PredicateTypeError(f"{type(predicate).__name__} needs at least one term")
        for term in predicate.terms:
            validate(term)
        return

    if predicate.op is Op.IN:
        if not isinstance(predicate.value, tuple) or not predicate.value:
            raise PredicateTypeError("`in` needs a non-empty tuple of scalars")
        return

    if isinstance(predicate.value, tuple):
        raise PredicateTypeError(f"a tuple value is only valid for `in`, not {predicate.op.value}")

    if predicate.op in _ORDERING_OPS and isinstance(predicate.value, Scalar):
        if predicate.value.kind not in _ORDERABLE_KINDS:
            raise PredicateTypeError(
                f"{predicate.op.value} needs an orderable value (quantity or date), "
                f"got {predicate.value.kind.value}. Values of that kind support only = and ≠."
            )


def matches(predicate: Predicate, record: Record) -> bool:
    """Evaluate in memory. Total: never raises on absent or ill-typed data."""
    if isinstance(predicate, Always):
        return True
    if isinstance(predicate, Not):
        return not matches(predicate.term, record)
    if isinstance(predicate, And):
        return all(matches(term, record) for term in predicate.terms)
    if isinstance(predicate, Or):
        return any(matches(term, record) for term in predicate.terms)

    left = predicate.path.resolve(record)
    if left is MISSING:
        return False

    if predicate.op is Op.CONTAINS:
        if not isinstance(left, (tuple, list)):
            return False
        wanted = _raw(predicate.value)
        return any(_raw(item) == wanted for item in left)

    if predicate.op is Op.IN:
        assert isinstance(predicate.value, tuple)
        return any(_raw(left) == _raw(candidate) for candidate in predicate.value)

    if isinstance(predicate.value, Path):
        right = predicate.value.resolve(record)
        if right is MISSING:
            return False
    else:
        right = predicate.value

    return _compare(_raw(left), predicate.op, _raw(right))


def _raw(value: Any) -> Any:
    return value.value if isinstance(value, Scalar) else value


def _compare(left: Any, op: Op, right: Any) -> bool:
    if op is Op.EQ:
        return bool(left == right)
    if op is Op.NE:
        return bool(left != right)
    try:
        if op is Op.LT:
            return bool(left < right)
        if op is Op.LE:
            return bool(left <= right)
        if op is Op.GT:
            return bool(left > right)
        return bool(left >= right)
    except TypeError:
        # Two fields of incompatible kinds. A data problem, not an expression
        # one -- raising here would abort a whole scan over one bad record, so
        # the record simply does not match. `validate` catches the case that IS
        # statically knowable (an unorderable literal).
        return False


def compile_to_mongo(predicate: Predicate) -> dict[str, Any]:
    """Compile to a Mongo filter document.

    Structural only: operator names come from a fixed table and values are
    carried through as data, so nothing the model writes can become a query
    operator.
    """
    if isinstance(predicate, Always):
        return {}
    if isinstance(predicate, And):
        return {"$and": [compile_to_mongo(term) for term in predicate.terms]}
    if isinstance(predicate, Or):
        return {"$or": [compile_to_mongo(term) for term in predicate.terms]}
    if isinstance(predicate, Not):
        # `$not` is field-scoped and cannot negate a composite expression;
        # `$nor` over a single term is expression-level negation.
        return {"$nor": [compile_to_mongo(predicate.term)]}

    field = predicate.path.dotted

    if predicate.op is Op.IN:
        assert isinstance(predicate.value, tuple)
        return {field: {"$in": [_raw(item) for item in predicate.value]}}

    if predicate.op is Op.CONTAINS:
        # Mongo matches an array field against a scalar by membership, so plain
        # equality IS containment here. Deliberately not a regex: compiling
        # model-supplied text into a pattern is both an injection surface and a
        # divergence from the in-memory engine, so text-contains is excluded.
        return {field: {"$eq": _raw(predicate.value)}}

    if isinstance(predicate.value, Path):
        return {"$expr": {_EXPR_OPS[predicate.op]: [f"${field}", f"${predicate.value.dotted}"]}}

    return {field: {_MONGO_OPS[predicate.op]: _raw(predicate.value)}}


def collect_paths(predicate: Predicate) -> tuple[Path, ...]:
    """Every path a predicate reads. Used by the pipeline checker to verify the
    fields exist before a scan begins."""
    if isinstance(predicate, Always):
        return ()
    if isinstance(predicate, Not):
        return collect_paths(predicate.term)
    if isinstance(predicate, (And, Or)):
        return tuple(path for term in predicate.terms for path in collect_paths(term))
    referenced = [predicate.path]
    if isinstance(predicate.value, Path):
        referenced.append(predicate.value)
    return tuple(referenced)


__all__ = [
    "Always",
    "And",
    "Comparison",
    "FactRef",
    "Not",
    "Op",
    "Or",
    "Path",
    "Predicate",
    "PredicateTypeError",
    "collect_paths",
    "compile_to_mongo",
    "matches",
    "validate",
]
