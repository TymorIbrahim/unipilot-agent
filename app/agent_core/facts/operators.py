"""The operator basis and its type checker -- phase 3 of
docs/agent/tools_implementation_plan.md.

The basis is relationally complete plus aggregation and ordering. That is a
narrower claim than "can express anything" and the difference matters: it
provably cannot express transitive closure, which is why `traverse` is a
separate primitive rather than an operator here.

Checking is TABLE-DRIVEN. Each operator declares its operand types, their
grounding position, and how each collection input reacts to incompleteness --
and one generic checker reads that table. The alternative, a hand-written
validation branch per operator, is what keeps a basis small: the cost of adding
an operator becomes O(error modes) by hand, so nobody adds one, so capability
arrives instead as pre-solved shortcuts. A table row is cheap enough that the
basis can be complete.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Union

from app.agent_core.facts.predicate import Path, Predicate, collect_paths
from app.agent_core.facts.types import InputRole, Scalar, ScalarKind


class Ty(Enum):
    COLLECTION = "collection"
    SCALAR = "scalar"


class OperandPosition(Enum):
    """Where a literal is permitted (plan §3.2)."""

    DATA = "data"
    """Must be a ref. A literal here is a laundered computed value."""

    CRITERION = "criterion"
    """A literal is allowed -- the threshold came from the question."""

    STRUCTURAL = "structural"
    """A literal is required -- field paths, limits, directions. Not data."""


@dataclass(frozen=True)
class OperandSpec:
    ty: Ty
    position: OperandPosition
    role: InputRole | None = None


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    operands: tuple[OperandSpec, ...]
    result: Ty
    summary: str
    usage: str = ""
    """The smallest legal JSON stage, rendered into the prompt beside the summary.

    A one-line summary names an operator without saying how to write one, so a
    model had to discover every argument by rejection -- a turn each, for
    fourteen operators. These shapes are the ones
    `test_reachability.py::_MINIMAL_USE` parses and executes, so a change that
    breaks them fails a test rather than quietly teaching a shape that no longer
    works.
    """


def _data(ty: Ty, role: InputRole | None = None) -> OperandSpec:
    return OperandSpec(ty=ty, position=OperandPosition.DATA, role=role)


def _structural(ty: Ty) -> OperandSpec:
    return OperandSpec(ty=ty, position=OperandPosition.STRUCTURAL)


_C = Ty.COLLECTION
_S = Ty.SCALAR
_MONO = InputRole.MONOTONE
_ALL = InputRole.REQUIRES_ALL

OPERATORS: dict[str, OperatorSpec] = {
    "select": OperatorSpec("select", (_data(_C, _MONO), _structural(_S)), _C, "filter by predicate", '{"op":"select","predicate":{"path":"grade","op":">","value":90}}'),
    "project": OperatorSpec("project", (_data(_C, _MONO), _structural(_S)), _C, "pick and rename fields", '{"op":"project","fields":{"code":"courseNumber"}}'),
    "extend": OperatorSpec("extend", (_data(_C, _MONO), _structural(_S)), _C, "compute a field per record", '{"op":"extend","fields":{"gap":{"sub":[{"path":"required"},{"path":"earned"}]}}}'),
    "join": OperatorSpec("join", (_data(_C, _MONO), _data(_C, _MONO), _structural(_S)), _C, "relate two collections", '{"op":"join","other":"<fact>","predicate":{"path":"left.courseId","op":"=","value":{"path":"right._id"}}}  (fields become left.x / right.y)'),
    "union": OperatorSpec("union", (_data(_C, _MONO), _data(_C, _MONO)), _C, "all records of both", '{"op":"union","other":"<fact>"}'),
    # The asymmetry that matters: an incomplete SUBTRAHEND wrongly RETAINS every
    # record missing from it, so the result is wrong rather than partial.
    "difference": OperatorSpec("difference", (_data(_C, _MONO), _data(_C, _ALL)), _C, "records of A not in B", '{"op":"difference","other":"<fact>","on":"courseNumber"}'),
    "distinct": OperatorSpec("distinct", (_data(_C, _MONO),), _C, "drop duplicate records, or duplicates by a key with `on` (keeps the first per key)", '{"op":"distinct","on":"courseNumber"}'),
    "unnest": OperatorSpec("unnest", (_data(_C, _MONO), _structural(_S)), _C, "one record per array element", '{"op":"unnest","field":"semesters"}'),
    "group": OperatorSpec("group", (_data(_C, _ALL), _structural(_S)), _C, "partition and aggregate", '{"op":"group","by":["courseNumber"],"agg":{"times":{"agg":"count"}}}'),
    # `only` closes a gap found on a live run: every other aggregate yields a
    # QUANTITY, and ordering identifiers is forbidden, so there was no expression
    # at all for "the id field of this one-record collection" -- a lookup key,
    # the most ordinary value there is.
    "aggregate": OperatorSpec("aggregate", (_data(_C, _ALL), _structural(_S)), _S, "collapse to one value (count/sum/avg/min/max/only)", '{"op":"aggregate","agg":"sum","field":"creditsEarned"}'),
    "sort": OperatorSpec("sort", (_data(_C, _MONO), _structural(_S)), _C, "order records", '{"op":"sort","field":"times","dir":"desc"}'),
    "limit": OperatorSpec("limit", (_data(_C, _MONO), _structural(_S)), _C, "keep the first n", '{"op":"limit","n":1}'),
    "arith": OperatorSpec("arith", (_data(_S), _data(_S), _structural(_S)), _S, "add/subtract/multiply/divide", '{"op":"arith","fn":"sub","other":"<fact>"}'),
    "compare": OperatorSpec("compare", (_data(_S), _data(_S), _structural(_S)), _S, "compare two scalars", '{"op":"compare","fn":">","other":"<fact>"}'),
}

_AGGREGATE_OPS = frozenset({"count", "sum", "avg", "min", "max", "only"})
_QUANTITY_AGGREGATES = frozenset({"sum", "avg", "min", "max"})


class ArithOp(Enum):
    ADD = "+"
    SUBTRACT = "-"
    MULTIPLY = "×"
    DIVIDE = "÷"
    # The larger / smaller of two quantities. MAX exists so a computed grade can
    # be FLOORED -- max(0, needed_min) -- rather than shipping a negative one; its
    # absence is why the negative-min-grade answers had no arithmetic way out.
    MAX = "max"
    MIN = "min"
    # Comparisons, which produce a BOOL rather than a quantity. Their absence
    # was a hole the eligibility question fell into every time: "am I eligible"
    # is `met_groups >= required_groups`, the grammar could add and subtract the
    # two counts but never compare them, and the model spent three turns
    # rediscovering that before answering with slots for facts it never managed
    # to derive. "Do I meet the minimum", "is my GPA above 65" and "have I
    # earned enough credits" are the same shape.
    GTE = ">="
    GT = ">"
    LTE = "<="
    LT = "<"
    EQ = "=="


COMPARISONS = frozenset({ArithOp.GTE, ArithOp.GT, ArithOp.LTE, ArithOp.LT, ArithOp.EQ})
"""The ops above that yield a truth value. Named once so the evaluator, the type
checker and the codec cannot disagree about which ones those are."""


@dataclass(frozen=True)
class PathRef:
    path: Path


@dataclass(frozen=True)
class Literal:
    value: Scalar


@dataclass(frozen=True)
class Held:
    """A held SCALAR fact used inside a per-record expression.

    The missing piece for "each row against a global aggregate": a per-course
    GPA threshold is `(85*(credits + total_credits) - total_points) / credits`,
    which mixes the record's `credits` with two DERIVED scalars. Those cannot be
    literals -- grounding forbids typing a computed number -- and `PathRef` only
    reaches record fields, so before this the whole analytic was inexpressible
    and the model correctly gave up on it. `Held` resolves the named scalar from
    the working set at evaluation, keeping the value grounded.
    """

    name: str


@dataclass(frozen=True)
class Arith:
    op: ArithOp
    left: "ScalarExpr"
    right: "ScalarExpr"


ScalarExpr = Union[PathRef, Literal, Held, Arith]


@dataclass(frozen=True)
class CollectionShape:
    """The field types of a collection's records -- what the checker reasons over."""

    fields: Mapping[str, Union[ScalarKind, "CollectionShape"]]

    def scalar_fields(self) -> dict[str, ScalarKind]:
        return {name: kind for name, kind in self.fields.items() if isinstance(kind, ScalarKind)}

    def quantity_fields(self) -> list[str]:
        return sorted(n for n, k in self.fields.items() if k is ScalarKind.QUANTITY)


@dataclass(frozen=True)
class Stage:
    op: str
    args: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Pipeline:
    """A named derivation: either a `source` collection put through `stages`, or
    a source-less scalar `value`.

    The scalar form exists because a figure computed FROM other figures --
    `gpa = total_points / total_credits`, the average grade a plan must clear --
    has no collection to hang on. The only way to express it before was to pick
    an arbitrary carrier collection, `extend` a constant onto every row, then
    `aggregate only` to collapse it back to one value. Live runs showed the model
    reaching, again and again, for a scalar arithmetic that simply was not there,
    then stalling. `value` is that operation: a scalar expression over held
    scalars (`{"fact": ...}`) and literals, evaluated once, with no source.
    """

    name: str
    source: str = ""
    stages: tuple[Stage, ...] = ()
    value: "ScalarExpr | None" = None


@dataclass(frozen=True)
class ExpressionDefect:
    """The pipeline is wrong and an edit can fix it. Always names what to switch to."""

    stage: int
    message: str


@dataclass(frozen=True)
class DataDefect:
    """The facts lack what any correct pipeline would need.

    No edit fixes this. A repair loop must branch on the TYPE rather than match
    on the message, or it burns its budget re-deriving a pipeline that was
    already right.
    """

    stage: int
    message: str


Defect = Union[ExpressionDefect, DataDefect]
CheckResult = Union[CollectionShape, Ty, ExpressionDefect, DataDefect]


# --------------------------------------------------------------------------
# Sugar (§3.8): declared redundancy with a canonical expansion to the basis.
# Anything that cannot expand is not sugar -- it is new capability, and means
# the basis was incomplete. Expansion is to PIPELINES rather than stages,
# because A ∩ B = A − (A − B) reads the source twice and a linear stage list
# cannot express that.
# --------------------------------------------------------------------------

SUGAR: dict[str, str] = {
    "intersection": "A ∩ B = A − (A − B)",
    "argmax": "argmax(C, p) = limit(sort(C, p, desc), 1)",
}


def expand_sugar(name: str, source: str, args: Mapping[str, Any]) -> tuple[Pipeline, ...]:
    """Expand sugar to basis-only pipelines. The LAST pipeline is the result."""
    if name == "intersection":
        removed = Pipeline(f"_{source}_minus", source, (Stage("difference", {"other": args["other"]}),))
        return (removed, Pipeline("result", source, (Stage("difference", {"other": removed.name}),)))
    if name == "argmax":
        return (
            Pipeline(
                "result",
                source,
                (
                    Stage("sort", {"path": args["path"], "dir": "desc"}),
                    Stage("limit", {"n": 1}),
                ),
            ),
        )
    raise KeyError(f"unknown sugar '{name}'; known: {sorted(SUGAR)}")


# --------------------------------------------------------------------------
# The generic checker
# --------------------------------------------------------------------------


def check_pipeline(pipeline: Pipeline, env: Mapping[str, CollectionShape]) -> CheckResult:
    """Type-check a pipeline against the shapes available to it.

    Returns the resulting shape (or `Ty.SCALAR`), or the first defect found.
    """
    if pipeline.source not in env:
        return ExpressionDefect(0, f"unknown source '{pipeline.source}'; available: {sorted(env)}")

    current: CollectionShape | Ty = env[pipeline.source]

    for index, stage in enumerate(pipeline.stages):
        spec = OPERATORS.get(stage.op)
        if spec is None:
            known = sorted(OPERATORS) + sorted(SUGAR)
            return ExpressionDefect(index, f"unknown operator '{stage.op}'; available: {known}")

        expected = spec.operands[0].ty
        actual = Ty.COLLECTION if isinstance(current, CollectionShape) else Ty.SCALAR
        if expected is not actual:
            return ExpressionDefect(
                index,
                f"'{stage.op}' expects a {expected.value} but stage {index} received a "
                f"{actual.value}. A {actual.value} cannot feed a {expected.value} stage.",
            )

        outcome = _apply(index, stage, current, env)
        if isinstance(outcome, (ExpressionDefect, DataDefect)):
            return outcome
        current = outcome

    return current


def _apply(
    index: int,
    stage: Stage,
    current: CollectionShape | Ty,
    env: Mapping[str, CollectionShape],
) -> CheckResult:
    op = stage.op
    args = stage.args

    if not isinstance(current, CollectionShape):
        return Ty.SCALAR  # scalar-to-scalar ops preserve scalarity

    if op in ("select", "sort"):
        paths = collect_paths(args["predicate"]) if op == "select" else (args["path"],)
        missing = _missing_paths(paths, current)
        if missing:
            return _unknown_field(index, missing, current)
        return current

    if op == "project":
        fields: Mapping[str, Path] = args["fields"]
        missing = _missing_paths(tuple(fields.values()), current)
        if missing:
            return _unknown_field(index, missing, current)
        return CollectionShape(fields={name: _kind_at(path, current) for name, path in fields.items()})

    if op == "extend":
        computed: dict[str, Union[ScalarKind, CollectionShape]] = dict(current.fields)
        for name, expression in args["fields"].items():
            inferred = _infer_scalar(index, expression, current)
            if isinstance(inferred, (ExpressionDefect, DataDefect)):
                return inferred
            computed[name] = inferred
        return CollectionShape(fields=computed)

    if op in ("join",):
        other = env.get(args["other"])
        if other is None:
            return ExpressionDefect(index, f"unknown collection '{args['other']}'; available: {sorted(env)}")
        # Qualification IS rename (Codd's rho): without it a self-join collides.
        qualified: dict[str, Union[ScalarKind, CollectionShape]] = {}
        qualified.update({f"left.{n}": k for n, k in current.fields.items()})
        qualified.update({f"right.{n}": k for n, k in other.fields.items()})
        return CollectionShape(fields=qualified)

    if op in ("union", "difference"):
        other = env.get(args["other"])
        if other is None:
            return ExpressionDefect(index, f"unknown collection '{args['other']}'; available: {sorted(env)}")
        merged = dict(current.fields)
        if op == "union":
            merged.update(other.fields)
        return CollectionShape(fields=merged)

    if op == "limit":
        return current

    if op == "distinct":
        on = args.get("on")
        if on is not None:
            missing = _missing_paths((on,), current)
            if missing:
                return _unknown_field(index, missing, current)
        return current

    if op == "unnest":
        path: Path = args["path"]
        inner = current.fields.get(path.dotted)
        if not isinstance(inner, CollectionShape):
            return ExpressionDefect(
                index,
                f"'{path.dotted}' is not a collection field, so it cannot be unnested. "
                f"Collection fields available: {sorted(n for n, k in current.fields.items() if isinstance(k, CollectionShape))}",
            )
        # Parent fields ride along (SQL lateral semantics), so a later group or
        # join on parent identity is still possible.
        merged = {n: k for n, k in current.fields.items() if n != path.dotted}
        merged.update(inner.fields)
        return CollectionShape(fields=merged)

    if op == "group":
        by: Sequence[Path] = args.get("by", ())
        missing = _missing_paths(tuple(by), current)
        if missing:
            return _unknown_field(index, missing, current)
        grouped: dict[str, Union[ScalarKind, CollectionShape]] = {
            p.dotted: _kind_at(p, current) for p in by
        }
        for name, (_agg, agg_path) in args.get("agg", {}).items():
            grouped[name] = ScalarKind.QUANTITY
            if _missing_paths((agg_path,), current):
                return _unknown_field(index, (agg_path,), current)
        return CollectionShape(fields=grouped)

    if op == "aggregate":
        agg = args.get("op")
        if agg not in _AGGREGATE_OPS:
            return ExpressionDefect(index, f"unknown aggregate '{agg}'; available: {sorted(_AGGREGATE_OPS)}")
        if agg == "count":
            return Ty.SCALAR
        if agg == "only":
            path = args.get("path")
            if path is None:
                return ExpressionDefect(index, "aggregate 'only' needs a path naming the field to read")
            if _missing_paths((path,), current):
                return _unknown_field(index, (path,), current)
            return Ty.SCALAR
        path = args.get("path")
        if path is None:
            return ExpressionDefect(index, f"aggregate '{agg}' needs a path (only 'count' may omit one)")
        if _missing_paths((path,), current):
            return _unknown_field(index, (path,), current)
        if _kind_at(path, current) is not ScalarKind.QUANTITY:
            return _not_a_quantity(index, path.dotted, current)
        return Ty.SCALAR

    return ExpressionDefect(index, f"operator '{op}' has no shape rule")


def _missing_paths(paths: Sequence[Path], shape: CollectionShape) -> tuple[Path, ...]:
    return tuple(p for p in paths if p.dotted not in shape.fields)


def _kind_at(path: Path, shape: CollectionShape) -> Union[ScalarKind, CollectionShape]:
    return shape.fields[path.dotted]


def _unknown_field(index: int, missing: Sequence[Path], shape: CollectionShape) -> ExpressionDefect:
    names = ", ".join(f"'{p.dotted}'" for p in missing)
    return ExpressionDefect(
        index,
        f"{names} is not a field on these records. Available fields: {sorted(shape.fields)}.",
    )


def _not_a_quantity(index: int, name: str, shape: CollectionShape) -> Defect:
    """Naming what to switch to is what separates a repairable error from a retry.

    When there is nothing to switch to, the expression was never the problem --
    that is a DATA defect and a repair loop must stop rather than re-derive.
    """
    numeric = shape.quantity_fields()
    if numeric:
        return ExpressionDefect(
            index,
            f"'{name}' is not a quantity, so it cannot be summed or averaged. "
            f"Quantity fields available: {numeric}.",
        )
    return DataDefect(
        index,
        f"'{name}' is not a quantity and these records carry no quantity field at all "
        f"(fields present: {sorted(shape.fields)}), so there is nothing to aggregate. "
        "No edit to this pipeline can fix that -- the quantity was never retrieved.",
    )


def _infer_scalar(index: int, expression: ScalarExpr, shape: CollectionShape) -> Union[ScalarKind, Defect]:
    if isinstance(expression, Literal):
        return expression.value.kind
    if isinstance(expression, Held):
        # A held scalar used in arithmetic is a QUANTITY; the working set is not
        # in scope at type-check time, so the value's kind is verified at
        # evaluation, where a non-quantity fact fails the arith cleanly.
        return ScalarKind.QUANTITY
    if isinstance(expression, PathRef):
        if _missing_paths((expression.path,), shape):
            return _unknown_field(index, (expression.path,), shape)
        kind = _kind_at(expression.path, shape)
        if isinstance(kind, CollectionShape):
            return ExpressionDefect(index, f"'{expression.path.dotted}' is a collection, not a scalar")
        return kind
    left = _infer_scalar(index, expression.left, shape)
    if isinstance(left, (ExpressionDefect, DataDefect)):
        return left
    right = _infer_scalar(index, expression.right, shape)
    if isinstance(right, (ExpressionDefect, DataDefect)):
        return right
    for operand, kind in ((expression.left, left), (expression.right, right)):
        if kind is not ScalarKind.QUANTITY:
            name = operand.path.dotted if isinstance(operand, PathRef) else "a literal"
            return _not_a_quantity(index, name, shape)
    # A comparison of two quantities is a truth value, not a quantity. Declaring
    # it QUANTITY would let it be summed, and would let the answer boundary see
    # a "number" that is really a yes.
    return ScalarKind.BOOL if expression.op in COMPARISONS else ScalarKind.QUANTITY


__all__ = [
    "OPERATORS",
    "SUGAR",
    "Arith",
    "ArithOp",
    "CheckResult",
    "CollectionShape",
    "DataDefect",
    "Defect",
    "ExpressionDefect",
    "Held",
    "Literal",
    "OperandPosition",
    "OperandSpec",
    "OperatorSpec",
    "PathRef",
    "Pipeline",
    "ScalarExpr",
    "Stage",
    "Ty",
    "check_pipeline",
    "expand_sugar",
]
