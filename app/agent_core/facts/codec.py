"""JSON to pipelines -- phase 9a of docs/agent/tools_implementation_plan.md.

The algebra is only worth having if a model can write it, so this is where the
design's constructibility claim gets tested by a parser rather than argued.

Every error here names both the mistake and the legal alternatives. That is not
politeness: a model handed "invalid pipeline" has nothing to act on and re-emits
the same thing, burning a turn per attempt. The old expression tree learned this
the hard way, and the lesson is cheaper to copy than to relearn.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Union

from app.agent_core.facts.operators import (
    OPERATORS,
    SUGAR,
    Arith,
    ArithOp,
    Held,
    Literal,
    PathRef,
    Pipeline,
    ScalarExpr,
    Stage,
)
from app.agent_core.facts.predicate import (
    Always,
    And,
    Comparison,
    FactRef,
    Not,
    Op,
    Or,
    Path,
    Predicate,
)
from app.agent_core.facts.types import Scalar, ScalarKind


class ParseError(Exception):
    """Malformed input, described well enough that the next attempt can differ."""


_COMPARATORS = {
    "=": Op.EQ, "==": Op.EQ,
    "!=": Op.NE, "≠": Op.NE,
    "<": Op.LT, "<=": Op.LE, "≤": Op.LE,
    ">": Op.GT, ">=": Op.GE, "≥": Op.GE,
    "in": Op.IN, "contains": Op.CONTAINS,
}

_ARITH = {
    "add": ArithOp.ADD, "+": ArithOp.ADD,
    "subtract": ArithOp.SUBTRACT, "sub": ArithOp.SUBTRACT, "minus": ArithOp.SUBTRACT, "-": ArithOp.SUBTRACT,
    "multiply": ArithOp.MULTIPLY, "mul": ArithOp.MULTIPLY, "times": ArithOp.MULTIPLY, "*": ArithOp.MULTIPLY, "×": ArithOp.MULTIPLY,
    "divide": ArithOp.DIVIDE, "div": ArithOp.DIVIDE, "/": ArithOp.DIVIDE, "÷": ArithOp.DIVIDE,
    # Comparisons. Spelled out as well as symbolic, because a model writing JSON
    # reaches for "gte" as readily as ">=".
    "gte": ArithOp.GTE, "ge": ArithOp.GTE, ">=": ArithOp.GTE, "at_least": ArithOp.GTE,
    "gt": ArithOp.GT, ">": ArithOp.GT, "greater_than": ArithOp.GT,
    "lte": ArithOp.LTE, "le": ArithOp.LTE, "<=": ArithOp.LTE, "at_most": ArithOp.LTE,
    "lt": ArithOp.LT, "<": ArithOp.LT, "less_than": ArithOp.LT,
    "eq": ArithOp.EQ, "==": ArithOp.EQ, "equals": ArithOp.EQ,
    "max": ArithOp.MAX, "maximum": ArithOp.MAX,
    "min": ArithOp.MIN, "minimum": ArithOp.MIN,
    # ceil(a / b). The unary spellings take ONE operand and are folded to
    # ceil(x / 1) below, because a model asked to "round up" reaches for
    # {"ceil": [expr]} far more readily than for a division that also rounds --
    # and a live run did exactly that, four times, against a grammar that had
    # neither.
    "ceil_div": ArithOp.CEIL_DIV, "divide_round_up": ArithOp.CEIL_DIV,
    "ceil": ArithOp.CEIL_DIV, "ceiling": ArithOp.CEIL_DIV, "round_up": ArithOp.CEIL_DIV,
}

_UNARY_FOLDS_TO_ONE = frozenset({"ceil", "ceiling", "round_up"})
"""Spellings that take a single operand, completed as `op(x, 1)`.

`ceil(x)` is exactly `ceil(x / 1)`, so the unary form needs no unary node and
the expression tree stays binary throughout."""

_KINDS = {kind.value: kind for kind in ScalarKind}

# Args that name a field rather than carry data, so they become Paths.
_PATH_ARGS = frozenset({"path", "on", "by"})

# Which function a stage applies. Several spellings, because a model reaches for
# whichever reads naturally and being right about the CONCEPT is what matters.
_FUNCTION_KEYS = frozenset({"agg", "fn", "operator", "function", "aggregate"})

# Everything else a stage may legitimately carry.
_STRUCTURAL_ARGS = frozenset({"dir", "n", "limit", "agg_spec", "carry", "max_depth", "value", "kind"})


def parse_pipelines(payload: Any) -> tuple[Pipeline, ...]:
    """Parse a list of pipeline objects."""
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes, Mapping)):
        raise ParseError(
            f"expected a LIST of pipelines, got {type(payload).__name__}. Even a single pipeline "
            "is written as a list of one, because a call may derive several facts at once."
        )
    return tuple(_pipeline(item, index) for index, item in enumerate(payload))


def _pipeline(payload: Any, index: int) -> Pipeline:
    if not isinstance(payload, Mapping):
        raise ParseError(f"pipeline {index} is a {type(payload).__name__}, expected an object")

    name = payload.get("name")
    if not isinstance(name, str) or not name:
        raise ParseError(
            f"pipeline {index} has no 'name'. Every pipeline needs one so other pipelines can "
            "reference its result and so failures can be reported against it individually."
        )

    # A scalar pipeline: one value computed from held scalars, no source. It is
    # `{"name": ..., "value": <expr>}` -- the same expression grammar as an
    # `extend` field, minus `{"path": ...}` (there are no records to read).
    if "value" in payload and "source" not in payload:
        return Pipeline(name=name, value=_scalar_expression(payload["value"], name, 0))

    source = payload.get("source")
    if not isinstance(source, str) or not source:
        raise ParseError(
            f"pipeline '{name}' has no 'source': name the fact or pipeline it reads from. "
            "(To compute one value FROM held scalars, drop 'source' and give a 'value' expression "
            'instead: {"name": "gpa", "value": {"div": [{"fact": "points"}, {"fact": "credits"}]}}.)'
        )

    stages = payload.get("stages", [])
    if not isinstance(stages, Sequence) or isinstance(stages, (str, bytes)):
        raise ParseError(f"pipeline '{name}' has a non-list 'stages'")

    return Pipeline(name=name, source=source, stages=tuple(_stage(s, name, i) for i, s in enumerate(stages)))


def _stage(payload: Any, pipeline: str, index: int) -> Stage:
    if not isinstance(payload, Mapping):
        raise ParseError(f"{pipeline} stage {index} is a {type(payload).__name__}, expected an object")

    op = payload.get("op")
    if op in SUGAR:
        raise ParseError(
            f"'{op}' is sugar, not a basis operator, so it cannot appear as a stage. "
            f"It expands to: {SUGAR[op]}. Write that expansion, or ask for the sugar to be "
            "expanded before execution."
        )
    if op not in OPERATORS:
        # An arithmetic or comparison FUNCTION written where a stage OPERATOR
        # belongs is the single most repeated mistake across live runs. Listing
        # the legal operators shows the answer without spelling it, so the
        # correction is written out.
        if op in _ARITH:
            raise ParseError(
                f"{pipeline} stage {index}: '{op}' is an arithmetic function, not a stage operator. "
                f'Write {{"op": "arith", "fn": "{op}", "other": "<other fact>"}} -- `arith` is the '
                "operator and `fn` names which arithmetic it does."
            )
        if op in _COMPARATORS:
            raise ParseError(
                f"{pipeline} stage {index}: '{op}' is a comparator, not a stage operator. "
                f'Write {{"op": "compare", "fn": "{op}", "other": "<other fact>"}}.'
            )
        raise ParseError(
            f"{pipeline} stage {index}: unknown operator '{op}'. Available: {sorted(OPERATORS)}."
        )

    args: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "op":
            continue
        if key == "predicate":
            args[key] = parse_predicate(value)
        elif key == "field" and op in ("aggregate", "sort", "unnest"):
            args["path"] = _path(value)
        elif op == "group" and key in _GROUP_AGGREGATE_KEYS:
            # `agg` means something DIFFERENT here, and the collision was silent.
            # Everywhere else `agg` names the stage's function; on `group` it
            # names a whole map of output columns. Routing it through
            # `_stage_function` stored it under `args["op"]`, `_group` read
            # `args["agg"]`, found nothing, and returned the group KEYS with no
            # aggregated column at all -- a grouped count that silently produced
            # no counts, which every layer downstream then treated as real.
            args["agg"] = _group_aggregates(value, pipeline, index)
        elif key in _FUNCTION_KEYS:
            # On the wire `op` ALWAYS means the stage operator. The function a
            # stage applies -- sum, divide, ">" -- arrives under `agg`/`fn`,
            # because one key with two meanings is how a parser starts guessing,
            # and the guess would be silent.
            args["op"] = _stage_function(op, value, pipeline, index)
        elif key == "fields":
            args[key] = _fields(value, pipeline, index)
        elif key == "other":
            # `{"fact": "x"}` and `"x"` mean the same thing. The model learned
            # the braced form for predicate values and generalised it here,
            # which is the right instinct -- both name a held fact. Accepting
            # one spelling and crashing on the other would be teaching an
            # arbitrary distinction.
            if isinstance(value, Mapping) and "fact" in value:
                args["other"] = str(value["fact"])
            elif isinstance(value, str):
                args["other"] = value
            else:
                raise ParseError(
                    f"{pipeline} stage {index}: 'other' names a fact, so it must be a string "
                    f'(or {{"fact": "name"}}). Got {type(value).__name__}.'
                )
        elif key in _PATH_ARGS:
            args[key] = tuple(_path(v) for v in value) if isinstance(value, list) else _path(value)
        elif key in _STRUCTURAL_ARGS:
            args[key] = value
        else:
            # Silently keeping an unrecognised key is how a naming guess becomes
            # a confusing error two steps later: `{"operator": "subtract"}` was
            # dropped here and resurfaced as "unknown arithmetic operator None",
            # which named neither the real key nor the real problem. Across live
            # runs the same function was spelled `fn`, `op2`, `kind` and
            # `operator` -- four turns spent on one silent drop.
            raise ParseError(
                f"{pipeline} stage {index}: '{op}' has no argument {key!r}. "
                f"Accepted here: {sorted(_STRUCTURAL_ARGS | _PATH_ARGS | _FUNCTION_KEYS | {'predicate', 'fields', 'other'})}."
            )

    return Stage(op=op, args=args)


_GROUP_AGGREGATE_KEYS = frozenset({"agg", "agg_spec", "aggregate", "aggregates"})
_AGGREGATE_FUNCTIONS = frozenset({"count", "sum", "avg", "min", "max", "only"})


def _group_aggregates(value: Any, pipeline: str, index: int) -> dict[str, tuple[str, Any]]:
    """`{"total": {"agg": "sum", "field": "credits"}}` -> `{"total": ("sum", Path)}`.

    Accepts the pair form `{"total": ["sum", "credits"]}` and the bare
    `{"n": "count"}` too, for the same reason `_FUNCTION_KEYS` accepts several
    spellings: being right about the CONCEPT is what matters, and a model that
    picked a different legal-looking shape should not lose a turn to it.
    """
    if not isinstance(value, Mapping):
        raise ParseError(
            f"{pipeline} stage {index}: 'group' needs an object of output name -> aggregate, "
            'e.g. {"agg": {"how_many": {"agg": "count"}}}. '
            f"Got {type(value).__name__}."
        )

    parsed: dict[str, tuple[str, Any]] = {}
    for name, spec in value.items():
        if isinstance(spec, str):
            function, path = spec, None
        elif isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)) and len(spec) == 2:
            function, path = spec[0], spec[1]
        elif isinstance(spec, Mapping):
            function = next((spec[k] for k in _FUNCTION_KEYS if k in spec), None)
            path = next((spec[k] for k in ("path", "field") if k in spec), None)
        else:
            raise ParseError(
                f"{pipeline} stage {index}: aggregate {name!r} must be a function name, "
                'a [function, field] pair, or {"agg": ..., "field": ...}.'
            )

        if function not in _AGGREGATE_FUNCTIONS:
            raise ParseError(
                f"{pipeline} stage {index}: aggregate {name!r} names {function!r}; "
                f"available: {sorted(_AGGREGATE_FUNCTIONS)}."
            )
        if path is None and function != "count":
            raise ParseError(
                f"{pipeline} stage {index}: aggregate {name!r} uses '{function}', which needs a "
                "field to read. Only 'count' may omit one."
            )
        parsed[name] = (function, _path(path) if path is not None else None)
    return parsed


def _stage_function(stage_op: str, value: Any, pipeline: str, index: int) -> Any:
    """Resolve a stage's function to the type the runner expects.

    `aggregate` keeps a plain string; `arith` and `compare` need real enums. The
    runner will not coerce on its behalf -- a stage carrying a raw string where
    an enum belongs fails at evaluation time, which is far too late to tell the
    model anything useful.
    """
    if stage_op == "arith":
        if value not in _ARITH:
            raise ParseError(
                f"{pipeline} stage {index}: unknown arithmetic function {value!r}; "
                f"available: {sorted(set(_ARITH) - set('+-*/×÷'))}."
            )
        return _ARITH[value]

    if stage_op == "compare":
        if value not in _COMPARATORS:
            raise ParseError(
                f"{pipeline} stage {index}: unknown comparator {value!r}; "
                f"available: {sorted(set(_COMPARATORS))}."
            )
        return _COMPARATORS[value]

    return value


def _fields(value: Any, pipeline: str, index: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ParseError(f"{pipeline} stage {index}: 'fields' must be an object of name -> path/expression")
    parsed: dict[str, Any] = {}
    for name, spec in value.items():
        parsed[name] = _path(spec) if isinstance(spec, str) else _scalar_expression(spec, pipeline, index)
    return parsed


def _scalar_expression(payload: Any, pipeline: str, index: int) -> ScalarExpr:
    if not isinstance(payload, Mapping):
        raise ParseError(f"{pipeline} stage {index}: expected a path string or an expression object")

    if "path" in payload:
        return PathRef(_path(payload["path"]))
    if "value" in payload:
        return Literal(_scalar(payload["value"], payload.get("kind")))
    if "fact" in payload:
        # A held scalar reference inside a per-record expression -- the same
        # {"fact": name} idiom used for predicate values, so a computed total can
        # feed a per-row formula (the per-course GPA threshold).
        return Held(str(payload["fact"]))

    for key, operator in _ARITH.items():
        if key in payload:
            operands = payload[key]
            if not isinstance(operands, Sequence) or isinstance(operands, (str, bytes)):
                raise ParseError(
                    f"{pipeline} stage {index}: '{key}' needs a list of operands, got a non-list."
                )
            # `{"ceil": [x]}` is the natural way to write a ceiling and is
            # completed to ceil(x / 1), which is the same number. Without this
            # the model's first, correct instinct parses as an arity error and
            # it goes looking for an operator that does not exist.
            if len(operands) == 1 and key in _UNARY_FOLDS_TO_ONE:
                return Arith(
                    op=operator,
                    left=_scalar_expression(operands[0], pipeline, index),
                    right=Literal(_scalar(1, None)),
                )
            if len(operands) != 2:
                raise ParseError(
                    f"{pipeline} stage {index}: '{key}' needs exactly two operands, got "
                    f"{len(operands)}."
                )
            return Arith(
                op=operator,
                left=_scalar_expression(operands[0], pipeline, index),
                right=_scalar_expression(operands[1], pipeline, index),
            )

    raise ParseError(
        f"{pipeline} stage {index}: expression must be {{'path': ...}}, {{'value': ...}}, or one of "
        f"{sorted(set(_ARITH) - set('+-*/×÷'))}."
    )


def parse_predicate(payload: Any) -> Predicate:
    """Parse a predicate object."""
    if not isinstance(payload, Mapping):
        raise ParseError(f"a predicate must be an object, got {type(payload).__name__}")

    if payload.get("always") is True:
        return Always()
    if "and" in payload:
        return And(tuple(parse_predicate(term) for term in payload["and"]))
    if "or" in payload:
        return Or(tuple(parse_predicate(term) for term in payload["or"]))
    if "not" in payload:
        return Not(parse_predicate(payload["not"]))

    if "path" not in payload:
        raise ParseError(
            "a predicate needs 'path', or one of 'and' / 'or' / 'not' / 'always'. "
            f"Got keys: {sorted(payload)}."
        )

    raw_op = payload.get("op")
    if raw_op not in _COMPARATORS:
        raise ParseError(f"unknown comparator {raw_op!r}; available: {sorted(set(_COMPARATORS))}.")
    op = _COMPARATORS[raw_op]

    value = payload.get("value")
    kind = payload.get("kind")

    if isinstance(value, Mapping) and "fact" in value:
        # Filtering by something the caller HOLDS rather than types. With
        # `field`, it is the SET of that field's values across a collection
        # fact -- a semi-join -- and is the natural spelling of `in`; without,
        # it is a single held value.
        field = value.get("field")
        ref = FactRef(str(value["fact"]), str(field) if field is not None else None)
        if op is Op.IN and field is None:
            raise ParseError(
                "'in' against a held fact needs the FIELD to draw values from: "
                '{"fact": "completed", "field": "courseId"}. A bare {"fact": name} is a single '
                "value, and 'in' compares against a set."
            )
        # A `{"fact": name, "field": f}` with a scalar op is a one-record field
        # extraction (resolved at dispatch); with `in` it is a set. Both are
        # valid, so the codec accepts the shape and lets dispatch read the fact.
        return Comparison(_path(payload["path"]), op, ref)

    if op is Op.IN:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ParseError(
                "'in' needs a list of values, or a collection you hold: "
                '{"fact": "my_courses", "field": "courseId"}.'
            )
        return Comparison(_path(payload["path"]), op, tuple(_scalar(v, kind) for v in value))

    if isinstance(value, Mapping) and "path" in value:
        # Comparing one field to another rather than to a literal.
        return Comparison(_path(payload["path"]), op, _path(value["path"]))

    return Comparison(_path(payload["path"]), op, _scalar(value, kind))


def _scalar(value: Any, kind: Any = None) -> Scalar:
    """A JSON value as a typed Scalar.

    Inference rule, applied everywhere: a JSON number is a QUANTITY, a JSON
    string is an IDENTIFIER, a JSON bool is a BOOL. That covers the two cases
    that actually recur -- `grade > 90` and `id == "00940224"` -- and anything
    else must say `kind` rather than be guessed at, because the guess that would
    be needed is precisely the leading-zero heuristic the type system removed.
    """
    if kind is not None:
        if kind not in _KINDS:
            raise ParseError(f"unknown kind {kind!r}; available: {sorted(_KINDS)}")
        return Scalar(_KINDS[kind], value)

    if isinstance(value, bool):
        return Scalar(ScalarKind.BOOL, value)
    if isinstance(value, (int, float)):
        return Scalar(ScalarKind.QUANTITY, value)
    if isinstance(value, str):
        return Scalar(ScalarKind.IDENTIFIER, value)
    raise ParseError(
        f"cannot type {value!r} ({type(value).__name__}); give an explicit 'kind' of {sorted(_KINDS)}."
    )


def _path(value: Any) -> Path:
    if not isinstance(value, str):
        raise ParseError(f"a field path must be a string, got {type(value).__name__}")
    return Path.parse(value)


__all__ = ["ParseError", "parse_pipelines", "parse_predicate"]
