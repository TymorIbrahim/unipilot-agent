"""The pipeline runner -- phase 4 of docs/agent/tools_implementation_plan.md.

One call evaluates MANY named pipelines. That is the turn-cost fix: if a call
yields one fact, an N-step derivation costs N turns and each turn carries its
own rejection risk, which is the wandering this redesign exists to remove.

Two properties the runner owes its caller:

  - **order comes from the references**, not the caller's listing, so a model
    that declares pipelines backwards still succeeds
  - **failure is per pipeline**. Discarding four good results because a fifth
    failed forces a repair loop to redo work that was already correct, and a
    dependent of a failure is BLOCKED rather than failed -- a different repair.
"""

from __future__ import annotations

from types import MappingProxyType

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Union

from app.agent_core.facts.operators import (
    COMPARISONS,
    OPERATORS,
    Arith,
    ArithOp,
    DataDefect,
    Defect,
    ExpressionDefect,
    Held,
    Literal,
    OperandPosition,
    PathRef,
    Pipeline,
    ScalarExpr,
    Stage,
    Ty,
)
from app.agent_core.facts.predicate import MISSING as _MISSING
from app.agent_core.facts.predicate import Op, Path, matches
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


@dataclass(frozen=True)
class Succeeded:
    name: str
    value: Union[Collection, Scalar]
    basis: Basis


@dataclass(frozen=True)
class Failed:
    name: str
    defect: Defect


@dataclass(frozen=True)
class Blocked:
    """A dependency failed, so this never ran. Not a defect of its own."""

    name: str
    waiting_on: str


Outcome = Union[Succeeded, Failed, Blocked]

_BINARY_OPS = frozenset({"join", "union", "difference"})



def run_pipelines(
    pipelines: Sequence[Pipeline],
    env: Mapping[str, Collection],
    resolve_predicates: "Callable[[Pipeline, Mapping[str, Any]], Any] | None" = None,
) -> dict[str, Outcome]:
    """Evaluate every pipeline, in dependency order, isolating failures.

    `resolve_predicates` is called for each pipeline just before it runs, with
    everything available AT THAT POINT -- the working set plus every sibling
    that has already succeeded. It exists because a `{"fact": ...}` inside a
    `select` used to be resolved for the whole call up front, against the
    working set alone, so a pipeline could use a sibling as its `source` but not
    as a filter VALUE. The catalog tells the model to prefer one call carrying
    several pipelines that reference each other, and half of that was a lie:

        {"name": "mine",  "source": "cat",   "stages": [{"op": "project", ...}]}
        {"name": "met",   "source": "edges", "stages": [{"op": "select",
            "predicate": {..., "value": {"fact": "mine", "field": "..."}}}]}

    came back "the filter refers to fact 'mine', which is not held" with `mine`
    defined one line above. Measured cost: an eligibility run spent eight
    consecutive turns rewriting that shape.

    Returning a defect from the callback fails only that pipeline, so the others
    in the call still land.
    """
    by_name = {p.name: p for p in pipelines}
    order, cycle = _topological_order(pipelines, env)

    results: dict[str, Outcome] = {}
    if cycle:
        for name in cycle:
            results[name] = Failed(name, ExpressionDefect(0, f"reference cycle among pipelines: {' -> '.join(cycle)}"))

    available: dict[str, Union[Collection, Scalar]] = dict(env)
    bases: dict[str, Basis] = {}

    for name in order:
        if name in results:
            continue
        pipeline = by_name[name]

        blocker = _first_unavailable(pipeline, available, results)
        if blocker is not None:
            results[name] = Blocked(name, waiting_on=blocker)
            continue

        if resolve_predicates is not None:
            defect = resolve_predicates(pipeline, available)
            if defect is not None:
                results[name] = Failed(name, defect)
                continue

        outcome = _run_one(pipeline, available, bases)
        results[name] = outcome
        if isinstance(outcome, Succeeded):
            # SCALAR results are published too, not just collections. Without
            # this a scalar is unreferenceable, and "is spring heavier than
            # autumn" -- two aggregates and a comparison -- cannot be expressed
            # at all, whatever the operator table says.
            available[name] = outcome.value
            bases[name] = outcome.basis

    return results


def _referenced(pipeline: Pipeline) -> tuple[str, ...]:
    # A scalar pipeline has no source; its dependencies are the held scalars its
    # value expression names, so ordering and availability turn on those alone.
    if pipeline.value is not None:
        return tuple(_held_names(pipeline.value))
    names = [pipeline.source]
    for stage in pipeline.stages:
        other = stage.args.get("other")
        if isinstance(other, str):
            names.append(other)
        # A `{"fact": ...}` inside a filter is a dependency exactly like a
        # `source` is. Without collecting it the sibling it names may not have
        # run yet, and the filter fails for what is really an ordering bug.
        names.extend(_fact_ref_names(stage.args.get("predicate")))
        # `Held` scalars inside an `extend` expression are dependencies too --
        # without collecting them the runner may evaluate a per-record formula
        # before the total it references exists, and the formula fails for a
        # reason that is really an ordering bug.
        for expression in (stage.args.get("fields") or {}).values():
            names.extend(_held_names(expression))
    return tuple(names)


def _fact_ref_names(predicate: Any) -> list[str]:
    """Every fact a predicate filters BY, however deeply nested in and/or/not."""
    if predicate is None:
        return []
    terms = getattr(predicate, "terms", None)
    if terms is not None:
        return [name for term in terms for name in _fact_ref_names(term)]
    term = getattr(predicate, "term", None)
    if term is not None:
        return _fact_ref_names(term)
    value = getattr(predicate, "value", None)
    name = getattr(value, "name", None)
    return [name] if isinstance(name, str) else []


def _held_names(expression: Any) -> list[str]:
    if isinstance(expression, Held):
        return [expression.name]
    if isinstance(expression, Arith):
        return _held_names(expression.left) + _held_names(expression.right)
    return []


def _topological_order(
    pipelines: Sequence[Pipeline],
    env: Mapping[str, Collection],
) -> tuple[list[str], list[str]]:
    """Dependency order, plus any names caught in a cycle.

    A cycle is reported rather than hung on -- an infinite wait looks like a
    timeout, which sends a caller looking in entirely the wrong place.
    """
    names = {p.name for p in pipelines}
    pending = {p.name: {ref for ref in _referenced(p) if ref in names and ref != p.name} for p in pipelines}

    ordered: list[str] = []
    while True:
        ready = sorted(name for name, deps in pending.items() if not deps)
        if not ready:
            break
        for name in ready:
            ordered.append(name)
            del pending[name]
        for deps in pending.values():
            deps.difference_update(ready)

    return ordered, sorted(pending)


def _first_unavailable(
    pipeline: Pipeline,
    available: Mapping[str, Union[Collection, Scalar]],
    results: Mapping[str, Outcome],
) -> str | None:
    for ref in _referenced(pipeline):
        if ref not in available:
            if ref in results and not isinstance(results[ref], Succeeded):
                return ref
            if ref in results:
                return ref
    return None


_EMPTY_RECORD = Record(fields={}, basis=Basis.OFFICIAL_RECORD)
"""The record a scalar pipeline evaluates against: it has no fields, so a stray
`{"path": ...}` fails cleanly (there is nothing to read), and its OFFICIAL_RECORD
basis is the strongest, so a `{"value": N}` literal never weakens a result on its
own -- only the held facts an expression pulls in decide its certainty."""


def _run_scalar(
    pipeline: Pipeline,
    available: Mapping[str, Union[Collection, Scalar]],
    bases: Mapping[str, Basis],
) -> Outcome:
    """Evaluate a source-less scalar pipeline against the held scalars.

    The result's basis is the weakest fact the expression consumed -- computed by
    the same `_eval_scalar` a per-record `extend` uses, so a scalar derived here
    is provenance-tracked exactly like one derived a row at a time."""
    held = {name: value for name, value in available.items() if isinstance(value, Scalar)}
    result = _eval_scalar(0, pipeline.value, _EMPTY_RECORD, scalars=held, bases=bases)
    if isinstance(result, (ExpressionDefect, DataDefect)):
        return Failed(pipeline.name, result)
    value, basis = result
    return Succeeded(pipeline.name, value, basis)


def _run_one(
    pipeline: Pipeline,
    available: Mapping[str, Union[Collection, Scalar]],
    bases: Mapping[str, Basis],
) -> Outcome:
    if pipeline.value is not None:
        return _run_scalar(pipeline, available, bases)

    source = available.get(pipeline.source)
    if source is None:
        return Failed(
            pipeline.name,
            ExpressionDefect(0, _no_such_fact("source", pipeline.source, available)),
        )

    current: Union[Collection, Scalar] = source
    basis = bases.get(pipeline.source) or (
        _collection_basis(source) if isinstance(source, Collection) else Basis.OFFICIAL_RECORD
    )

    for index, stage in enumerate(pipeline.stages):
        spec = OPERATORS.get(stage.op)
        if spec is None:
            return Failed(pipeline.name, ExpressionDefect(index, f"unknown operator '{stage.op}'"))

        outcome = _apply(index, stage, current, available, basis, bases)
        if isinstance(outcome, (ExpressionDefect, DataDefect)):
            return Failed(pipeline.name, outcome)
        current, basis = outcome

    return Succeeded(pipeline.name, current, basis)


def _collection_basis(collection: Collection) -> Basis:
    if not collection.records:
        return Basis.OFFICIAL_RECORD
    return weakest([record.basis for record in collection.records])


def _completeness(
    index: int,
    stage: Stage,
    inputs: Sequence[Completeness],
) -> Union[Completeness, DataDefect]:
    spec = OPERATORS[stage.op]
    roles = tuple(
        operand.role or InputRole.MONOTONE
        for operand in spec.operands
        if operand.position is OperandPosition.DATA and operand.ty is Ty.COLLECTION
    )
    verdict = completeness_after(roles[: len(inputs)], inputs)
    if isinstance(verdict, Refusal):
        return DataDefect(index, verdict.reason)
    return verdict


def _apply(
    index: int,
    stage: Stage,
    current: Union[Collection, Scalar],
    available: Mapping[str, Union[Collection, Scalar]],
    basis: Basis,
    bases: Mapping[str, Basis],
) -> Union[tuple[Union[Collection, Scalar], Basis], Defect]:
    op = stage.op
    args = stage.args

    if isinstance(current, Scalar):
        return _apply_scalar(index, stage, current, available, basis, bases)

    binary = args.get("other")
    other: Collection | None = None
    if op in _BINARY_OPS and not isinstance(binary, str):
        # A bare `assert` here crashed a whole live run when the model omitted
        # `other`. A malformed call is the model's mistake to repair, not an
        # invariant breach -- it has to come back as something it can act on.
        return ExpressionDefect(
            index,
            f"'{op}' combines two collections and needs 'other' naming the second one. "
            f"Got {binary!r}. Available: {sorted(available)}.",
        )
    if isinstance(binary, str):
        resolved = available.get(binary)
        if resolved is None:
            return ExpressionDefect(index, _no_such_fact("collection", binary, available))
        if not isinstance(resolved, Collection):
            return ExpressionDefect(
                index, f"'{binary}' is a scalar, but '{op}' needs a collection on both sides"
            )
        other = resolved

    inputs = (current.completeness,) if other is None else (current.completeness, other.completeness)
    completeness = _completeness(index, stage, inputs)
    if isinstance(completeness, DataDefect):
        return completeness

    if op == "select":
        kept = tuple(r for r in current.records if matches(args["predicate"], r))
        return Collection(kept, completeness), _basis_of(kept, basis)

    if op == "project":
        rebuilt = []
        for record in current.records:
            fields, field_basis = {}, {}
            for out_name, path in args["fields"].items():
                value = _resolve(path, record)
                if value is _MISSING:
                    return ExpressionDefect(index, f"'{path.dotted}' missing on a record; cannot project it")
                fields[out_name] = value
                field_basis[out_name] = record.basis_for(path.dotted)
            rebuilt.append(Record(fields=fields, basis=record.basis, field_basis=field_basis))
        return Collection(tuple(rebuilt), completeness), _basis_of(rebuilt, basis)

    if op == "extend":
        # Held scalars from OTHER pipelines are in scope, so a per-record field
        # can combine the record with a global aggregate -- the per-course GPA
        # threshold being the case that forced it.
        scalar_env = {name: value for name, value in available.items() if isinstance(value, Scalar)}
        rebuilt = []
        for record in current.records:
            fields = dict(record.fields)
            field_basis = dict(record.field_basis)
            for out_name, expression in args["fields"].items():
                computed = _eval_scalar(index, expression, record, scalar_env, bases)
                if isinstance(computed, (ExpressionDefect, DataDefect)):
                    return computed
                value, consumed = computed
                fields[out_name] = value
                field_basis[out_name] = consumed
            rebuilt.append(Record(fields=fields, basis=record.basis, field_basis=field_basis))
        return Collection(tuple(rebuilt), completeness), _basis_of(rebuilt, basis)

    if op == "join":
        merged = []
        for left in current.records:
            for right in other.records:
                fields = {f"left.{n}": v for n, v in left.fields.items()}
                fields.update({f"right.{n}": v for n, v in right.fields.items()})
                # Per-field provenance survives the join. Flattening here would
                # degrade every field to the weakest side and never recover.
                field_basis = {f"left.{n}": left.basis_for(n) for n in left.fields}
                field_basis.update({f"right.{n}": right.basis_for(n) for n in right.fields})
                candidate = Record(fields=fields, basis=weakest([left.basis, right.basis]), field_basis=field_basis)
                if matches(args["predicate"], candidate):
                    merged.append(candidate)
        return Collection(tuple(merged), completeness), _basis_of(merged, basis)

    if op == "union":
        combined = current.records + other.records
        return Collection(combined, completeness), _basis_of(combined, basis)

    if op == "difference":
        key: Path | None = args.get("on")
        if key is None:
            removed = {_signature(r) for r in other.records}
            kept = tuple(r for r in current.records if _signature(r) not in removed)
        else:
            missing_key = _first_missing_key(key, current.records) or _first_missing_key(key, other.records)
            if missing_key is not None:
                return DataDefect(
                    index,
                    f"a record has no value for the difference key '{key.dotted}'. Dropping it "
                    "silently would wrongly RETAIN records in the result, so this fails closed. "
                    "The unresolvable records must be fixed upstream.",
                )
            removed = {_raw(_resolve(key, r)) for r in other.records}
            kept = tuple(r for r in current.records if _raw(_resolve(key, r)) not in removed)
        return Collection(kept, completeness), _basis_of(kept, basis)

    if op == "distinct":
        # Unkeyed: whole-record identity. Keyed (`on`): first record per key value,
        # which is the only way to collapse two rows for the SAME course that carry
        # a differing field -- e.g. one tagged type "required" and one "elective"
        # after a union, which a whole-record distinct keeps as two and the answer
        # boundary then refuses as a course listed twice.
        on: Path | None = args.get("on")
        if on is not None:
            missing = _first_missing_key(on, current.records)
            if missing is not None:
                return ExpressionDefect(index, f"'{on.dotted}' missing on a record; cannot dedupe on it")
        seen, kept = set(), []
        for record in current.records:
            signature = _raw(_resolve(on, record)) if on is not None else _signature(record)
            if signature not in seen:
                seen.add(signature)
                kept.append(record)
        return Collection(tuple(kept), completeness), _basis_of(kept, basis)

    if op == "unnest":
        return _unnest(index, args["path"], current, completeness, basis)

    if op == "sort":
        path: Path = args["path"]
        descending = args.get("dir", "asc") == "desc"
        missing = _first_missing_key(path, current.records)
        if missing is not None:
            return ExpressionDefect(index, f"'{path.dotted}' missing on a record; cannot sort on it")
        # Python's sort is stable, so ties keep input order in BOTH directions.
        # That is what makes `limit(sort(...), 1)` reproducible rather than a
        # coin flip between runs.
        ordered = sorted(current.records, key=lambda r: _raw(_resolve(path, r)), reverse=descending)
        return Collection(tuple(ordered), completeness), basis

    if op == "limit":
        return Collection(current.records[: int(args["n"])], completeness), basis

    if op == "aggregate":
        return _aggregate(index, args, current, completeness)

    if op == "group":
        return _group(index, args, current, completeness, basis)

    return ExpressionDefect(index, f"operator '{op}' has no evaluation rule")


def _apply_scalar(
    index: int,
    stage: Stage,
    current: Scalar,
    available: Mapping[str, Union[Collection, Scalar]],
    basis: Basis,
    bases: Mapping[str, Basis],
) -> Union[tuple[Scalar, Basis], Defect]:
    """Combine this pipeline's scalar with another pipeline's, or a literal.

    The operand-position rule (§3.2) is enforced here rather than described:
    `arith` refuses a literal because a number typed straight into an
    arithmetic operand is a laundered computed value -- the `155 - 62.5` bug --
    while `compare` accepts one, because a threshold really can come from the
    question, and the answer boundary catches an ungrounded number that reaches
    the answer anyway.
    """
    op = stage.op
    if op not in ("arith", "compare"):
        return ExpressionDefect(
            index,
            f"'{op}' expects a collection but received a scalar. A scalar cannot feed a "
            "collection stage.",
        )

    reference = stage.args.get("other")
    literal = stage.args.get("value")

    if reference is not None:
        resolved = available.get(reference)
        if resolved is None:
            return ExpressionDefect(index, f"unknown result '{reference}'; available: {sorted(available)}")
        if not isinstance(resolved, Scalar):
            return ExpressionDefect(
                index, f"'{reference}' is a collection; '{op}' combines two scalars. Aggregate it first."
            )
        right, right_basis = resolved, bases.get(reference, basis)
    elif literal is not None:
        if op == "arith":
            return ExpressionDefect(
                index,
                "arith needs a ref on both sides, not a literal. A number typed directly into an "
                "arithmetic operand is an ungrounded value wearing the shape of a result -- name the "
                "pipeline that produced it instead.",
            )
        if not isinstance(literal, Scalar):
            return ExpressionDefect(index, "'value' must be a Scalar")
        right, right_basis = literal, basis
    else:
        return ExpressionDefect(index, f"'{op}' needs either 'other' (a result name) or 'value' (a literal)")

    combined = weakest([basis, right_basis])

    if op == "compare":
        comparator = stage.args.get("op")
        outcome = _compare_scalars(current, comparator, right)
        if isinstance(outcome, ExpressionDefect):
            return outcome
        return Scalar(ScalarKind.BOOL, outcome), combined

    if not current.is_quantity or not right.is_quantity:
        return ExpressionDefect(index, "arith needs quantities on both sides")
    arith_op = stage.args.get("op")
    if not isinstance(arith_op, ArithOp):
        return ExpressionDefect(index, f"unknown arithmetic operator {arith_op!r}")
    if arith_op is ArithOp.DIVIDE and right.value == 0:
        return DataDefect(index, "division by zero: the divisor pipeline evaluated to 0")

    value = {
        ArithOp.ADD: current.value + right.value,
        ArithOp.SUBTRACT: current.value - right.value,
        ArithOp.MULTIPLY: current.value * right.value,
        ArithOp.DIVIDE: current.value / right.value if right.value else 0,
        ArithOp.MAX: max(current.value, right.value),
        ArithOp.MIN: min(current.value, right.value),
        ArithOp.GTE: current.value >= right.value,
        ArithOp.GT: current.value > right.value,
        ArithOp.LTE: current.value <= right.value,
        ArithOp.LT: current.value < right.value,
        ArithOp.EQ: current.value == right.value,
    }[arith_op]
    # Same rule as the expression path: a comparison yields a truth value, and
    # calling it a quantity would let it be summed and would show the answer
    # boundary a "number" that is really a yes.
    kind = ScalarKind.BOOL if arith_op in COMPARISONS else ScalarKind.QUANTITY
    return Scalar(kind, value), combined


def _compare_scalars(left: Scalar, comparator: Any, right: Scalar) -> Union[bool, ExpressionDefect]:
    comparisons = {
        Op.EQ: lambda: left.value == right.value,
        Op.NE: lambda: left.value != right.value,
        Op.LT: lambda: left.value < right.value,
        Op.LE: lambda: left.value <= right.value,
        Op.GT: lambda: left.value > right.value,
        Op.GE: lambda: left.value >= right.value,
    }
    if comparator not in comparisons:
        return ExpressionDefect(0, f"unknown comparator {comparator!r}")
    if comparator not in (Op.EQ, Op.NE) and not (left.is_quantity and right.is_quantity):
        return ExpressionDefect(0, "ordering comparisons need quantities on both sides")
    return comparisons[comparator]()


def _aggregate(
    index: int,
    args: Mapping[str, Any],
    collection: Collection,
    completeness: Completeness,
) -> Union[tuple[Scalar, Basis], Defect]:
    op = args.get("op")
    if op == "count":
        return Scalar(ScalarKind.QUANTITY, len(collection.records)), _collection_basis(collection)

    path: Path | None = args.get("path")
    if path is None:
        return ExpressionDefect(index, f"aggregate '{op}' needs a path (only 'count' may omit one)")

    if op == "only":
        # Deliberately not "first". A cardinality assumption that is wrong should
        # be loud: reading field X off a collection that turned out to hold nine
        # records is a silently arbitrary answer, and this is usually reached
        # for exactly when the caller believes there is one.
        if len(collection.records) != 1:
            return DataDefect(
                index,
                f"'only' reads a field from a collection holding exactly one record, but "
                f"'{path.dotted}' was asked of {len(collection.records)}. Filter it down first.",
            )
        value = _resolve(path, collection.records[0])
        if value is _MISSING:
            return ExpressionDefect(index, f"'{path.dotted}' is not a field on that record")
        if not isinstance(value, Scalar):
            return ExpressionDefect(index, f"'{path.dotted}' is not a scalar")
        return value, collection.records[0].basis_for(path.dotted)

    values, bases, absent = [], [], 0
    for record in collection.records:
        raw = _resolve(path, record)
        if raw is _MISSING:
            absent += 1
            continue
        if not isinstance(raw, Scalar) or not raw.is_quantity:
            return ExpressionDefect(index, f"'{path.dotted}' is not a quantity on every record")
        values.append(raw.value)
        bases.append(record.basis_for(path.dotted))

    if absent and values:
        # Filtering tolerates absence; ACCOUNTING does not. Skipping the records
        # that lack the field would return a sum over a subset, stamped with the
        # confidence of a sum over everything -- the same silent-partial failure
        # as aggregating a truncated page, arriving by a different route.
        return DataDefect(
            index,
            f"{absent} of {len(collection.records)} records carry no value at '{path.dotted}', "
            f"so '{op}' would report a total over only {len(values)} of them. No edit to this "
            "pipeline can fix that -- the missing values were never retrieved.",
        )

    if not values:
        return DataDefect(index, f"no record carries a value at '{path.dotted}', so there is nothing to aggregate")

    computed = {
        "sum": sum(values),
        "avg": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }.get(op)
    if computed is None:
        return ExpressionDefect(index, f"unknown aggregate '{op}'")

    return Scalar(ScalarKind.QUANTITY, computed), weakest(bases)


def _group(
    index: int,
    args: Mapping[str, Any],
    collection: Collection,
    completeness: Completeness,
    basis: Basis,
) -> Union[tuple[Collection, Basis], Defect]:
    by: Sequence[Path] = args.get("by", ())
    if not by:
        return ExpressionDefect(
            index,
            "'group' needs 'by' naming the field(s) to partition on. To collapse the whole "
            "collection to one value, use 'aggregate' instead.",
        )
    if not args.get("agg"):
        # Bare group keys are what the codec collision produced for a while, and
        # they read as a successful grouping that simply found nothing to count.
        # Refusing names the operator that actually does this.
        return ExpressionDefect(
            index,
            "'group' with no aggregate returns nothing but the group keys. Name what to compute "
            'per group -- {"agg": {"how_many": {"agg": "count"}}} -- or use "distinct" if the '
            "distinct key values are all you wanted.",
        )
    for key in by:
        if _first_missing_key(key, collection.records) is not None:
            return DataDefect(index, f"a record has no value for the group key '{key.dotted}'; this fails closed")

    buckets: dict[tuple, list[Record]] = {}
    for record in collection.records:
        signature = tuple(_raw(_resolve(key, record)) for key in by)
        buckets.setdefault(signature, []).append(record)

    rows = []
    for signature, members in buckets.items():
        fields = {key.dotted: _resolve(key, members[0]) for key in by}
        for out_name, (agg_op, agg_path) in args.get("agg", {}).items():
            inner = _aggregate(index, {"op": agg_op, "path": agg_path}, Collection(tuple(members), completeness), completeness)
            if isinstance(inner, (ExpressionDefect, DataDefect)):
                return inner
            fields[out_name] = inner[0]
        rows.append(Record(fields=fields, basis=weakest([m.basis for m in members])))
    return Collection(tuple(rows), completeness), _basis_of(rows, basis)


def _eval_scalar(
    index: int,
    expression: ScalarExpr,
    record: Record,
    scalars: Mapping[str, Scalar] = MappingProxyType({}),
    bases: Mapping[str, Basis] = MappingProxyType({}),
) -> Union[tuple[Scalar, Basis], Defect]:
    if isinstance(expression, Literal):
        return expression.value, record.basis
    if isinstance(expression, Held):
        value = scalars.get(expression.name)
        if value is None:
            return ExpressionDefect(
                index,
                f"the expression refers to held scalar '{expression.name}', which is not a scalar "
                f"fact in scope. Available scalars: {sorted(scalars)}.",
            )
        return value, bases.get(expression.name, record.basis)
    if isinstance(expression, PathRef):
        value = _resolve(expression.path, record)
        if value is _MISSING:
            return ExpressionDefect(index, f"'{expression.path.dotted}' is not a field on these records")
        if not isinstance(value, Scalar):
            return ExpressionDefect(index, f"'{expression.path.dotted}' is not a scalar")
        return value, record.basis_for(expression.path.dotted)

    left = _eval_scalar(index, expression.left, record, scalars, bases)
    if isinstance(left, (ExpressionDefect, DataDefect)):
        return left
    right = _eval_scalar(index, expression.right, record, scalars, bases)
    if isinstance(right, (ExpressionDefect, DataDefect)):
        return right

    (left_value, left_basis), (right_value, right_basis) = left, right
    if not left_value.is_quantity or not right_value.is_quantity:
        return ExpressionDefect(index, "arithmetic needs quantities on both sides")
    if expression.op is ArithOp.DIVIDE and right_value.value == 0:
        return DataDefect(index, "division by zero: the divisor evaluated to 0 on a record")

    result = {
        ArithOp.ADD: left_value.value + right_value.value,
        ArithOp.SUBTRACT: left_value.value - right_value.value,
        ArithOp.MULTIPLY: left_value.value * right_value.value,
        ArithOp.DIVIDE: left_value.value / right_value.value if right_value.value else 0,
        ArithOp.MAX: max(left_value.value, right_value.value),
        ArithOp.MIN: min(left_value.value, right_value.value),
        ArithOp.GTE: left_value.value >= right_value.value,
        ArithOp.GT: left_value.value > right_value.value,
        ArithOp.LTE: left_value.value <= right_value.value,
        ArithOp.LT: left_value.value < right_value.value,
        ArithOp.EQ: left_value.value == right_value.value,
    }[expression.op]
    kind = ScalarKind.BOOL if expression.op in COMPARISONS else ScalarKind.QUANTITY
    return Scalar(kind, result), weakest([left_basis, right_basis])


def _unnest(
    index: int,
    path: Path,
    collection: Collection,
    completeness: Completeness,
    basis: Basis,
) -> Union[tuple[Collection, Basis], Defect]:
    """One record per element of a nested array, parent fields riding along.

    SQL lateral / Mongo `$unwind` semantics, and NOT a free choice: the type
    checker in `operators.py` already declares this shape -- parent fields kept,
    the array field replaced by the element's fields merged at top level. An
    implementation that nested the element under its own name instead would type
    -check against paths it then failed to resolve, so the checker's rule is the
    specification here rather than a parallel opinion.

    Nothing was calling this for a while: `unnest` sat in `OPERATORS`, passed
    the type checker, and got rendered into the system prompt by `catalog.py`,
    while the runner had no branch for it -- so a model following the prompt's
    own operator table hit "operator 'unnest' has no evaluation rule". Declared
    is not implemented.
    """
    expanded: list[Record] = []
    every_element_present = True
    for record in collection.records:
        elements = _resolve(path, record)
        if elements is _MISSING:
            return DataDefect(
                index,
                f"a record has no '{path.dotted}' to unnest. Expanding the records that do have "
                "one and silently dropping the rest would return a partial result that counts "
                "like a whole one, so this fails closed.",
            )
        if not isinstance(elements, Collection):
            return ExpressionDefect(
                index,
                f"'{path.dotted}' holds a single value, not an array, so there is nothing to "
                "expand. Read it directly instead of unnesting it.",
            )
        every_element_present &= elements.completeness.complete

        parent = {name: value for name, value in record.fields.items() if name != path.dotted}
        parent_basis = {name: record.basis_for(name) for name in parent}
        for element in elements.records:
            expanded.append(
                Record(
                    # The element's own fields win a name collision, matching the
                    # checker's `merged.update(inner.fields)`.
                    fields={**parent, **element.fields},
                    # The element is known on the same basis as the document it
                    # was stored in, so per-field provenance survives the expansion.
                    basis=element.basis,
                    field_basis={**parent_basis, **dict(element.field_basis)},
                )
            )

    # An empty array contributes no rows, exactly as `$unwind` does. That is a
    # real answer ("this plan has no semesters"), not a truncation -- so only a
    # genuinely partial nested array makes the result incomplete.
    return (
        Collection(
            tuple(expanded),
            completeness if every_element_present else Completeness(complete=False, total=None),
        ),
        _basis_of(expanded, basis),
    )


def _resolve(path: Path, record: Record) -> Any:
    """Delegates to `Path.resolve` -- deliberately NOT a second implementation.

    The exact-name-before-segment-walk rule lives in one place. A private copy
    here would be a second resolver for one rule, which is the same silent-drift
    trap as two predicate engines, and it would diverge exactly where `join`
    output is read back.
    """
    return path.resolve(record)


def _first_missing_key(path: Path, records: Sequence[Record]) -> Record | None:
    for record in records:
        value = _resolve(path, record)
        if value is _MISSING or (isinstance(value, Scalar) and value.value is None):
            return record
    return None


def _raw(value: Any) -> Any:
    return value.value if isinstance(value, Scalar) else value


_KNOWN_SOURCE_NAMES = frozenset({
    "courses", "completed_courses", "course_offerings", "semester_plans",
    "student_profiles", "degree_programs", "prerequisite_edges",
})


def _no_such_fact(role: str, name: str, available: Mapping[str, object]) -> str:
    """The name is not a held fact. If it is a data SOURCE, say so and how to fix it.

    The most common `compute` error a live model hits: it writes `join(other:
    "courses")` or `source: "course_offerings"`, naming a data SOURCE where a
    HELD FACT belongs. The old message ("unknown collection 'courses'; available:
    [...]") never said `courses` was a source it could fetch, so the model
    retried the same shape for turns. `compute` reads facts; `find` reads
    sources -- and that is the sentence the model needs.
    """
    base = f"'{name}' is not a fact you hold. Held facts: {sorted(available)}."
    if name in _KNOWN_SOURCE_NAMES:
        return (
            f"'{name}' is a data SOURCE, not a held fact -- `compute` only reads facts. Fetch it "
            f"first with a `find` (or a semi-join) into a named fact, then use that name here. {base}"
        )
    return base


def _signature(record: Record) -> tuple:
    """A hashable identity for a whole record, nested fields included.

    `distinct` and an unkeyed `difference` both put this in a set, so a field
    holding a sub-record or a nested array used to raise `TypeError:
    unhashable type: 'dict'` -- an exception escaping the runner rather than a
    defect it could report, which ends the turn instead of failing one stage.
    Unreachable while every source declared flat scalars; reachable the moment
    one declared an array.

    Field names are unique within a record, so `sorted` never has to compare two
    values of different shapes.
    """
    return tuple(sorted((name, _hashable(value)) for name, value in record.fields.items()))


def _hashable(value: Any) -> Any:
    if isinstance(value, Scalar):
        return value.value
    if isinstance(value, Record):
        return _signature(value)
    if isinstance(value, Collection):
        return tuple(_signature(record) for record in value.records)
    return value


def _basis_of(records: Sequence[Record], fallback: Basis) -> Basis:
    return weakest([r.basis for r in records]) if records else fallback


__all__ = ["Blocked", "Failed", "Outcome", "Succeeded", "run_pipelines"]
