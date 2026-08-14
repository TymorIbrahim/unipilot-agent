"""`find` -- structured admission. Phase 5 of docs/agent/tools_implementation_plan.md.

The algebra is closed over facts and no operator produces a fact from nothing,
so something has to let records in. This is it, and identity fetch is just the
degenerate predicate `key == X` rather than a second tool.

Two jobs beyond fetching:

**Typing.** Mongo stores `"3.5"` and `"00940224"` as the same Python type. One is
a quantity and one is an identifier, and no heuristic separates them -- a leading
zero is a convention, not a type. So `find` requires a declared `SourceSchema`
and converts against it. This is the one place phase 1's ban on
`Scalar(QUANTITY, "3.5")` is lifted, because here the kind is *declared*.

**Completeness.** A page of 50 from a true 73 yields a confidently wrong `count`
while every fact in it reports full confidence. `find` establishes `total` with a
source-side count and marks the collection complete only when the records
account for all of it -- measured against the PREDICATE, not the collection, or
every filtered result would be permanently unaggregatable.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from numbers import Number
from typing import Any, Union

from app.agent_core.facts.operators import DataDefect, Defect, ExpressionDefect
from app.agent_core.facts.predicate import (
    Always,
    Predicate,
    collect_paths,
    compile_to_mongo,
    matches,
)
from app.agent_core.facts.types import (
    Basis,
    Collection,
    Completeness,
    Record,
    Scalar,
    ScalarKind,
)

DEFAULT_LIMIT = 200
"""Never unbounded. An unbounded fetch is how a working set silently becomes
a whole catalog; a truncated one at least says so."""


@dataclass(frozen=True)
class Sub:
    """A declared sub-document -- a field holding an object rather than a value."""

    fields: Mapping[str, "FieldSpec"]


@dataclass(frozen=True)
class ArrayOf:
    """A field holding an array. `element` declares what ONE element looks like.

    Arrays are the half of the data model that `find` could not see. `unnest`
    was in the operator table and in the system prompt, but every source
    declared flat scalars only, so nothing could produce an array for it to
    expand -- and `semester_plans.semesters[]`, the only stored thing shaped
    like `optimize`'s slots, was invisible.
    """

    element: Union[ScalarKind, Sub]


FieldSpec = Union[ScalarKind, Sub, ArrayOf]


@dataclass(frozen=True)
class SourceSchema:
    """What a stored collection contains, declared rather than inferred."""

    collection: str
    key: str
    fields: Mapping[str, FieldSpec]
    basis: Basis
    joins: tuple[tuple[str, str], ...] = ()
    """How this source reaches another: `(local path, "other_source.field")`.

    The sources list showed each collection's fields and nothing about how they
    relate, so the route from a transcript row to a course CODE -- join
    `courseId` to `courses._id` -- was invisible. Across a live eval the model
    repeatedly projected `courseId` under the name `courseCode` instead, which
    is precisely the fact-naming lie the system prompt warns about: the value is
    a real ObjectId and the name says course code.
    """

    yields: frozenset[str] = frozenset()
    """Tool input kinds this source can supply -- "edges", "slots".

    What makes a tool's advertisement honest. `traverse` needs edges and
    `optimize` needs slots; both were offered to the model while no source
    produced either, so the model could reach for them and never succeed.
    Declared on the source rather than listed beside the catalog, because the
    two drift apart the moment a source is added or removed.
    """

    object_id_fields: frozenset[str] = frozenset()
    """Identifier fields stored as BSON ObjectId rather than as strings.

    Declared, because both read as `IDENTIFIER` and nothing in the value
    distinguishes them. Without this, a filter like `userId = "6a5cfb..."`
    compiles to a string comparison against an ObjectId, matches nothing, and
    returns an EMPTY result marked complete -- a student with no transcript,
    reported confidently. Silence is the worst failure this layer can produce,
    so the mapping is explicit rather than guessed from the value's shape.
    """


@dataclass(frozen=True)
class DerivedSchema:
    """A source whose records are COMPUTED rather than stored.

    Prerequisite edges are the case that forced this. They are not in any
    collection -- `courses.prerequisitesText` is prose -- so `traverse` had no
    reachable input and was advertised to the model anyway. The alternative was
    a `get_prerequisites` tool, which is the composite pattern returning under a
    new name: one call, one pre-solved question, no generality.

    A SOURCE instead of a TOOL. The tool set stays at eight and stays general;
    sources are where the domain lives, which is already true of `courses` and
    `completed_courses`. To the model this is just another name in `find`.
    """

    collection: str
    key: str
    fields: Mapping[str, FieldSpec]
    basis: Basis
    produce: Callable[[], Sequence[Mapping[str, Any]]]
    joins: tuple[tuple[str, str], ...] = ()
    yields: frozenset[str] = frozenset()


AnySchema = Union[SourceSchema, DerivedSchema]


async def find(
    database: Any,
    schema: AnySchema,
    predicate: Predicate | None = None,
    limit: int = DEFAULT_LIMIT,
) -> Union[Collection, Defect]:
    """Fetch typed, provenance-stamped records with honest completeness."""
    predicate = predicate or Always()

    unknown = sorted(p.dotted for p in collect_paths(predicate) if _spec_at(schema.fields, p.dotted) is None)
    if unknown:
        return ExpressionDefect(
            0,
            f"{', '.join(repr(u) for u in unknown)} is not a field on '{schema.collection}'. "
            f"Available fields: {declared_paths(schema)}.",
        )

    if isinstance(schema, DerivedSchema):
        return _find_derived(schema, predicate, limit)

    query = _bind_object_ids(compile_to_mongo(predicate), schema)
    collection = database[schema.collection]

    # Counted against the same filter the fetch uses, so a filtered result can
    # still be complete.
    total = await collection.count_documents(query)

    # A stable sort is not cosmetic: without one the PAGE varies between runs and
    # every answer derived from it varies too.
    cursor = collection.find(query).sort(schema.key, 1).limit(limit)
    documents = [document async for document in cursor]

    records = []
    for document in documents:
        converted = _to_record(document, schema)
        if isinstance(converted, DataDefect):
            return converted
        records.append(converted)

    return Collection(
        records=tuple(records),
        completeness=Completeness(complete=len(records) == total, total=total),
    )


def _find_derived(
    schema: DerivedSchema, predicate: Predicate, limit: int
) -> Union[Collection, Defect]:
    """The same contract as a stored fetch: typed, sorted, honestly counted.

    The predicate runs in memory here rather than being pushed down. That is
    not a second semantics -- `matches` and `compile_to_mongo` are two engines
    for ONE grammar, and `test_predicate.py` holds them to identical results
    across the shared matrix.
    """
    records = []
    for document in schema.produce():
        converted = _to_record(document, schema)
        if isinstance(converted, DataDefect):
            return converted
        if matches(predicate, converted):
            records.append(converted)

    # Sorted for the same reason a stored fetch is: without a stable order the
    # PAGE varies between runs, and every answer derived from it varies too.
    records.sort(key=lambda record: str(record.fields[schema.key].value))
    total = len(records)
    kept = tuple(records[:limit])

    return Collection(
        records=kept,
        completeness=Completeness(complete=len(kept) == total, total=total),
    )


def _bind_object_ids(query: Any, schema: SourceSchema) -> Any:
    """Rewrite comparisons on ObjectId-backed fields to compare against ObjectIds.

    A predicate arrives with string values, because that is what a model can
    write. Comparing a string to a stored ObjectId matches nothing and reports
    an empty result as COMPLETE -- so a student's whole transcript would come
    back as "no records" with full confidence.

    Walks the compiled filter rather than the predicate tree so it also covers
    the operators the predicate grammar compiles into (`$in`, `$and`, ...).
    """
    from bson import ObjectId
    from bson.errors import InvalidId

    if not schema.object_id_fields:
        return query

    def convert(value: Any) -> Any:
        if isinstance(value, str):
            try:
                return ObjectId(value)
            except (InvalidId, TypeError):
                # Not an ObjectId-shaped string. Left alone rather than
                # rejected: the field may legitimately hold both during a
                # migration, and a wrong value should return nothing, not raise.
                return value
        if isinstance(value, list):
            return [convert(item) for item in value]
        return value

    def walk(node: Any, field: str | None = None) -> Any:
        if isinstance(node, dict):
            rewritten = {}
            for key, value in node.items():
                if key.startswith("$"):
                    rewritten[key] = (
                        [walk(item, field) for item in value]
                        if isinstance(value, list) and key in ("$and", "$or", "$nor")
                        else (convert(value) if field in schema.object_id_fields else walk(value, field))
                    )
                else:
                    rewritten[key] = walk(value, key)
            return rewritten
        return node

    return walk(query)


def _to_record(document: Mapping[str, Any], schema: AnySchema) -> Union[Record, DataDefect]:
    fields = _convert_fields(document, schema.fields, schema.basis)

    if schema.key not in fields:
        stored = document.get(schema.key, _ABSENT)
        return _unresolvable_key(
            document,
            schema,
            reason="is missing" if stored is _ABSENT else "could not be read as an identifier",
        )

    return Record(fields=fields, basis=schema.basis)


_ABSENT = object()


def _convert_fields(
    document: Mapping[str, Any], declared: Mapping[str, FieldSpec], basis: Basis
) -> dict[str, Any]:
    """Declared fields of one document, typed. Undeclared keys never enter.

    A value that cannot be honoured is OMITTED rather than defaulted -- a `0`
    here would make a sum quietly wrong, where an absent field makes any
    aggregate over it fail closed with a message naming the field. The key's
    absence is caught by the caller, which alone knows which field is the key.
    """
    fields: dict[str, Any] = {}

    for name, spec in declared.items():
        if name not in document:
            continue
        value = _convert(document[name], spec, basis, name)
        if value is not None:
            fields[name] = value

    return fields


def _convert(value: Any, spec: FieldSpec, basis: Basis, name: str) -> Any:
    """One stored value -> a `FieldValue`, or `None` when it cannot be honoured."""
    if isinstance(spec, ScalarKind):
        coerced = _coerce(value, spec)
        return None if coerced is None else Scalar(spec, coerced)

    if isinstance(spec, Sub):
        if not isinstance(value, Mapping):
            return None
        return Record(fields=_convert_fields(value, spec.fields, basis), basis=basis)

    if not isinstance(value, (list, tuple)):
        # Declared an array, stored as something else. Omitted rather than
        # wrapped in a one-element array: silently promoting a scalar would make
        # `unnest` report one row where the declaration promised a list, and the
        # caller could not tell the difference.
        return None

    records = []
    for element in value:
        if isinstance(spec.element, ScalarKind):
            coerced = _coerce(element, spec.element)
            if coerced is None:
                continue
            # Named after the field it came from, because `unnest` merges an
            # element's fields into the parent: a generic name like `value`
            # would collide the moment two such arrays were expanded in one
            # pipeline, and the second would silently overwrite the first.
            records.append(Record(fields={name: Scalar(spec.element, coerced)}, basis=basis))
        elif isinstance(element, Mapping):
            records.append(Record(fields=_convert_fields(element, spec.element.fields, basis), basis=basis))

    return Collection(
        records=tuple(records),
        # Every element of a stored array is in hand: the whole document was
        # read. Marking it unknown would make any aggregate over an unnested
        # array fail closed for no reason.
        completeness=Completeness(complete=True, total=len(records)),
    )


def declared_paths(schema: AnySchema) -> list[str]:
    """Every readable path on a source, nested ones included.

    The unknown-field message lists these. Listing only top-level names while
    the schema declares nested structure would tell a model that
    `semesters.order` does not exist when it does.
    """
    def walk(declared: Mapping[str, FieldSpec], prefix: str) -> list[str]:
        found: list[str] = []
        for name, spec in declared.items():
            dotted = f"{prefix}{name}"
            if isinstance(spec, ScalarKind):
                found.append(dotted)
            elif isinstance(spec, Sub):
                found.extend(walk(spec.fields, f"{dotted}."))
            elif isinstance(spec.element, ScalarKind):
                found.append(dotted)
            else:
                found.append(dotted)
                found.extend(walk(spec.element.fields, f"{dotted}."))
        return found

    return sorted(walk(schema.fields, ""))


def array_paths(schema: AnySchema) -> frozenset[str]:
    """Paths that hold an array, so a caller can mark them as needing `unnest`.

    The prompt lists a source's fields; without this it would show `semesters`
    beside ordinary scalars and the model would filter on it as though it were
    one.
    """
    return frozenset(
        path for path in declared_paths(schema) if isinstance(_spec_at(schema.fields, path), ArrayOf)
    )


def _spec_at(declared: Mapping[str, FieldSpec], dotted: str) -> FieldSpec | None:
    """The declaration at a dotted path, or `None` when nothing is declared there.

    Walks THROUGH arrays without an index, which is what Mongo does: a filter on
    `semesters.order` matches a document if any element matches.
    """
    spec: FieldSpec | None = None
    current: Mapping[str, FieldSpec] | None = declared

    for segment in dotted.split("."):
        if current is None:
            return None
        spec = current.get(segment)
        if spec is None:
            return None
        if isinstance(spec, ScalarKind):
            current = None
        elif isinstance(spec, Sub):
            current = spec.fields
        else:
            current = spec.element.fields if isinstance(spec.element, Sub) else None

    return spec


def _unresolvable_key(document: Mapping[str, Any], schema: AnySchema, *, reason: str) -> DataDefect:
    """A record whose identity cannot be established is refused outright.

    Admitting it without a key would let a later `difference` or `join` drop it
    silently, and a set difference that silently drops records RETAINS what it
    should have removed -- while every surviving fact still reports full
    confidence. Loudly here beats wrongly later.
    """
    return DataDefect(
        0,
        f"a record in '{schema.collection}' has no usable key: '{schema.key}' {reason}. "
        f"Fields present: {sorted(k for k in document if k != '_id')}. "
        "Records without identity are refused rather than admitted, because a later "
        "difference or join would drop them silently and quietly return the wrong set.",
    )


def _coerce(value: Any, kind: ScalarKind) -> Any:
    """Storage value -> declared kind, or `None` when it cannot be honoured."""
    if value is None:
        return None

    if kind is ScalarKind.QUANTITY:
        if isinstance(value, bool):
            return None
        if isinstance(value, Number):
            return value
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return None
        return None

    if kind is ScalarKind.IDENTIFIER:
        # Deliberately str() rather than a numeric check: the declaration already
        # settled that this is an identifier, so a leading zero must survive.
        if isinstance(value, (str, int)):
            return str(value)
        # A Mongo ObjectId IS an identifier -- it is how `completed_courses`
        # references a catalog row. Refusing it would make every transcript
        # fetch fail at admission, since the key itself is one.
        if type(value).__name__ == "ObjectId":
            return str(value)
        return None

    if kind is ScalarKind.TEXT:
        return value if isinstance(value, str) else None

    if kind is ScalarKind.BOOL:
        return value if isinstance(value, bool) else None

    if kind is ScalarKind.DATE:
        return value if isinstance(value, (date, datetime)) else None

    return None


__all__ = [
    "DEFAULT_LIMIT",
    "AnySchema",
    "ArrayOf",
    "DerivedSchema",
    "FieldSpec",
    "SourceSchema",
    "Sub",
    "declared_paths",
    "find",
]
