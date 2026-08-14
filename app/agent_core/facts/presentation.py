"""How facts are shown to the model -- phase 9c of docs/agent/tools_implementation_plan.md.

The model writes pipelines against whatever it believes it is holding, so this
is where most avoidable failures are either prevented or invited.

Three things it must show, and one it must not:

  FIELDS AND THEIR KINDS -- so `select` names a field that exists and `arith`
  operates on a quantity rather than a course code.

  COVERAGE -- which fields are missing on some records. Aggregating a field that
  53 records have and 7 do not fails closed, and saying so up front costs a line
  here instead of a turn there.

  COMPLETENESS -- with the consequence stated, not just the flag. "Truncated"
  means nothing to a caller who does not know that aggregates over it refuse.

  NEVER THE PAYLOAD. A working set that inlines its data stops being a working
  set. Shapes and counts, never rows.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Union

from app.agent_core.facts.types import Collection, Record, Scalar, ScalarKind

_MAX_FIELDS_SHOWN = 12
_SAMPLE_VALUES = 3


def render_facts(facts: Mapping[str, Union[Collection, Scalar]]) -> str:
    """The facts block of the working set."""
    if not facts:
        return "  (none yet)"
    return "\n".join(_render_one(name, value) for name, value in facts.items())


def _render_one(name: str, value: Union[Collection, Scalar]) -> str:
    if isinstance(value, Scalar):
        return f"  {name} = {_render_scalar(value)}"
    return _render_collection(name, value)


def _render_scalar(value: Scalar) -> str:
    return f"{value.value} ({value.kind.value})"


def _render_collection(name: str, collection: Collection) -> str:
    count = len(collection.records)
    completeness = collection.completeness

    if completeness.complete:
        header = f"  {name} = [{count} records]"
    else:
        # The consequence, not just the flag. A model told "truncated" will
        # still try to count it; a model told the count will be REFUSED will
        # fetch the rest first.
        of = f" of {completeness.total}" if completeness.total is not None else ""
        header = (
            f"  {name} = [{count}{of} records, TRUNCATED -- "
            "count/sum/average over this will be refused]"
        )

    if not collection.records:
        return header + "\n      (no records; a filter over this yields nothing)"

    lines = [header]
    for line in _render_fields(collection):
        lines.append(f"      {line}")
    bases = {record.basis.label for record in collection.records}
    lines.append(f"      basis: {', '.join(sorted(bases))}")
    return "\n".join(lines)


def _render_fields(collection: Collection) -> list[str]:
    """Every field across ALL records, with kind and coverage.

    The union, not a sample of the first record. Sampling under-reports the
    moment the data is uneven -- and uneven is the normal case, since a record
    missing a field is exactly what dirty upstream data produces.
    """
    total = len(collection.records)
    seen: dict[str, tuple[str, int]] = {}

    for record in collection.records:
        for field, value in record.fields.items():
            kind = _kind_of(value)
            previous = seen.get(field)
            if previous is None:
                seen[field] = (kind, 1)
            else:
                # A field with two kinds across records is worth flagging rather
                # than silently picking one -- it means an operator will fail on
                # some rows and not others.
                merged = previous[0] if previous[0] == kind else f"{previous[0]}|{kind}"
                seen[field] = (merged, previous[1] + 1)

    lines = []
    for field, (kind, present) in sorted(seen.items())[:_MAX_FIELDS_SHOWN]:
        coverage = "" if present == total else f" -- on {present} of {total}, aggregates will refuse"
        lines.append(f"{field}: {kind}{coverage}")

    if len(seen) > _MAX_FIELDS_SHOWN:
        lines.append(f"... and {len(seen) - _MAX_FIELDS_SHOWN} more fields")

    scalars = _sample_scalars(collection)
    if scalars:
        lines.append(f"e.g. {scalars}")
    return lines


def _sample_scalars(collection: Collection) -> str:
    """A few values from the identifying field, so the fact reads as real data.

    Measured previously: a bare "[list of N items]" reads as empty, and a
    sub-loop gave up on a list it was actually holding.
    """
    identifiers = [
        field
        for field, value in collection.records[0].fields.items()
        if isinstance(value, Scalar) and value.kind is ScalarKind.IDENTIFIER
    ]
    if not identifiers:
        return ""
    field = identifiers[0]
    values = []
    for record in collection.records[:_SAMPLE_VALUES]:
        value = record.fields.get(field)
        if isinstance(value, Scalar):
            values.append(str(value.value))
    more = ", …" if len(collection.records) > _SAMPLE_VALUES else ""
    return f"{field}: {', '.join(values)}{more}" if values else ""


def _kind_of(value: object) -> str:
    if isinstance(value, Scalar):
        return value.kind.value
    if isinstance(value, Collection):
        return "collection"
    if isinstance(value, Record):
        return "record"
    return type(value).__name__


__all__ = ["render_facts"]
