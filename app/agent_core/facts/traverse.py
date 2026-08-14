"""`traverse` -- transitive closure. Phase 7 of docs/agent/tools_implementation_plan.md.

The primitive with a theorem behind it. Relational algebra provably cannot
express transitive closure -- that is why SQL needed `WITH RECURSIVE` -- so no
pipeline over the basis, however long, reaches a prerequisite chain of unknown
depth. A fixed pipeline of N joins reaches exactly N levels, and the depth is a
property of the data.

It takes a Collection of edges and returns a Collection of reached nodes, so
admission feeds it and the algebra consumes its output. The recursion is the
only thing that happens outside the basis.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

from app.agent_core.facts.predicate import Path
from app.agent_core.facts.types import (
    Basis,
    Collection,
    Completeness,
    Record,
    Scalar,
    ScalarKind,
    weakest,
)

DEFAULT_MAX_DEPTH = 10


def traverse(
    edges: Collection,
    start: str,
    *,
    from_path: Path,
    to_path: Path,
    max_depth: int = DEFAULT_MAX_DEPTH,
    carry: Sequence[str] = (),
) -> Collection:
    """Every node reachable from `start`, with its shortest depth.

    Breadth-first, so a node reached by two routes is reported once at the
    shorter one. `carry` copies named edge fields onto the reached node -- used
    for the alternative-group label, because prerequisites carry AND/OR
    structure and flattening two alternatives into a plain reachable set reports
    every option as mandatory.

    Evaluating AND/OR SATISFACTION is a separate recursive problem and is
    deliberately not claimed here; this preserves the structure so something
    else can.
    """
    outgoing: dict[str, list[Record]] = {}
    for edge in edges.records:
        source = _value(from_path, edge)
        if source is not None:
            outgoing.setdefault(source, []).append(edge)

    reached: dict[str, Record] = {}
    queue: deque[tuple[str, int, Basis]] = deque([(start, 0, Basis.OFFICIAL_RECORD)])
    seen = {start}
    truncated = False

    while queue:
        node, depth, inherited = queue.popleft()
        if depth >= max_depth:
            # Only a truncation if this node actually had somewhere else to go.
            if outgoing.get(node):
                truncated = True
            continue

        for edge in outgoing.get(node, ()):
            target = _value(to_path, edge)
            if target is None:
                continue

            # A node is only as certain as the weakest edge on the path to it.
            basis = weakest([inherited, edge.basis]) if depth else edge.basis

            if target not in reached:
                fields = {
                    "node": Scalar(ScalarKind.IDENTIFIER, target),
                    "depth": Scalar(ScalarKind.QUANTITY, depth + 1),
                }
                for name in carry:
                    value = edge.fields.get(name)
                    if value is not None:
                        fields[name] = value
                reached[target] = Record(fields=fields, basis=basis)

            if target not in seen:
                seen.add(target)
                queue.append((target, depth + 1, basis))

    return Collection(
        records=tuple(reached.values()),
        # A bounded walk is a truncation like any other: an aggregate over a
        # partial chain must fail closed rather than count it.
        completeness=Completeness(
            complete=edges.completeness.complete and not truncated,
            total=len(reached) if not truncated else None,
        ),
    )


def _value(path: Path, record: Record) -> str | None:
    resolved = path.resolve(record)
    return resolved.value if isinstance(resolved, Scalar) else None


__all__ = ["DEFAULT_MAX_DEPTH", "traverse"]
