"""`optimize` -- constrained search. Phase 7b of docs/agent/tools_implementation_plan.md.

The algebra EVALUATES a query over facts that exist. It cannot search a space of
assignments that do not exist yet, and no amount of selection or joining
constructs one. That is the boundary.

**The vocabulary is items / slots / constraints / objective, and nothing in it is
academic.** This is deliberate and it is the whole design risk: a
`generate_semester_plan(student, track)` would be the composite pattern back
under a new name -- one call, one pre-solved question shape, zero generality.
Courses and semesters are simply items and slots, the same way a build step and
a time window would be.

Eligibility reuses the ORDINARY predicate grammar, so there is one predicate
language across admission, filtering and search rather than a second one to keep
in step.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Union

from app.agent_core.facts.predicate import Predicate, matches
from app.agent_core.facts.types import (
    Basis,
    Collection,
    Completeness,
    Record,
    Scalar,
    ScalarKind,
)

DEFAULT_NODE_BUDGET = 10_000


@dataclass(frozen=True)
class Item:
    id: str
    attributes: Mapping[str, Scalar] = field(default_factory=dict)


@dataclass(frozen=True)
class Slot:
    id: str
    index: int
    attributes: Mapping[str, Scalar] = field(default_factory=dict)


@dataclass(frozen=True)
class Precedence:
    """`after` must land in a strictly later slot than `before`."""

    before: str
    after: str


@dataclass(frozen=True)
class Capacity:
    """The summed attribute of everything in one slot may not exceed `limit`."""

    attribute: str
    limit: float


@dataclass(frozen=True)
class Eligibility:
    """`item` may only occupy a slot whose attributes satisfy the predicate."""

    item: str
    slot_predicate: Predicate


Constraint = Union[Precedence, Capacity, Eligibility]


class Objective(Enum):
    MINIMIZE_SLOTS = "minimize_slots"
    """Pack: finish in as few slots as possible."""

    BALANCE_LOAD = "balance_load"
    """Spread: keep slot loads as even as possible."""

    FILL = "fill"
    """Fill a FIXED set of slots to capacity, leaving overflow unscheduled.

    The others place EVERY item and fail if they cannot -- right for "how many
    semesters to finish everything". This is the other planning question: "what
    goes in my next two semesters?" There the slots are fixed (winter, spring)
    and most remaining courses will NOT fit -- that is expected, not a failure.
    So `fill` places as many as fit, in the order given (the caller sorts by
    priority -- mandatory before elective), and reports the rest as
    "(unscheduled)" rather than refusing the whole plan.
    """


UNSCHEDULED = "(unscheduled)"
"""The slot label for an item that did not fit the fixed slots under `fill`.
A real answer, not an error: it is what belongs in a LATER semester."""


@dataclass(frozen=True)
class Plan:
    assignment: Collection
    proven_optimal: bool
    nodes_explored: int


@dataclass(frozen=True)
class Infeasible:
    """No assignment satisfies the constraints, and WHICH one blocked it.

    "No plan exists" is not actionable. "No plan exists because `c` weighs 9 and
    no slot admits more than 4" tells the caller what to change.
    """

    reason: str


def optimize(
    items: Sequence[Item],
    slots: Sequence[Slot],
    constraints: Sequence[Constraint] = (),
    objective: Objective = Objective.MINIMIZE_SLOTS,
    node_budget: int = DEFAULT_NODE_BUDGET,
) -> Union[Plan, Infeasible]:
    """Assign every item to a slot, or explain why that cannot be done."""
    precedence = [c for c in constraints if isinstance(c, Precedence)]
    capacities = [c for c in constraints if isinstance(c, Capacity)]
    eligibility: dict[str, list[Predicate]] = {}
    for constraint in constraints:
        if isinstance(constraint, Eligibility):
            eligibility.setdefault(constraint.item, []).append(constraint.slot_predicate)

    if objective is Objective.FILL:
        # A different question -- fill fixed slots, overflow allowed -- so a
        # different, non-backtracking path. It never returns Infeasible for
        # "too many items": that IS the answer for a two-semester plan.
        return _fill(items, slots, capacities, eligibility, precedence)

    ordered = _topological(items, precedence)
    if ordered is None:
        return Infeasible("the precedence constraints contain a cycle, so no ordering of the items exists")

    slots_by_index = sorted(slots, key=lambda s: s.index)
    allowed = {}
    for item in items:
        permitted = [s for s in slots_by_index if _eligible(item, s, eligibility)]
        if not permitted:
            return Infeasible(
                f"'{item.id}' is eligible for no slot: its eligibility predicate excludes all "
                f"{len(slots_by_index)} available slots."
            )
        if not _fits_anywhere(item, capacities):
            over = next(c for c in capacities if _weight(item, c.attribute) > c.limit)
            return Infeasible(
                f"'{item.id}' has {over.attribute}={_weight(item, over.attribute)}, which exceeds the "
                f"per-slot limit of {over.limit}. No slot can hold it."
            )
        allowed[item.id] = permitted

    placement: dict[str, Slot] = {}
    explored = _Counter(budget=node_budget)
    by_id = {item.id: item for item in items}

    if not _place(ordered, 0, placement, allowed, capacities, precedence, objective, explored, by_id):
        if explored.exhausted:
            return Infeasible(
                f"search budget of {node_budget} nodes was exhausted before a valid assignment was "
                "found; no conclusion about feasibility can be drawn from that."
            )
        return Infeasible("no assignment satisfies all constraints simultaneously")

    records = tuple(
        # A plan is a proposal about a future that has not happened. Anything
        # derived from it must be visibly weaker than anything derived from a
        # record -- which the basis ordering does without special-casing.
        Record(
            fields=_placement_fields(by_id[item_id].attributes, item_id, slot.id, slot.index),
            basis=Basis.SIMULATED,
        )
        for item_id, slot in sorted(placement.items(), key=lambda kv: (kv[1].index, kv[0]))
    )

    return Plan(
        assignment=Collection(
            records=records,
            completeness=Completeness(complete=not explored.exhausted, total=len(records)),
        ),
        # Greedy placement in a fixed order finds A valid assignment; it does not
        # show there is no better one. Claiming optimality would be a lie the
        # caller could not detect.
        proven_optimal=False,
        nodes_explored=explored.count,
    )


def _fill(
    items: Sequence[Item],
    slots: Sequence[Slot],
    capacities: Sequence[Capacity],
    eligibility: Mapping[str, Sequence[Predicate]],
    precedence: Sequence[Precedence],
) -> Plan:
    """Greedy first-fit into fixed slots; whatever does not fit stays unscheduled.

    Items are placed in the ORDER GIVEN -- the caller controls priority by
    sorting them (mandatory before elective, say) -- each into the earliest
    eligible slot that still has room and does not put it before a placed
    prerequisite. This finds A good filling, not a provably optimal one; a
    two-semester plan does not need the optimum, and claiming one would be a lie
    the caller could not check.
    """
    slots_by_index = sorted(slots, key=lambda s: s.index)
    load: dict[str, float] = {slot.id: 0.0 for slot in slots_by_index}
    placement: dict[str, Slot] = {}

    for item in items:
        chosen: Slot | None = None
        for slot in slots_by_index:
            if not _eligible(item, slot, eligibility):
                continue
            if any(
                load[slot.id] + _weight(item, capacity.attribute) > capacity.limit
                for capacity in capacities
            ):
                continue
            if not _precedence_holds(item, slot, placement, precedence):
                continue
            chosen = slot
            break
        if chosen is not None:
            placement[item.id] = chosen
            for capacity in capacities:
                load[chosen.id] += _weight(item, capacity.attribute)

    records = []
    for item in items:
        slot = placement.get(item.id)
        records.append(
            Record(
                # Unscheduled items sort last (a real index otherwise) and carry
                # their attributes onward like every placed row.
                fields=_placement_fields(
                    item.attributes,
                    item.id,
                    slot.id if slot else UNSCHEDULED,
                    slot.index if slot else _UNSCHEDULED_INDEX,
                ),
                basis=Basis.SIMULATED,
            )
        )
    records.sort(key=lambda r: (r.fields["slot_index"].value, r.fields["item"].value))

    return Plan(
        assignment=Collection(
            records=tuple(records),
            # Every item is accounted for -- placed or explicitly unscheduled --
            # so the plan is complete even though not everything was scheduled.
            completeness=Completeness(complete=True, total=len(records)),
        ),
        proven_optimal=False,
        nodes_explored=len(items),
    )


_UNSCHEDULED_INDEX = 10**9
"""Sorts unscheduled items after every real slot."""


def _placement_fields(
    attributes: Mapping[str, Scalar], item_id: str, slot_id: str, slot_index: int
) -> dict[str, Scalar]:
    """A placed row: the item's own attributes, PLUS where it landed.

    The attributes ride along -- a placed course keeps its `credits`, title and
    kind -- because reading them back is almost always the caller's next move:
    split the plan by `slot`, total `credits` per semester, or compute a
    per-course figure (a min grade to hold a GPA) from `credits`. Dropping them
    left the plan a bare list of ids, and every use then needed a re-join the
    caller kept getting wrong on the last mile. The structural keys are written
    last so `item`/`slot`/`slot_index` always win a name clash and mean the
    placement's own, never an item attribute that happened to share the name.
    """
    return {
        **dict(attributes),
        "item": Scalar(ScalarKind.IDENTIFIER, item_id),
        "slot": Scalar(ScalarKind.IDENTIFIER, slot_id),
        "slot_index": Scalar(ScalarKind.QUANTITY, slot_index),
    }


@dataclass
class _Counter:
    budget: int
    count: int = 0
    exhausted: bool = False

    def tick(self) -> bool:
        self.count += 1
        if self.count > self.budget:
            self.exhausted = True
            return False
        return True


def _place(
    ordered: Sequence[Item],
    position: int,
    placement: dict[str, Slot],
    allowed: Mapping[str, Sequence[Slot]],
    capacities: Sequence[Capacity],
    precedence: Sequence[Precedence],
    objective: Objective,
    explored: _Counter,
    by_id: Mapping[str, Item],
) -> bool:
    """Backtracking placement. Deterministic: candidate order is fixed, so the
    same problem always yields the same plan -- a planner that answers
    differently each time is unusable however good any one answer is."""
    if position == len(ordered):
        return True
    if explored.exhausted:
        return False

    item = ordered[position]
    for slot in _candidates(item, allowed, placement, capacities, objective):
        if not explored.tick():
            return False
        if not _precedence_holds(item, slot, placement, precedence):
            continue
        if not _capacity_holds(item, slot, placement, capacities, by_id):
            continue

        placement[item.id] = slot
        if _place(ordered, position + 1, placement, allowed, capacities, precedence, objective, explored, by_id):
            return True
        del placement[item.id]

    return False


def _candidates(
    item: Item,
    allowed: Mapping[str, Sequence[Slot]],
    placement: Mapping[str, Slot],
    capacities: Sequence[Capacity],
    objective: Objective,
) -> Sequence[Slot]:
    options = list(allowed[item.id])
    if objective is Objective.MINIMIZE_SLOTS:
        # Earliest first: pack into slots already in use.
        return options
    # BALANCE_LOAD: try the emptiest slot first, ties broken by index so the
    # result stays deterministic.
    loads = {slot.id: 0.0 for slot in options}
    for placed_item, slot in placement.items():
        if slot.id in loads:
            loads[slot.id] += 1.0
    return sorted(options, key=lambda s: (loads.get(s.id, 0.0), s.index))


def _precedence_holds(
    item: Item,
    slot: Slot,
    placement: Mapping[str, Slot],
    precedence: Sequence[Precedence],
) -> bool:
    for rule in precedence:
        if rule.after == item.id and rule.before in placement:
            if placement[rule.before].index >= slot.index:
                return False
        if rule.before == item.id and rule.after in placement:
            if slot.index >= placement[rule.after].index:
                return False
    return True


def _capacity_holds(
    item: Item,
    slot: Slot,
    placement: Mapping[str, Slot],
    capacities: Sequence[Capacity],
    by_id: Mapping[str, Item],
) -> bool:
    """Would adding `item` to `slot` push any capacity past its limit?

    Attributes are read back through `by_id` rather than cached in module state:
    a cache here would be shared across concurrent calls with different item
    sets, and a capacity check reading another call's weights fails OPEN --
    producing an over-loaded plan that looks valid.
    """
    for capacity in capacities:
        used = sum(
            _weight(by_id[placed_id], capacity.attribute)
            for placed_id, placed_slot in placement.items()
            if placed_slot.id == slot.id
        )
        if used + _weight(item, capacity.attribute) > capacity.limit:
            return False
    return True


def _weight(item: Item, attribute: str) -> float:
    value = item.attributes.get(attribute)
    return float(value.value) if isinstance(value, Scalar) and value.is_quantity else 0.0


def _fits_anywhere(item: Item, capacities: Sequence[Capacity]) -> bool:
    return all(_weight(item, c.attribute) <= c.limit for c in capacities)


def _eligible(item: Item, slot: Slot, eligibility: Mapping[str, Sequence[Predicate]]) -> bool:
    predicates = eligibility.get(item.id)
    if not predicates:
        return True
    as_record = Record(fields=dict(slot.attributes), basis=Basis.OFFICIAL_RECORD)
    return all(matches(predicate, as_record) for predicate in predicates)


def _topological(items: Sequence[Item], precedence: Sequence[Precedence]) -> Sequence[Item] | None:
    """Items in dependency order, or `None` when the constraints contain a cycle."""
    by_id = {item.id: item for item in items}
    pending = {
        item.id: {rule.before for rule in precedence if rule.after == item.id and rule.before in by_id}
        for item in items
    }

    ordered: list[Item] = []
    while pending:
        ready = sorted(name for name, deps in pending.items() if not deps)
        if not ready:
            return None
        for name in ready:
            ordered.append(by_id[name])
            del pending[name]
        for deps in pending.values():
            deps.difference_update(ready)
    return ordered


__all__ = [
    "DEFAULT_NODE_BUDGET",
    "Capacity",
    "Constraint",
    "Eligibility",
    "Infeasible",
    "Item",
    "Objective",
    "Plan",
    "Precedence",
    "Slot",
    "UNSCHEDULED",
    "optimize",
]
