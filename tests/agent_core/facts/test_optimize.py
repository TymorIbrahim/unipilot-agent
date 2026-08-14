"""`optimize` -- phase 7b of docs/agent/tools_implementation_plan.md.

Constrained search. Relational algebra EVALUATES a query over facts that exist;
it cannot search a space of assignments that do not exist yet. That is the
boundary, and plan generation lives on the far side of it.

The risk this file exists to hold off: `optimize` becoming
`generate_semester_plan(student, track)`. That would be the composite pattern
returning under a new name -- a pre-solved question shape, one call, zero
generality. So the vocabulary here is items / slots / constraints / objective,
with nothing academic in it, and the tests use a non-academic problem to prove
the point.
"""

from __future__ import annotations

from app.agent_core.facts.operators import DataDefect, Pipeline, Stage
from app.agent_core.facts.optimize import (
    Capacity,
    Eligibility,
    Infeasible,
    Item,
    Objective,
    Precedence,
    Slot,
    optimize,
)
from app.agent_core.facts.predicate import Always, Comparison, Op, Path
from app.agent_core.facts.runner import Succeeded, run_pipelines
from app.agent_core.facts.types import Basis, Scalar, ScalarKind

Q = ScalarKind.QUANTITY
I = ScalarKind.IDENTIFIER
T = ScalarKind.TEXT


def _item(name: str, weight: float) -> Item:
    return Item(id=name, attributes={"weight": Scalar(Q, weight)})


def _slot(name: str, index: int, kind: str = "any") -> Slot:
    return Slot(id=name, index=index, attributes={"kind": Scalar(I, kind)})


SLOTS = (_slot("s1", 0), _slot("s2", 1), _slot("s3", 2))


class TestPrecedence:
    def test_a_dependency_is_placed_before_its_dependent(self) -> None:
        result = optimize(
            items=(_item("a", 1), _item("b", 1)),
            slots=SLOTS,
            constraints=(Precedence(before="b", after="a"),),
        )
        placed = {r.fields["item"].value: r.fields["slot_index"].value for r in result.assignment.records}
        assert placed["b"] < placed["a"]

    def test_a_chain_of_precedence_spreads_across_slots(self) -> None:
        result = optimize(
            items=(_item("a", 1), _item("b", 1), _item("c", 1)),
            slots=SLOTS,
            constraints=(Precedence(before="a", after="b"), Precedence(before="b", after="c")),
        )
        placed = {r.fields["item"].value: r.fields["slot_index"].value for r in result.assignment.records}
        assert placed["a"] < placed["b"] < placed["c"]

    def test_a_precedence_cycle_is_reported_as_infeasible(self) -> None:
        result = optimize(
            items=(_item("a", 1), _item("b", 1)),
            slots=SLOTS,
            constraints=(Precedence(before="a", after="b"), Precedence(before="b", after="a")),
        )
        assert isinstance(result, Infeasible)
        assert "cycle" in result.reason.lower()


class TestCapacity:
    def test_a_slot_is_not_loaded_past_its_limit(self) -> None:
        result = optimize(
            items=(_item("a", 3), _item("b", 3), _item("c", 3)),
            slots=SLOTS,
            constraints=(Capacity(attribute="weight", limit=4.0),),
        )
        per_slot: dict[float, float] = {}
        for record in result.assignment.records:
            index = record.fields["slot_index"].value
            per_slot[index] = per_slot.get(index, 0) + 3
        assert all(total <= 4.0 for total in per_slot.values())

    def test_an_item_larger_than_any_slot_is_infeasible_and_says_which(self) -> None:
        """'No plan exists' is useless. 'No plan exists because c weighs 9 and
        no slot holds more than 4' is something a caller can act on."""
        result = optimize(
            items=(_item("c", 9),),
            slots=SLOTS,
            constraints=(Capacity(attribute="weight", limit=4.0),),
        )
        assert isinstance(result, Infeasible)
        assert "c" in result.reason


class TestEligibility:
    def test_an_item_only_goes_where_its_predicate_holds(self) -> None:
        """Eligibility reuses the SAME predicate grammar as `find` and `select`.
        One predicate language across admission, filtering and search -- a second
        one would be a second thing to keep in step."""
        winter_only = Eligibility(
            item="w",
            slot_predicate=Comparison(Path.parse("kind"), Op.EQ, Scalar(I, "winter")),
        )
        slots = (_slot("s1", 0, "spring"), _slot("s2", 1, "winter"), _slot("s3", 2, "spring"))
        result = optimize(items=(_item("w", 1),), slots=slots, constraints=(winter_only,))
        assert result.assignment.records[0].fields["slot"].value == "s2"

    def test_an_item_eligible_nowhere_is_infeasible(self) -> None:
        impossible = Eligibility(
            item="x", slot_predicate=Comparison(Path.parse("kind"), Op.EQ, Scalar(I, "summer"))
        )
        result = optimize(items=(_item("x", 1),), slots=SLOTS, constraints=(impossible,))
        assert isinstance(result, Infeasible)
        assert "x" in result.reason

    def test_an_always_predicate_admits_every_slot(self) -> None:
        result = optimize(
            items=(_item("a", 1),), slots=SLOTS, constraints=(Eligibility(item="a", slot_predicate=Always()),)
        )
        assert result.assignment.records[0].fields["slot_index"].value == 0


class TestOutput:
    def test_the_assignment_is_a_collection_the_algebra_consumes(self) -> None:
        result = optimize(items=(_item("a", 1), _item("b", 1)), slots=SLOTS, constraints=())
        pipeline = Pipeline("first", "plan", (
            Stage("select", {"predicate": Comparison(Path.parse("slot_index"), Op.EQ, Scalar(Q, 0))}),
        ))
        outcome = run_pipelines((pipeline,), {"plan": result.assignment})["first"]
        assert isinstance(outcome, Succeeded)

    def test_a_generated_plan_is_SIMULATED_not_an_official_record(self) -> None:
        """A plan is a proposal about a future that has not happened. Anything
        computed from it must be visibly weaker than anything computed from a
        transcript, and the basis ordering does that automatically."""
        result = optimize(items=(_item("a", 1),), slots=SLOTS, constraints=())
        assert all(r.basis is Basis.SIMULATED for r in result.assignment.records)

    def test_the_search_reports_whether_it_proved_optimality(self) -> None:
        """Honesty about the objective. A bounded search that found *a* plan has
        not shown there is no better one, and saying so is the difference
        between a heuristic and a lie."""
        result = optimize(items=(_item("a", 1),), slots=SLOTS, constraints=())
        assert isinstance(result.proven_optimal, bool)


class TestObjective:
    def test_minimising_slots_packs_rather_than_spreads(self) -> None:
        result = optimize(
            items=(_item("a", 1), _item("b", 1), _item("c", 1)),
            slots=SLOTS,
            constraints=(Capacity(attribute="weight", limit=5.0),),
            objective=Objective.MINIMIZE_SLOTS,
        )
        used = {r.fields["slot_index"].value for r in result.assignment.records}
        assert used == {0}

    def test_balancing_spreads_rather_than_packs(self) -> None:
        result = optimize(
            items=(_item("a", 1), _item("b", 1), _item("c", 1)),
            slots=SLOTS,
            constraints=(Capacity(attribute="weight", limit=5.0),),
            objective=Objective.BALANCE_LOAD,
        )
        used = {r.fields["slot_index"].value for r in result.assignment.records}
        assert len(used) == 3


class TestDeterminism:
    def test_the_same_problem_yields_the_same_plan_every_time(self) -> None:
        """A planner that returns a different answer to the same question each
        time is unusable regardless of how good any single answer is."""
        problem = dict(
            items=(_item("a", 2), _item("b", 2), _item("c", 2)),
            slots=SLOTS,
            constraints=(Capacity(attribute="weight", limit=4.0),),
        )
        first = [(r.fields["item"].value, r.fields["slot"].value) for r in optimize(**problem).assignment.records]
        for _ in range(10):
            again = [(r.fields["item"].value, r.fields["slot"].value) for r in optimize(**problem).assignment.records]
            assert again == first


class TestBudget:
    def test_exhausting_the_node_budget_reports_incompleteness(self) -> None:
        result = optimize(
            items=tuple(_item(f"i{n}", 1) for n in range(12)),
            slots=SLOTS,
            constraints=(Capacity(attribute="weight", limit=1.0),),
            node_budget=5,
        )
        assert isinstance(result, (Infeasible, DataDefect)) or result.assignment.completeness.complete is False


class TestFillObjective:
    """`fill` -- fixed slots, overflow allowed. The other planning question:
    "what goes in my next two semesters?", where most items will NOT fit and
    that is the answer, not a failure. The complete-assignment objectives
    refused that outright ("no assignment satisfies all constraints"), which is
    why a two-semester plan could never be produced."""

    def test_it_fills_fixed_slots_and_leaves_the_overflow_unscheduled(self) -> None:
        from app.agent_core.facts.optimize import UNSCHEDULED

        # Two slots of capacity 4; six items of weight 2 => 2 fit per slot, 2 spill.
        result = optimize(
            items=tuple(_item(chr(ord("a") + n), 2) for n in range(6)),
            slots=(_slot("winter", 0), _slot("spring", 1)),
            constraints=(Capacity(attribute="weight", limit=4),),
            objective=Objective.FILL,
        )
        assert not isinstance(result, Infeasible), "fill must never refuse for overflow"
        by_slot: dict[str, list[str]] = {}
        for record in result.assignment.records:
            by_slot.setdefault(record.fields["slot"].value, []).append(record.fields["item"].value)
        assert len(by_slot["winter"]) == 2
        assert len(by_slot["spring"]) == 2
        assert len(by_slot[UNSCHEDULED]) == 2, "the overflow is reported, not dropped"

    def test_no_slot_is_loaded_past_its_capacity(self) -> None:
        """The load-bearing property: a filled semester never exceeds the cap."""
        from app.agent_core.facts.optimize import UNSCHEDULED

        result = optimize(
            items=tuple(_item(f"c{n}", 3.5) for n in range(20)),
            slots=(_slot("winter", 0), _slot("spring", 1)),
            constraints=(Capacity(attribute="weight", limit=20),),
            objective=Objective.FILL,
        )
        load: dict[str, float] = {}
        for record in result.assignment.records:
            slot = record.fields["slot"].value
            if slot != UNSCHEDULED:
                load[slot] = load.get(slot, 0) + 3.5
        assert all(total <= 20.0 for total in load.values()), load

    def test_every_item_is_accounted_for(self) -> None:
        """Placed or explicitly unscheduled -- nothing vanishes, so the plan is
        complete even though not everything was scheduled."""
        items = tuple(_item(f"c{n}", 2) for n in range(10))
        result = optimize(
            items=items, slots=(_slot("w", 0),),
            constraints=(Capacity(attribute="weight", limit=4),),
            objective=Objective.FILL,
        )
        assert len(result.assignment.records) == len(items)
        assert result.assignment.completeness.complete

    def test_priority_is_the_item_order(self) -> None:
        """The caller controls what gets in by ORDERING items -- mandatory first.
        Greedy first-fit places earlier items before it runs out of room."""
        from app.agent_core.facts.optimize import UNSCHEDULED

        result = optimize(
            items=(_item("must_have", 4), _item("nice_to_have", 4)),
            slots=(_slot("w", 0),),
            constraints=(Capacity(attribute="weight", limit=4),),
            objective=Objective.FILL,
        )
        placed = {r.fields["item"].value: r.fields["slot"].value for r in result.assignment.records}
        assert placed["must_have"] == "w"
        assert placed["nice_to_have"] == UNSCHEDULED

    def test_a_plan_is_simulated(self) -> None:
        result = optimize(
            items=(_item("a", 1),), slots=(_slot("w", 0),), objective=Objective.FILL
        )
        assert all(r.basis is Basis.SIMULATED for r in result.assignment.records)


class TestPlacementCarriesAttributes:
    """A placed row keeps the item's own attributes, not just its id.

    The caller's next move after a plan is almost always to read those
    attributes back -- split by `slot`, total `credits` per semester, compute a
    per-course figure from `credits`. Dropping them forced a re-join the caller
    kept getting wrong on the last mile of a plan, so they ride along.
    """

    def _items(self):
        return (
            Item(id="0960327", attributes={
                "courseNumber": Scalar(I, "0960327"), "title": Scalar(T, "Nonlinear OR"),
                "credits": Scalar(Q, 3.5), "type": Scalar(T, "elective"),
            }),
            Item(id="0940314", attributes={
                "courseNumber": Scalar(I, "0940314"), "title": Scalar(T, "Stochastic OR"),
                "credits": Scalar(Q, 3.0), "type": Scalar(T, "required"),
            }),
        )

    def test_fill_carries_attributes_onto_placed_and_unscheduled_rows(self) -> None:
        slots = (Slot(id="winter", index=0), Slot(id="spring", index=1))
        plan = optimize(
            items=self._items(), slots=slots,
            constraints=(Capacity("credits", 3.5),), objective=Objective.FILL,
        )
        by_item = {r.fields["item"].value: r for r in plan.assignment.records}
        assert by_item["0960327"].fields["title"].value == "Nonlinear OR"
        assert by_item["0960327"].fields["type"].value == "elective"
        assert by_item["0960327"].fields["credits"].value == 3.5
        # every placed row also keeps the structural keys
        assert by_item["0960327"].fields["slot"].value in {"winter", "spring", "(unscheduled)"}

    def test_minimize_slots_carries_attributes_too(self) -> None:
        slots = (Slot(id="s1", index=0), Slot(id="s2", index=1), Slot(id="s3", index=2))
        plan = optimize(items=self._items(), slots=slots, objective=Objective.MINIMIZE_SLOTS)
        row = next(r for r in plan.assignment.records if r.fields["item"].value == "0940314")
        assert row.fields["title"].value == "Stochastic OR"
        assert row.fields["type"].value == "required"

    def test_structural_keys_win_a_name_clash(self) -> None:
        """An item attribute literally named `slot` must not shadow where it
        landed -- the placement's own slot has to win."""
        items = (Item(id="x", attributes={"slot": Scalar(I, "not-a-real-slot"), "credits": Scalar(Q, 1.0)}),)
        plan = optimize(items=items, slots=(Slot(id="real", index=0),), objective=Objective.FILL)
        assert plan.assignment.records[0].fields["slot"].value == "real"
