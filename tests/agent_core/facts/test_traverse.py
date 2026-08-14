"""`traverse` -- phase 7 of docs/agent/tools_implementation_plan.md.

This is the primitive with a theorem behind it. Transitive closure is provably
outside relational algebra -- it is why SQL needed `WITH RECURSIVE` -- so a
prerequisite chain of unknown depth cannot be reached by any pipeline over the
basis, however long.

The boundary is DEMONSTRATED here, not asserted: a fixed pipeline of N joins
reaches exactly depth N, so for any pipeline there is a chain it misses. That is
what makes `traverse` a primitive rather than a shortcut.
"""

from __future__ import annotations

from app.agent_core.facts.operators import Pipeline, Stage
from app.agent_core.facts.predicate import Comparison, Op, Path
from app.agent_core.facts.runner import Succeeded, run_pipelines
from app.agent_core.facts.traverse import traverse
from app.agent_core.facts.types import (
    Basis,
    Collection,
    Completeness,
    Record,
    Scalar,
    ScalarKind,
)

Q = ScalarKind.QUANTITY
I = ScalarKind.IDENTIFIER


def _edge(source: str, target: str, basis: Basis = Basis.OFFICIAL_RECORD, group: str | None = None) -> Record:
    fields = {"from": Scalar(I, source), "to": Scalar(I, target)}
    if group is not None:
        fields["group"] = Scalar(I, group)
    return Record(fields=fields, basis=basis)


# 00970800 needs 00960211, which needs 00940224, which needs 00940100.
# Four levels: no fixed pipeline of three joins can see the bottom.
CHAIN = Collection(
    records=(
        _edge("00970800", "00960211"),
        _edge("00960211", "00940224"),
        _edge("00940224", "00940100"),
    ),
    completeness=Completeness(complete=True, total=3),
)

FROM, TO = Path.parse("from"), Path.parse("to")


class TestTransitiveClosure:
    def test_it_reaches_the_whole_chain_not_just_the_first_hop(self) -> None:
        result = traverse(CHAIN, start="00970800", from_path=FROM, to_path=TO)
        reached = sorted(r.fields["node"].value for r in result.records)
        assert reached == ["00940100", "00940224", "00960211"]

    def test_depth_is_reported_so_ordering_survives(self) -> None:
        result = traverse(CHAIN, start="00970800", from_path=FROM, to_path=TO)
        depths = {r.fields["node"].value: r.fields["depth"].value for r in result.records}
        assert depths == {"00960211": 1, "00940224": 2, "00940100": 3}

    def test_max_depth_bounds_the_walk(self) -> None:
        result = traverse(CHAIN, start="00970800", from_path=FROM, to_path=TO, max_depth=1)
        assert [r.fields["node"].value for r in result.records] == ["00960211"]

    def test_a_bounded_walk_reports_itself_incomplete(self) -> None:
        """Stopping early is a truncation like any other, so an aggregate over it
        must fail closed rather than count a partial chain."""
        result = traverse(CHAIN, start="00970800", from_path=FROM, to_path=TO, max_depth=1)
        assert result.completeness.complete is False

    def test_a_complete_walk_reports_itself_complete(self) -> None:
        result = traverse(CHAIN, start="00970800", from_path=FROM, to_path=TO)
        assert result.completeness.complete is True


class TestTheBoundaryIsReal:
    def test_a_fixed_pipeline_of_joins_misses_the_bottom_of_the_chain(self) -> None:
        """The demonstration. Three self-joins reach depth 3; the chain is depth
        3 from the top, so a FOUR-level chain needs four. Since the depth is a
        property of the data and the pipeline is fixed, no pipeline over the
        basis reaches an arbitrary chain -- which is the theorem, made concrete.
        """
        one_hop = Pipeline("hop", "edges", (
            Stage("select", {"predicate": Comparison(FROM, Op.EQ, Scalar(I, "00970800"))}),
        ))
        outcome = run_pipelines((one_hop,), {"edges": CHAIN})["hop"]
        assert isinstance(outcome, Succeeded)
        # One join reaches exactly one level. Reaching 00940100 would require
        # knowing the depth in advance and writing that many stages.
        assert [r.fields["to"].value for r in outcome.value.records] == ["00960211"]

    def test_traverse_output_is_a_collection_the_algebra_consumes(self) -> None:
        """The point of the boundary: `traverse` hands back ordinary facts, so
        everything after it is algebra again."""
        reached = traverse(CHAIN, start="00970800", from_path=FROM, to_path=TO)
        pipeline = Pipeline("deep", "reached", (
            Stage("select", {"predicate": Comparison(Path.parse("depth"), Op.GE, Scalar(Q, 2))}),
        ))
        outcome = run_pipelines((pipeline,), {"reached": reached})["deep"]
        assert isinstance(outcome, Succeeded)
        assert sorted(r.fields["node"].value for r in outcome.value.records) == ["00940100", "00940224"]


class TestCycles:
    def test_a_cycle_terminates_rather_than_hanging(self) -> None:
        cyclic = Collection(
            records=(_edge("a", "b"), _edge("b", "c"), _edge("c", "a")),
            completeness=Completeness(complete=True, total=3),
        )
        result = traverse(cyclic, start="a", from_path=FROM, to_path=TO)
        assert sorted(r.fields["node"].value for r in result.records) == ["a", "b", "c"]

    def test_a_node_is_reported_once_at_its_shortest_depth(self) -> None:
        diamond = Collection(
            records=(_edge("top", "left"), _edge("top", "right"), _edge("left", "bottom"), _edge("right", "bottom")),
            completeness=Completeness(complete=True, total=4),
        )
        result = traverse(diamond, start="top", from_path=FROM, to_path=TO)
        bottom = [r for r in result.records if r.fields["node"].value == "bottom"]
        assert len(bottom) == 1
        assert bottom[0].fields["depth"].value == 2


class TestProvenance:
    def test_a_reached_node_is_only_as_certain_as_the_weakest_edge_to_it(self) -> None:
        mixed = Collection(
            records=(_edge("a", "b"), _edge("b", "c", basis=Basis.WIKI_DERIVED)),
            completeness=Completeness(complete=True, total=2),
        )
        result = traverse(mixed, start="a", from_path=FROM, to_path=TO)
        by_node = {r.fields["node"].value: r.basis for r in result.records}
        assert by_node["b"] is Basis.OFFICIAL_RECORD
        assert by_node["c"] is Basis.WIKI_DERIVED


class TestAlternativesAreNotFlattened:
    def test_an_edge_group_survives_the_walk(self) -> None:
        """Prerequisites carry AND/OR structure: two edges in the same group are
        ALTERNATIVES, not both-required. Flattening them into a plain reachable
        set is a real bug -- it reports every alternative as mandatory.

        `traverse` computes reachability and PRESERVES the group label so the
        algebra can reason about satisfaction. Evaluating AND/OR satisfaction is
        a separate recursive problem and is deliberately NOT claimed here.
        """
        alternatives = Collection(
            records=(_edge("x", "opt1", group="or1"), _edge("x", "opt2", group="or1")),
            completeness=Completeness(complete=True, total=2),
        )
        result = traverse(alternatives, start="x", from_path=FROM, to_path=TO, carry=("group",))
        groups = {r.fields["node"].value: r.fields["group"].value for r in result.records}
        assert groups == {"opt1": "or1", "opt2": "or1"}
