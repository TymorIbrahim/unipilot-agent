"""Reachability -- is every advertised tool actually usable?

The check that was missing. Every primitive had passing unit tests and the
catalog described all eight, yet only three could be used against the live
context: two crashed on a `None` dependency and three had no obtainable inputs.
Nothing noticed, because each tool was correct in isolation and no test asked
whether anything could FEED it.

A capability is not available because it is implemented. It is available when
something can supply its inputs, so that is what this asserts.

Connects through the agent's own settings rather than a bespoke env var -- an
earlier version read `MONGODB_URI`, which this project does not set, so it
skipped everywhere while appearing to pass. It also imports
`_fresh_mongo_client_per_test`, without which a memoized motor client from a
previous test's closed event loop raises `RuntimeError` here and the whole file
skips for a reason that has nothing to do with reachability.

Both of those made this file report "not verified" while the database was
sitting right there. A skip is not a neutral outcome -- it is a claim that
something could not be checked, and a wrong one is indistinguishable from a
pass.
"""

from __future__ import annotations

import pytest

from app.agent_core.facts.answer import HeldFact
from app.agent_core.facts.catalog import PRIMITIVES, available_tools
from app.agent_core.facts.dispatch import DispatchContext, dispatch
from app.agent_core.facts.codec import parse_pipelines
from app.agent_core.facts.operators import OPERATORS
from app.agent_core.facts.runner import Failed, Succeeded, run_pipelines
from app.agent_core.facts.sources import REGISTRY
from app.agent_core.facts.types import (
    Basis,
    Collection,
    Completeness,
    Record,
    Scalar,
    ScalarKind,
)
from app.agent_core.facts.wiring import (
    build_context,
    build_wiring,
    obtainable_from,
    prerequisite_edges_source,
)
from app.db.mongo import get_database
from tests.agent_core.ise_student_fixture import (  # noqa: F401 -- autouse fixture injection
    _fresh_mongo_client_per_test,
)


@pytest.fixture
async def database():
    try:
        db = await get_database()
        await db.command("ping")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"NOT VERIFIED: no database ({type(exc).__name__}). Tool reachability UNCHECKED.")
    return db


class TestCatalogHonesty:
    def test_a_tool_is_advertised_only_when_its_dependency_is_wired(self) -> None:
        bare = DispatchContext(schemas=REGISTRY)
        advertised = {spec.name for spec in available_tools(bare)}
        assert "search_corpus" not in advertised, "a corpus tool with no corpus must not be offered"
        assert "interpret" not in advertised

    def test_wiring_the_dependency_makes_the_tool_appear(self) -> None:
        wired = DispatchContext(schemas=REGISTRY, retriever=object(), extractor=object())
        advertised = {spec.name for spec in available_tools(wired)}
        assert {"search_corpus", "interpret"} <= advertised

    def test_no_tool_is_permanently_unadvertisable(self) -> None:
        """Every primitive must be reachable under SOME wiring, or it is dead
        code being carried in the prompt."""
        schemas = {**REGISTRY, "prerequisite_edges": prerequisite_edges_source(_EmptyEngine())}
        everything = DispatchContext(
            schemas=schemas,
            retriever=object(),
            extractor=object(),
            database=object(),
            # Through the production function, not a hand-written set. A literal
            # here would pass while the real `build_context` advertised something
            # different, which is the drift this whole file exists to catch.
            obtainable=obtainable_from(schemas),
        )
        assert {spec.name for spec in available_tools(everything)} == {spec.name for spec in PRIMITIVES}

    def test_obtainability_is_read_off_the_sources_not_declared_beside_them(self) -> None:
        """Drop the source, lose the tool. The two must move together."""
        without_plans = {n: s for n, s in REGISTRY.items() if n != "semester_plans"}
        assert "slots" in obtainable_from(REGISTRY)
        assert "slots" not in obtainable_from(without_plans)


class TestEveryPromptedCapabilityExists:
    """The prompt advertises more than tools, and the extra was where the gap was.

    `catalog.render_catalog` renders the OPERATOR TABLE into the system prompt
    straight from `OPERATORS`. `unnest` sat in that table -- documented, type
    -checked, taught to the model -- with no branch in the runner, so any model
    that used the operator the prompt offered got "operator 'unnest' has no
    evaluation rule". Eight tools were being checked for reachability while
    fourteen operators were not being checked at all.
    """

    @pytest.mark.parametrize("operator", sorted(OPERATORS))
    def test_every_declared_operator_parses_and_runs_from_json(self, operator: str) -> None:
        """Through the CODEC, because that is the path a model actually takes.

        An earlier version built `Stage` objects directly and asserted only that
        the runner had a branch. It passed for `group` while the codec was
        filing the aggregate spec under the wrong key -- so `group` parsed, ran,
        and returned the group keys with NO aggregated column, silently. Testing
        the runner alone tests half the path.
        """
        source = Collection(
            records=(
                Record(
                    fields={
                        "id": Scalar(ScalarKind.IDENTIFIER, "a"),
                        "n": Scalar(ScalarKind.QUANTITY, 2),
                        "items": Collection(
                            records=(Record(fields={"leaf": Scalar(ScalarKind.QUANTITY, 1)}, basis=Basis.OFFICIAL_RECORD),),
                            completeness=Completeness(complete=True, total=1),
                        ),
                    },
                    basis=Basis.OFFICIAL_RECORD,
                ),
            ),
            completeness=Completeness(complete=True, total=1),
        )
        env = {"src": source, "other": source, "scalar": Scalar(ScalarKind.QUANTITY, 1)}
        payload = [{"name": "out", **_MINIMAL_USE[operator]}]

        pipelines = parse_pipelines(payload)  # a ParseError here is a codec gap
        outcome = run_pipelines(pipelines, env)["out"]

        assert isinstance(outcome, Succeeded), (
            f"'{operator}' is in OPERATORS, so `render_catalog` teaches it to the model, but the "
            f"smallest legal use of it does not work: "
            f"{getattr(outcome, 'defect', outcome)}"
        )


_MINIMAL_USE: dict[str, dict] = {
    "select": {"source": "src", "stages": [{"op": "select", "predicate": {"path": "n", "op": ">", "value": 1}}]},
    "project": {"source": "src", "stages": [{"op": "project", "fields": {"id": "id"}}]},
    "extend": {"source": "src", "stages": [{"op": "extend", "fields": {"double": {"add": [{"path": "n"}, {"path": "n"}]}}}]},
    "join": {"source": "src", "stages": [{"op": "join", "other": "other", "predicate": {"path": "left.id", "op": "=", "value": "a"}}]},
    "union": {"source": "src", "stages": [{"op": "union", "other": "other"}]},
    "difference": {"source": "src", "stages": [{"op": "difference", "other": "other", "on": "id"}]},
    "distinct": {"source": "src", "stages": [{"op": "distinct"}]},
    "unnest": {"source": "src", "stages": [{"op": "unnest", "field": "items"}]},
    "group": {"source": "src", "stages": [{"op": "group", "by": ["id"], "agg": {"how_many": {"agg": "count"}}}]},
    "aggregate": {"source": "src", "stages": [{"op": "aggregate", "agg": "count"}]},
    "sort": {"source": "src", "stages": [{"op": "sort", "field": "n", "dir": "desc"}]},
    "limit": {"source": "src", "stages": [{"op": "limit", "n": 1}]},
    "arith": {"source": "scalar", "stages": [{"op": "arith", "fn": "add", "other": "scalar"}]},
    "compare": {"source": "scalar", "stages": [{"op": "compare", "fn": ">=", "other": "scalar"}]},
}
"""One smallest legal use of each operator, written the way a MODEL writes it.

Hand-written on purpose: a generated call would have to know each operator's
arguments, which is the very knowledge being tested. Missing an entry fails the
test below rather than skipping it, so adding an operator forces adding a use.
"""


class TestGroupAggregates:
    """The silent one. `group` parsed, ran, and returned bare keys."""

    def test_it_actually_produces_the_aggregated_column(self) -> None:
        records = tuple(
            Record(
                fields={"k": Scalar(ScalarKind.IDENTIFIER, key), "n": Scalar(ScalarKind.QUANTITY, value)},
                basis=Basis.OFFICIAL_RECORD,
            )
            for key, value in (("a", 2), ("a", 3), ("b", 9))
        )
        env = {"src": Collection(records=records, completeness=Completeness(complete=True, total=3))}
        pipelines = parse_pipelines([{
            "name": "g", "source": "src",
            "stages": [{"op": "group", "by": ["k"], "agg": {"total": {"agg": "sum", "field": "n"}}}],
        }])

        rows = {r.fields["k"].value: r.fields["total"].value for r in run_pipelines(pipelines, env)["g"].value.records}

        assert rows == {"a": 5, "b": 9}

    def test_a_group_with_no_aggregate_is_refused_rather_than_returning_bare_keys(self) -> None:
        env = {"src": Collection(
            records=(Record(fields={"k": Scalar(ScalarKind.IDENTIFIER, "a")}, basis=Basis.OFFICIAL_RECORD),),
            completeness=Completeness(complete=True, total=1),
        )}
        pipelines = parse_pipelines([{"name": "g", "source": "src", "stages": [{"op": "group", "by": ["k"]}]}])
        outcome = run_pipelines(pipelines, env)["g"]
        assert isinstance(outcome, Failed)
        assert "distinct" in outcome.defect.message


class TestOperatorUsesAreExhaustive:
    def test_every_operator_has_a_minimal_use(self) -> None:
        assert set(_MINIMAL_USE) == set(OPERATORS), (
            "an operator was added or removed without updating _MINIMAL_USE, so it would go "
            f"unchecked: {set(OPERATORS) ^ set(_MINIMAL_USE)}"
        )


class _EmptyEngine:
    """A built graph with no courses in it -- enough to construct the source."""

    _built = True

    class graph:  # noqa: N801 -- mimics the engine's attribute, not a real class
        nodes: dict = {}


class TestUnwiredToolsFailSoftly:
    """They used to crash. A missing capability must be a defect the loop can
    report, not an exception that ends the turn."""

    @pytest.mark.parametrize(
        ("call", "expect"),
        [
            ({"tool": "search_corpus", "as": "p", "args": {"query": "x"}}, "no corpus"),
            ({"tool": "interpret", "as": "v", "args": {"slug": "s", "expect": "quantity"}}, "no interpreter"),
        ],
    )
    async def test_it_reports_rather_than_raises(self, call, expect) -> None:
        result = await dispatch(call, DispatchContext(schemas=REGISTRY))
        assert expect in next(iter(result.defects.values())).message


class TestEveryAdvertisedToolCanBeFed:
    """The heart of it: obtain each tool's inputs from the real registry and
    wiring, call it, and require a fact back."""

    async def test_find_reads_every_registered_source(self, database) -> None:
        context = DispatchContext(database=database, schemas=REGISTRY)
        unusable = []
        for name in REGISTRY:
            result = await dispatch(
                {"tool": "find", "as": "probe", "args": {"source": name, "limit": 2}}, context
            )
            if not result.facts:
                unusable.append(f"{name}: {next(iter(result.defects.values())).message[:90]}")
        assert not unusable, "registered sources that cannot be read: " + "; ".join(unusable)

    async def test_compute_derives_from_a_fetched_fact(self, database) -> None:
        """Note the FILTER. An unfiltered `limit: 5` over 2,613 courses is
        incomplete, and counting it fails closed -- correctly. The first version
        of this test did exactly that and read as a reachability failure when it
        was the completeness rule doing its job."""
        context = DispatchContext(database=database, schemas=REGISTRY)
        fetched = await dispatch(
            {"tool": "find", "as": "one_course", "args": {
                "source": "courses",
                "predicate": {"path": "courseNumber", "op": "=", "value": "00960211"},
                "limit": 50,
            }},
            context,
        )
        context.facts.update(fetched.facts)
        assert fetched.facts["one_course"].value.completeness.complete, "a filtered fetch should be complete"

        result = await dispatch(
            {"tool": "compute", "args": {"pipelines": [
                {"name": "n", "source": "one_course", "stages": [{"op": "aggregate", "agg": "count"}]}
            ]}},
            context,
        )
        assert result.facts["n"].value.value >= 1

    async def test_counting_an_incomplete_fetch_still_refuses(self, database) -> None:
        """The other half, asserted so the fix above cannot quietly become a
        loosening of the rule."""
        context = DispatchContext(database=database, schemas=REGISTRY)
        page = await dispatch({"tool": "find", "as": "page", "args": {"source": "courses", "limit": 5}}, context)
        context.facts.update(page.facts)
        assert page.facts["page"].value.completeness.complete is False

        result = await dispatch(
            {"tool": "compute", "args": {"pipelines": [
                {"name": "n", "source": "page", "stages": [{"op": "aggregate", "agg": "count"}]}
            ]}},
            context,
        )
        assert "n" not in result.facts and result.defects

    async def test_forecast_can_be_fed_from_course_offerings(self, database) -> None:
        """The source that was missing from the registry entirely -- 6,580
        documents of exactly the history `forecast` needs, which is why offering
        questions were unanswerable."""
        context = DispatchContext(database=database, schemas=REGISTRY)
        fetched = await dispatch(
            {"tool": "find", "as": "offerings", "args": {
                "source": "course_offerings",
                "predicate": {"path": "courseNumber", "op": "=", "value": "00960211"},
                "limit": 100,
            }},
            context,
        )
        context.facts.update(fetched.facts)
        assert fetched.facts["offerings"].value.records, "no offering history for a known course"

        result = await dispatch(
            {"tool": "forecast", "as": "summer", "args": {
                "observations": "offerings", "period_path": "semesterName", "target": "summer",
            }},
            context,
        )
        assert "summer" in result.facts, next(iter(result.defects.values())).message

    async def test_no_advertised_tool_needs_an_input_the_model_cannot_obtain(self, database) -> None:
        """The check that should have existed the first time.

        `traverse` was advertised while no `find` source yielded edges. The
        previous version of this test seeded `context.facts["edges"]` by hand
        and passed -- verifying reachability from TEST CODE, which is not the
        boundary that matters. A tool is reachable when a MODEL can obtain its
        inputs by calling tools.

        Runs against `build_context`, the same assembly production uses, so a
        context that is wired differently there cannot pass here.
        """
        context = build_context(database)
        unfeedable = [
            spec.name
            for spec in available_tools(context)
            if spec.needs_source and spec.needs_source not in context.obtainable
        ]
        assert not unfeedable, (
            f"advertised but unfeedable: {unfeedable}. Either make the input obtainable "
            "through a tool the model can call, or stop advertising the tool."
        )

    async def test_traverse_walks_prerequisites_fetched_through_find(self, database) -> None:
        """The whole route, taken the way a model takes it.

        No seeded facts: `find` on a registered source produces the edges, and
        `traverse` consumes them. The earlier version of this test built the
        edge collection itself and passed while the model had no route at all.
        """
        context = build_context(database)
        if "prerequisite_edges" not in context.schemas:
            pytest.skip("NOT VERIFIED: no graph engine, so prerequisite edges cannot be derived")

        fetched = await dispatch(
            {"tool": "find", "as": "edges", "args": {"source": "prerequisite_edges", "limit": 5000}},
            context,
        )
        assert fetched.facts, next(iter(fetched.defects.values())).message
        edges = fetched.facts["edges"].value
        assert edges.records, "the prerequisite graph produced no edges"
        context.facts.update(fetched.facts)

        # A course whose prerequisites themselves have prerequisites, so the
        # walk has more than one level to prove.
        depth_one = {r.fields["requires"].value for r in edges.records}
        start = next(
            r.fields["course"].value
            for r in edges.records
            if r.fields["requires"].value in depth_one
            and any(e.fields["course"].value == r.fields["requires"].value for e in edges.records)
        )

        walked = await dispatch(
            {"tool": "traverse", "as": "chain", "args": {
                "edges": "edges", "start": start, "from": "course", "to": "requires",
                "carry": ["group"],
            }},
            context,
        )
        assert walked.facts, next(iter(walked.defects.values())).message
        reached = walked.facts["chain"].value
        assert max(r.fields["depth"].value for r in reached.records) > 1, (
            "a chain of depth 1 proves nothing `join` could not already do -- transitive closure "
            "is the only reason this primitive exists"
        )
        # Parsed from catalog prose, so weaker than a record, and it must say so.
        assert walked.facts["chain"].basis is Basis.WIKI_DERIVED

    async def test_prerequisite_alternatives_keep_their_group(self, database) -> None:
        """"A or B" is ONE obligation. Flattened into two edges it reads as two,
        and anything counting missing prerequisites double-counts a choice."""
        context = build_context(database)
        if "prerequisite_edges" not in context.schemas:
            pytest.skip("NOT VERIFIED: no graph engine")

        fetched = await dispatch(
            {"tool": "find", "as": "edges", "args": {"source": "prerequisite_edges", "limit": 5000}},
            context,
        )
        by_group: dict[str, set[str]] = {}
        for record in fetched.facts["edges"].value.records:
            by_group.setdefault(record.fields["group"].value, set()).add(record.fields["requires"].value)

        assert any(len(targets) > 1 for targets in by_group.values()), (
            "no group holds alternatives, so either the catalog has no OR-prerequisites "
            "or the AND/OR structure is being flattened again"
        )

    async def test_optimize_places_courses_into_semesters_fetched_through_find(self, database) -> None:
        """`optimize`'s route: find a plan, unnest it twice, place the result.

        Slots are not stored as a collection anywhere -- they are the nested
        `semesters[]` of a plan -- so this is the step that proves `find` +
        `unnest` really is a route to them and not just a claim in `yields`.
        """
        context = build_context(database)
        plan = await database["semester_plans"].find_one({"semesters.0.plannedCourses.0": {"$exists": True}})
        if plan is None:
            pytest.skip("NOT VERIFIED: no stored plan has a semester holding courses")

        fetched = await dispatch(
            {"tool": "find", "as": "plan", "args": {
                "source": "semester_plans",
                "predicate": {"path": "_id", "op": "=", "value": str(plan["_id"])},
                "limit": 5,
            }},
            context,
        )
        assert fetched.facts, next(iter(fetched.defects.values())).message
        context.facts.update(fetched.facts)

        derived = await dispatch(
            {"tool": "compute", "args": {"pipelines": [
                {"name": "slots", "source": "plan", "stages": [{"op": "unnest", "field": "semesters"}]},
                {"name": "items", "source": "slots", "stages": [{"op": "unnest", "field": "plannedCourses"}]},
            ]}},
            context,
        )
        assert not derived.defects, {n: d.message for n, d in derived.defects.items()}
        context.facts.update(derived.facts)
        assert derived.facts["slots"].value.records, "unnesting a plan yielded no semesters"

        placed = await dispatch(
            {"tool": "optimize", "as": "schedule", "args": {
                "items": "items", "item_id": "courseNumber",
                "slots": "slots", "slot_id": "semesterCode", "slot_index": "order",
                "constraints": [{"kind": "capacity", "attribute": "credits", "limit": 40}],
                "objective": "minimize_slots",
            }},
            context,
        )
        assert placed.facts, next(iter(placed.defects.values())).message
        # A plan is about a future that has not happened.
        assert placed.facts["schedule"].basis is Basis.SIMULATED

    async def test_optimize_refuses_to_guess_what_identifies_a_slot(self, database) -> None:
        """The bug this argument exists for: unnested slots all carry the parent
        plan's `_id`, so an identity GUESS pooled every semester into one slot
        with room for everything."""
        context = build_context(database)
        context.facts["c"] = HeldFact(
            value=Collection(
                records=(Record(fields={"x": Scalar(ScalarKind.IDENTIFIER, "1")}, basis=Basis.OFFICIAL_RECORD),),
                completeness=Completeness(complete=True, total=1),
            ),
            basis=Basis.OFFICIAL_RECORD,
        )
        result = await dispatch(
            {"tool": "optimize", "as": "p", "args": {"items": "c", "slots": "c", "item_id": "x"}},
            context,
        )
        assert "slot_id" in next(iter(result.defects.values())).message

    async def test_search_corpus_returns_passages_when_wired(self, database) -> None:
        wiring = build_wiring()
        if "retriever" not in wiring:
            pytest.skip("NOT VERIFIED: no corpus configured")

        context = DispatchContext(database=database, schemas=REGISTRY, retriever=wiring["retriever"])
        result = await dispatch(
            {"tool": "search_corpus", "as": "hits", "args": {"query": "appealing a grade", "limit": 3}},
            context,
        )
        assert result.facts["hits"].value.records, "the corpus returned nothing for a policy query"
        # And the passages are remembered, so `interpret` can be handed a slug
        # rather than retyped prose.
        assert context.passages

    async def test_propose_builds_from_a_real_fetched_fact(self, database) -> None:
        context = DispatchContext(database=database, schemas=REGISTRY)
        fetched = await dispatch(
            {"tool": "find", "as": "courses", "args": {"source": "courses", "limit": 1}}, context
        )
        context.facts.update(fetched.facts)
        result = await dispatch(
            {"tool": "propose", "as": "p", "args": {
                "action": "register", "target": "00960211", "grounds": ["courses"],
            }},
            context,
        )
        assert result.proposal is not None
        assert result.facts == {}, "a proposal is not a fact -- nothing has happened yet"


# `optimize` is the one tool nothing can currently feed: it needs a collection of
# future semesters, and no source stores them. Deliberately NOT asserted -- the
# first version of this file tried, with a "does any schema have a semester-ish
# field" heuristic that matched three unrelated sources. A heuristic dressed as
# an assertion is worse than a note, because it fails for reasons nobody trusts.


class TestOptimizeDedupsItems:
    """A course reached through two offerings (winter AND spring) is one course
    to place. The natural join-to-offerings yields it twice; optimize collapses
    duplicates by item_id rather than refusing, which was the last-mile trip on
    a two-semester plan."""

    async def test_duplicate_item_ids_collapse_to_one_placement(self, database) -> None:
        from app.agent_core.facts.answer import HeldFact
        from app.agent_core.facts.types import Basis, Collection, Completeness, Record, Scalar, ScalarKind

        def coll(*rows):
            recs = tuple(
                Record(
                    fields={
                        k: (Scalar(ScalarKind.QUANTITY, v) if isinstance(v, (int, float)) else Scalar(ScalarKind.IDENTIFIER, v))
                        for k, v in row.items()
                    },
                    basis=Basis.OFFICIAL_RECORD,
                )
                for row in rows
            )
            return Collection(records=recs, completeness=Completeness(complete=True, total=len(recs)))

        context = DispatchContext(database=database, schemas=REGISTRY)
        context.facts["items"] = HeldFact(
            value=coll({"courseNumber": "A", "credits": 4}, {"courseNumber": "A", "credits": 4}, {"courseNumber": "B", "credits": 3}),
            basis=Basis.OFFICIAL_RECORD,
        )
        context.facts["slots"] = HeldFact(value=coll({"semesterName": "winter"}, {"semesterName": "spring"}), basis=Basis.OFFICIAL_RECORD)
        result = await dispatch(
            {"tool": "optimize", "as": "plan", "args": {
                "items": "items", "item_id": "courseNumber", "slots": "slots",
                "slot_id": "semesterName", "slot_index": "semesterName",
                "constraints": [{"kind": "capacity", "attribute": "credits", "limit": 20}], "objective": "fill",
            }},
            context,
        )
        assert result.facts, next(iter(result.defects.values())).message
        placed = [r.fields["item"].value for r in result.facts["plan"].value.records]
        assert placed.count("A") == 1, "the duplicated course must appear once"
        assert "B" in placed
