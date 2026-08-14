"""Phase 5 gate for docs/agent/tools_implementation_plan.md.

The stated gate: a truncated fetch reports `complete=false`, and an `aggregate`
over it fails closed.

`find` is where raw storage becomes typed facts, so it owns the decision phase 1
refused to make with a heuristic: whether "3.5" is a quantity and "00940224" is
an identifier. Both are strings in Mongo. Only a declared schema can tell them
apart, which is what "convert at admission, where the source schema is known"
means in practice.
"""

from __future__ import annotations

import os

import pytest
from motor.motor_asyncio import AsyncIOMotorClient

from tests.agent_core.ise_student_fixture import (  # noqa: F401 -- autouse fixture
    _fresh_mongo_client_per_test,
)
from app.agent_core.facts.find import SourceSchema, find
from app.agent_core.facts.operators import DataDefect, ExpressionDefect, Pipeline, Stage
from app.agent_core.facts.predicate import Comparison, Op, Path
from app.agent_core.facts.runner import Failed, Succeeded, run_pipelines
from app.agent_core.facts.types import Basis, Scalar, ScalarKind

Q = ScalarKind.QUANTITY
I = ScalarKind.IDENTIFIER

# Deliberately dirty, the way the real catalog is: credits arrive as STRINGS,
# course codes are numeric-looking strings with leading zeros, and one record is
# missing a field entirely.
DOCUMENTS = [
    {"courseNumber": "00940224", "credits": "3.5", "grade": 95, "track": "cs"},
    {"courseNumber": "00960211", "credits": "3.0", "grade": 60, "track": "cs"},
    {"courseNumber": "00970800", "credits": "4.0", "grade": 88, "track": "ee"},
    {"courseNumber": "00940594", "credits": "2.5", "track": "cs"},  # no grade
]

SCHEMA = SourceSchema(
    collection="courses",
    key="courseNumber",
    fields={"courseNumber": I, "credits": Q, "grade": Q, "track": I},
    basis=Basis.OFFICIAL_RECORD,
)


@pytest.fixture
async def database():
    # Through the agent's own settings, not a bespoke env var: an earlier
    # version read MONGODB_URI, which this project does not set, so these
    # skipped on every machine while looking like passes.
    from app.db.mongo import get_database

    try:
        database_handle = await get_database()
        await database_handle.command("ping")
        client = database_handle.client
    except Exception as exc:  # noqa: BLE001
        pytest.skip(
            f"NOT VERIFIED: no database ({type(exc).__name__}). "
            "`find`'s push-down, typing and completeness reporting are UNCHECKED in this run."
        )
    db = client["unipilot_find_phase5"]
    await db["courses"].delete_many({})
    await db["courses"].insert_many([dict(d) for d in DOCUMENTS])
    yield db
    await client.drop_database("unipilot_find_phase5")
    client.close()


class TestTyping:
    async def test_a_numeric_string_becomes_a_quantity_when_the_schema_says_so(self, database) -> None:
        """Phase 1 refuses `Scalar(QUANTITY, "3.5")` on purpose. This is where
        that conversion is allowed to happen, because here the kind is declared
        rather than guessed."""
        result = await find(database, SCHEMA)
        credits = result.records[0].fields["credits"]
        assert credits.kind is Q and credits.value == 3.5

    async def test_a_course_code_stays_an_identifier(self, database) -> None:
        """Same shape of string, opposite kind. No heuristic could separate
        these -- the leading zero is a convention, not a type."""
        code = (await find(database, SCHEMA)).records[0].fields["courseNumber"]
        assert code.kind is I and code.value == "00940224"

    async def test_an_absent_field_is_absent_rather_than_defaulted(self, database) -> None:
        """A record with no grade must not acquire one. Defaulting to 0 would
        make an average silently wrong."""
        result = await find(database, SCHEMA)
        missing_grade = [r for r in result.records if r.fields["courseNumber"].value == "00940594"][0]
        assert "grade" not in missing_grade.fields

    async def test_records_carry_the_declared_basis(self, database) -> None:
        result = await find(database, SCHEMA)
        assert all(r.basis is Basis.OFFICIAL_RECORD for r in result.records)


class TestPushDown:
    async def test_the_predicate_filters_at_the_source(self, database) -> None:
        result = await find(database, SCHEMA, predicate=Comparison(Path.parse("track"), Op.EQ, Scalar(I, "cs")))
        assert sorted(r.fields["courseNumber"].value for r in result.records) == [
            "00940224", "00940594", "00960211"
        ]

    async def test_an_unknown_field_is_rejected_naming_what_exists(self, database) -> None:
        result = await find(database, SCHEMA, predicate=Comparison(Path.parse("deficit"), Op.GT, Scalar(Q, 0)))
        assert isinstance(result, ExpressionDefect)
        assert "deficit" in result.message and "credits" in result.message


class TestCompleteness:
    async def test_an_untruncated_fetch_is_complete(self, database) -> None:
        result = await find(database, SCHEMA, limit=100)
        assert result.completeness.complete is True
        assert result.completeness.total == 4

    async def test_a_truncated_fetch_reports_the_true_total(self, database) -> None:
        """THE gate, first half."""
        result = await find(database, SCHEMA, limit=2)
        assert len(result.records) == 2
        assert result.completeness.complete is False
        assert result.completeness.total == 4

    async def test_completeness_is_measured_against_the_PREDICATE_not_the_collection(self, database) -> None:
        """A filtered fetch is complete when it has every MATCHING record, even
        though it holds fewer than the collection. Counting against the whole
        collection would mark every filtered result incomplete and make
        aggregates permanently impossible."""
        result = await find(database, SCHEMA, predicate=Comparison(Path.parse("track"), Op.EQ, Scalar(I, "ee")))
        assert result.completeness.complete is True
        assert result.completeness.total == 1

    async def test_aggregate_over_a_truncated_find_fails_closed(self, database) -> None:
        """THE gate, second half -- the two halves only matter together."""
        page = await find(database, SCHEMA, limit=2)
        pipeline = Pipeline("n", "page", (Stage("aggregate", {"op": "count"}),))
        results = run_pipelines((pipeline,), {"page": page})
        assert isinstance(results["n"], Failed)
        assert isinstance(results["n"].defect, DataDefect)
        assert "4" in results["n"].defect.message

    async def test_aggregate_over_a_complete_find_succeeds(self, database) -> None:
        whole = await find(database, SCHEMA, limit=100)
        pipeline = Pipeline("total", "whole", (
            Stage("aggregate", {"op": "sum", "path": Path.parse("credits")}),
        ))
        results = run_pipelines((pipeline,), {"whole": whole})
        assert isinstance(results["total"], Succeeded)
        assert results["total"].value.value == 13.0


class TestDeterminism:
    async def test_a_truncated_fetch_returns_the_same_page_every_time(self, database) -> None:
        """§4.3 one level up from argmax: without a stable order, the PAGE
        itself varies between runs and every downstream answer varies with it."""
        first = [r.fields["courseNumber"].value for r in (await find(database, SCHEMA, limit=2)).records]
        for _ in range(5):
            again = [r.fields["courseNumber"].value for r in (await find(database, SCHEMA, limit=2)).records]
            assert again == first


class TestFailClosed:
    async def test_a_record_missing_the_key_fails_the_whole_fetch(self, database) -> None:
        """An unresolvable key is the 7-dangling-courseId class. Admitting the
        record without its key would let a later difference silently retain it."""
        await database["courses"].insert_one({"credits": "1.0", "track": "cs"})
        result = await find(database, SCHEMA)
        assert isinstance(result, DataDefect)
        assert "courseNumber" in result.message

    async def test_aggregating_a_field_some_records_lack_fails_closed(self, database) -> None:
        """The other route to a silent partial. `grade` is absent on one course,
        so an average over it would be an average of three reported with the
        confidence of an average of four -- indistinguishable from correct."""
        whole = await find(database, SCHEMA, limit=100)
        pipeline = Pipeline("avg", "whole", (
            Stage("aggregate", {"op": "avg", "path": Path.parse("grade")}),
        ))
        results = run_pipelines((pipeline,), {"whole": whole})
        assert isinstance(results["avg"], Failed)
        assert isinstance(results["avg"].defect, DataDefect)
        assert "grade" in results["avg"].defect.message

    async def test_an_uncoercible_quantity_omits_the_field_rather_than_guessing(self, database) -> None:
        """Not fatal -- one dirty non-key value should not sink a whole fetch --
        but it must not become 0 either. Absent, so an aggregate over it fails
        closed downstream with a message that names the field."""
        await database["courses"].insert_one({"courseNumber": "00000001", "credits": "n/a", "track": "cs"})
        result = await find(database, SCHEMA)
        dirty = [r for r in result.records if r.fields["courseNumber"].value == "00000001"][0]
        assert "credits" not in dirty.fields


class TestDerivedSources:
    """A source whose records are computed rather than stored.

    `traverse` needs edges, and prerequisite edges are parsed out of catalog
    prose rather than stored anywhere -- so without this the tool was advertised
    with no reachable input. The contract must be the SAME contract: typed,
    ordered, honestly counted. A second admission path with looser rules would
    be a hole in the layer that exists to close holes.
    """

    EDGES = [
        {"edge": "c->b", "course": "c", "requires": "b", "group": "g1"},
        {"edge": "a->b", "course": "a", "requires": "b", "group": "g1"},
        {"edge": "b->x", "course": "b", "requires": "x", "group": "g2"},
    ]

    def _schema(self, documents=None):
        from app.agent_core.facts.find import DerivedSchema

        return DerivedSchema(
            collection="edges",
            key="edge",
            fields={"edge": I, "course": I, "requires": I, "group": I},
            basis=Basis.WIKI_DERIVED,
            produce=lambda: self.EDGES if documents is None else documents,
            yields=frozenset({"edges"}),
        )

    async def test_it_produces_typed_records(self) -> None:
        result = await find(None, self._schema())
        assert len(result.records) == 3
        assert result.records[0].fields["course"].kind is I

    async def test_records_carry_the_declared_basis(self) -> None:
        """Parsed from prose, so weaker than an official record -- and any answer
        built on it has to inherit that."""
        result = await find(None, self._schema())
        assert all(r.basis is Basis.WIKI_DERIVED for r in result.records)

    async def test_the_predicate_filters(self) -> None:
        result = await find(None, self._schema(), predicate=Comparison(Path.parse("course"), Op.EQ, Scalar(I, "a")))
        assert [r.fields["edge"].value for r in result.records] == ["a->b"]

    async def test_a_filtered_result_is_still_complete(self) -> None:
        result = await find(None, self._schema(), predicate=Comparison(Path.parse("course"), Op.EQ, Scalar(I, "a")))
        assert result.completeness.complete is True
        assert result.completeness.total == 1

    async def test_truncation_is_reported(self) -> None:
        result = await find(None, self._schema(), limit=2)
        assert len(result.records) == 2
        assert result.completeness.complete is False
        assert result.completeness.total == 3

    async def test_the_order_is_stable(self) -> None:
        """Same reason a stored fetch sorts: without it the PAGE varies between
        runs and every answer derived from it varies with it."""
        first = [r.fields["edge"].value for r in (await find(None, self._schema(), limit=2)).records]
        for _ in range(5):
            again = [r.fields["edge"].value for r in (await find(None, self._schema(), limit=2)).records]
            assert again == first
        assert first == ["a->b", "b->x"]

    async def test_an_unknown_field_is_rejected_naming_what_exists(self) -> None:
        result = await find(None, self._schema(), predicate=Comparison(Path.parse("nope"), Op.EQ, Scalar(I, "a")))
        assert isinstance(result, ExpressionDefect)
        assert "requires" in result.message

    async def test_a_record_without_a_key_fails_the_whole_fetch(self) -> None:
        """The same fail-closed rule as a stored fetch. A derived source that
        admitted keyless records would let a later difference drop them silently."""
        result = await find(None, self._schema([{"course": "a", "requires": "b"}]))
        assert isinstance(result, DataDefect)
        assert "edge" in result.message


class TestArraysOfPlainValues:
    """`ArrayOf(ScalarKind)` -- the other array shape a source may declare.

    Elements are named after the FIELD, not something generic, because `unnest`
    merges an element's fields into the parent: two arrays expanded in one
    pipeline would otherwise both arrive as `value` and the second would
    silently overwrite the first.
    """

    def _schema(self):
        from app.agent_core.facts.find import ArrayOf, DerivedSchema

        return DerivedSchema(
            collection="offered",
            key="courseNumber",
            fields={"courseNumber": I, "semestersOffered": ArrayOf(Q)},
            basis=Basis.OFFICIAL_RECORD,
            produce=lambda: [{"courseNumber": "00940224", "semestersOffered": [200, 201, 202]}],
        )

    async def test_elements_are_named_after_their_field(self) -> None:
        result = await find(None, self._schema())
        nested = result.records[0].fields["semestersOffered"]
        assert [r.fields["semestersOffered"].value for r in nested.records] == [200, 201, 202]

    async def test_unnesting_yields_one_record_per_value(self) -> None:
        fetched = await find(None, self._schema())
        pipeline = Pipeline("out", "src", (Stage("unnest", {"path": Path.parse("semestersOffered")}),))
        results = run_pipelines((pipeline,), {"src": fetched})
        assert isinstance(results["out"], Succeeded)
        assert [r.fields["semestersOffered"].value for r in results["out"].value.records] == [200, 201, 202]
        # And the parent identity rides along, so a placement can be attributed back.
        assert results["out"].value.records[0].fields["courseNumber"].value == "00940224"
