"""The Phase 5 gate, asked of Postgres: a truncated fetch reports
`complete=false`, and an `aggregate` over it fails closed.

Ported from `tests/pending_supabase/test_find.py`, which asked Mongo the same
questions against a throwaway database. There is no throwaway database here, so
this creates a real table, fills it with the same deliberately dirty rows, and
drops it. A prefixed name, never `courses`: the real catalog is 2,613 rows the
whole agent reads.

The dirty data is the point and survives the port unchanged. Every column is
TEXT, because that is the question `find` exists to answer -- "3.5" and
"00940224" are the same shape of string, and no heuristic separates a quantity
from an identifier. Only the declared schema can, which is what "convert at
admission, where the source schema is known" means. Postgres would happily have
typed `credits` as double precision and tested nothing.

What did NOT survive: nothing was dropped, but two tests changed shape. Mongo let
a document simply omit a field; a table cannot, so absence is NULL here, and the
distinction that matters -- absent is not zero -- is asserted the same way.
"""

from __future__ import annotations

import pytest

from app.agent_core.facts.find import SourceSchema, find
from app.agent_core.facts.operators import DataDefect, ExpressionDefect, Pipeline, Stage
from app.agent_core.facts.predicate import Comparison, Op, Path
from app.agent_core.facts.runner import Failed, Succeeded, run_pipelines
from app.agent_core.facts.types import Basis, Scalar, ScalarKind
from app.db.postgres import get_database

pytestmark = pytest.mark.supabase

Q = ScalarKind.QUANTITY
I = ScalarKind.IDENTIFIER

TABLE = "pytest_find_courses"

# Credits arrive as STRINGS, codes are numeric-looking strings with leading
# zeros, and one row has no grade at all.
ROWS = [
    ("00940224", "3.5", "95", "cs"),
    ("00960211", "3.0", "60", "cs"),
    ("00970800", "4.0", "88", "ee"),
    ("00940594", "2.5", None, "cs"),  # no grade
]

SCHEMA = SourceSchema(
    collection=TABLE,
    key="courseNumber",
    fields={"courseNumber": I, "credits": Q, "grade": Q, "track": I},
    basis=Basis.OFFICIAL_RECORD,
)


@pytest.fixture
async def database():
    db = await get_database()
    await db.execute(f'drop table if exists "{TABLE}"')
    await db.execute(
        f'create table "{TABLE}" ('
        '"courseNumber" text, "credits" text, "grade" text, "track" text)'
    )
    for row in ROWS:
        await db.execute(
            f'insert into "{TABLE}" ("courseNumber", "credits", "grade", "track") '
            "values ($1, $2, $3, $4)",
            *row,
        )
    yield db
    await db.execute(f'drop table if exists "{TABLE}"')


class TestTyping:
    async def test_a_numeric_string_becomes_a_quantity_when_the_schema_says_so(
        self, database
    ) -> None:
        """`Scalar(QUANTITY, "3.5")` is refused everywhere else on purpose. Here
        the conversion is allowed, because the kind is DECLARED rather than
        guessed."""
        result = await find(database, SCHEMA)
        credits = result.records[0].fields["credits"]
        assert credits.kind is Q and credits.value == 3.5

    async def test_a_course_code_stays_an_identifier(self, database) -> None:
        """Same shape of string, opposite kind. No heuristic could separate
        these -- the leading zero is a convention, not a type."""
        code = (await find(database, SCHEMA)).records[0].fields["courseNumber"]
        assert code.kind is I and code.value == "00940224"

    async def test_an_absent_field_is_absent_rather_than_defaulted(self, database) -> None:
        """A row with no grade must not acquire one. Defaulting to 0 would make
        an average silently wrong -- and wrong in the direction that looks
        plausible."""
        result = await find(database, SCHEMA)
        no_grade = [r for r in result.records if r.fields["courseNumber"].value == "00940594"][0]
        assert "grade" not in no_grade.fields

    async def test_records_carry_the_declared_basis(self, database) -> None:
        result = await find(database, SCHEMA)
        assert all(r.basis is Basis.OFFICIAL_RECORD for r in result.records)


class TestPushDown:
    async def test_the_predicate_filters_at_the_source(self, database) -> None:
        result = await find(
            database, SCHEMA, predicate=Comparison(Path.parse("track"), Op.EQ, Scalar(I, "cs"))
        )
        assert sorted(r.fields["courseNumber"].value for r in result.records) == [
            "00940224",
            "00940594",
            "00960211",
        ]

    async def test_an_unknown_field_is_rejected_naming_what_exists(self, database) -> None:
        result = await find(
            database, SCHEMA, predicate=Comparison(Path.parse("deficit"), Op.GT, Scalar(Q, 0))
        )
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

    async def test_completeness_is_measured_against_the_PREDICATE_not_the_collection(
        self, database
    ) -> None:
        """A filtered fetch is complete when it holds every MATCHING record, even
        though it holds fewer than the table. Counting against the whole table
        would mark every filtered result incomplete and make aggregates
        permanently impossible."""
        result = await find(
            database, SCHEMA, predicate=Comparison(Path.parse("track"), Op.EQ, Scalar(I, "ee"))
        )
        assert result.completeness.complete is True
        assert result.completeness.total == 1

    async def test_aggregate_over_a_truncated_find_fails_closed(self, database) -> None:
        """THE gate, second half. The two halves only matter together: reporting
        incompleteness that nothing acts on is a comment, not a guarantee."""
        page = await find(database, SCHEMA, limit=2)
        pipeline = Pipeline("n", "page", (Stage("aggregate", {"op": "count"}),))
        results = run_pipelines((pipeline,), {"page": page})
        assert isinstance(results["n"], Failed)
        assert isinstance(results["n"].defect, DataDefect)
        assert "4" in results["n"].defect.message, "the refusal must name the true total"

    async def test_aggregate_over_a_complete_find_succeeds(self, database) -> None:
        whole = await find(database, SCHEMA, limit=100)
        pipeline = Pipeline(
            "total", "whole", (Stage("aggregate", {"op": "sum", "path": Path.parse("credits")}),)
        )
        results = run_pipelines((pipeline,), {"whole": whole})
        assert isinstance(results["total"], Succeeded)
        assert results["total"].value.value == 13.0


class TestDeterminism:
    async def test_a_truncated_fetch_returns_the_same_page_every_time(self, database) -> None:
        """Without a stable order the PAGE itself varies between runs, and every
        downstream answer varies with it. Postgres makes this sharper than Mongo
        did: an unordered SELECT may legitimately return rows in any order, and
        is free to change its mind between calls."""
        first = [
            r.fields["courseNumber"].value for r in (await find(database, SCHEMA, limit=2)).records
        ]
        for _ in range(5):
            again = [
                r.fields["courseNumber"].value
                for r in (await find(database, SCHEMA, limit=2)).records
            ]
            assert again == first


class TestFailClosed:
    async def test_a_record_missing_the_key_fails_the_whole_fetch(self, database) -> None:
        """An unresolvable key is the dangling-courseId class. Admitting the
        record without its key would let a later difference silently retain it."""
        await database.execute(
            f'insert into "{TABLE}" ("credits", "track") values ($1, $2)', "1.0", "cs"
        )
        result = await find(database, SCHEMA)
        assert isinstance(result, DataDefect)
        assert "courseNumber" in result.message

    async def test_aggregating_a_field_some_records_lack_fails_closed(self, database) -> None:
        """The other route to a silent partial. `grade` is absent on one course,
        so an average over it would be an average of three reported with the
        confidence of an average of four -- indistinguishable from correct."""
        whole = await find(database, SCHEMA, limit=100)
        pipeline = Pipeline(
            "avg", "whole", (Stage("aggregate", {"op": "avg", "path": Path.parse("grade")}),)
        )
        results = run_pipelines((pipeline,), {"whole": whole})
        assert isinstance(results["avg"], Failed)
        assert isinstance(results["avg"].defect, DataDefect)
        assert "grade" in results["avg"].defect.message

    async def test_an_uncoercible_quantity_omits_the_field_rather_than_guessing(
        self, database
    ) -> None:
        """Not fatal -- one dirty non-key value should not sink a whole fetch --
        but it must not become 0 either. Absent, so an aggregate over it fails
        closed downstream with a message that names the field."""
        await database.execute(
            f'insert into "{TABLE}" ("courseNumber", "credits", "track") values ($1, $2, $3)',
            "00000001",
            "n/a",
            "cs",
        )
        result = await find(database, SCHEMA)
        dirty = [r for r in result.records if r.fields["courseNumber"].value == "00000001"][0]
        assert "credits" not in dirty.fields
